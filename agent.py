# agent.py
import re
import logging
from llm_interface import query_ollama, MODEL_NAME as REACT_LLM_MODEL_NAME
from tools import TOOLS_AVAILABLE, get_tool_descriptions_for_prompt

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 15
ERROR_MEMORY_LIMIT = 3

# --- REVISED REACT PROMPT TEMPLATE (Self-Reflection, Tool Fallback) ---
REACT_PROMPT_TEMPLATE = """
You are a highly efficient, precise, and self-critical AI assistant using the ReAct framework. Your goal is to answer user questions accurately and robustly, minimizing errors and iterations.

**ReAct Process & Strict Rules:**

1.  **Thought**:
    *   Analyze the **Current User Question** and **Conversation History**.
    *   **Decision Criteria for Tools:**
        *   **Policy Document Questions:** If the question is about employee leave policy, company policy, or specifically mentions 'policy.docx', your PRIMARY and MOST DIRECT action is `PolicyDocumentQA[user's full question about policy]`.
        *   **CSV Data/Numerical Questions:** If the question refers to 'report.csv', sales data, or requires calculations based on structured data:
            1.  If 'report.csv' has not been read or its relevant data (e.g., revenue list) is not in recent observations, use `Action: FileReaderTool[report.csv]`.
            2.  Analyze the Observation from `FileReaderTool`. If it provides numerical lists (e.g., "Revenue values: [10000, 12000, 15000]"), use `Action: CalculatorTool[[10000, 12000, 15000]]` or `CalculatorTool[10000+12000+15000]` if the question asks for a sum or other calculation on these *specific* numbers.
        *   **General Calculation:** If the question is a direct math problem not tied to a file, use `Action: CalculatorTool[mathematical expression]`.
    *   **Formulate Input Precisely:** The input to tools must be exactly what the tool expects (e.g., full question for `PolicyDocumentQA`, filename for `FileReaderTool`, expression/list for `CalculatorTool`).
    *   **Goal:** Get to the `Final Answer` using the fewest, most effective tool calls. Avoid guessing.

2.  **Action** (If a tool is deemed necessary):
    *   Write EXACTLY ONE `Action:` line: `Action: ToolName[input]`.
    *   Examples:
        *   `Action: PolicyDocumentQA[What is the new rule for leave carry-over and its deadline?]`
        *   `Action: FileReaderTool[report.csv]`
        *   `Action: CalculatorTool[[10000, 12000, 15000]]`
        *   `Action: CalculatorTool[5 * (3 + 2)]`

3.  **Observation** (System-provided output from a tool):
    *   This is your PRIMARY source of information. Analyze it critically.
    *   If it's an answer from `PolicyDocumentQA` or `CalculatorTool`, it's likely ready for `Final Answer`.
    *   If it's from `FileReaderTool` for a CSV, it guides your next `CalculatorTool` action if needed.
    *   If an error occurs, note it from "Recent Errors" and this observation. Do NOT repeat the failing action. Re-evaluate.

4.  **Final Answer** (When the question is fully answered based on tool observations):
    *   `Final Answer: [Your concise answer, based DIRECTLY and verifiable from the tool's `Observation`. If `PolicyDocumentQA` indicates information is not found, state that. For calculations, provide the numerical result.]`
    *   Do NOT invent or infer beyond the observation. If the tool observation is the answer, often you can just rephrase it slightly or state it directly.

**Available Tools:**
{tool_descriptions}

**Conversation History (For contextual understanding of multi-turn queries):**
{conversation_history}

**Current User Question:** {user_question}

**Recent Errors (AVOID these specific errors/patterns. Explain in Thought how you are avoiding them):**
{recent_errors}

---
Start with:
Thought: [My detailed analysis of the question leads to goal X. The most direct and self-checked first step is Action Y with input Z (or if I can answer from history/observation: Final Answer). I will avoid previous error E by taking corrective measure F.]
[Follow with EITHER `Action: ToolName[input]` OR `Final Answer: ...`]
---
"""

class ReActAgent:
    def __init__(self):
        self.tools = TOOLS_AVAILABLE
        self.tool_descriptions = get_tool_descriptions_for_prompt()
        self.current_file_content_cache = {} # Still potentially useful for FileReaderTool outputs if needed by LLM's next thought for Calculator
        self.error_memory = []
        logger.info("ReActAgent instance created with self-reflection and fallback guidance.")

    def _add_error_to_memory(self, error_message):
        # Sanitize error message for prompt
        error_message = re.sub(r'\n+', ' ', error_message) # Replace newlines with spaces
        error_message = error_message[:200] # Truncate long error messages
        if len(self.error_memory) >= ERROR_MEMORY_LIMIT:
            self.error_memory.pop(0)
        self.error_memory.append(error_message)

    def _format_recent_errors(self):
        return "\n".join([f"- {err}" for err in self.error_memory[-ERROR_MEMORY_LIMIT:]]) or "None"

    def run(self, user_question: str, conversation_history_str: str = "") -> tuple[str, str]:
        reasoning_trace_list = [f"**User Question:** {user_question}\n"]

        current_interaction_log = REACT_PROMPT_TEMPLATE.format(
            tool_descriptions=self.tool_descriptions,
            conversation_history=conversation_history_str or "No previous conversation.",
            user_question=user_question,
            recent_errors=self._format_recent_errors()
        )
        last_action = None # To detect immediate exact repeats

        for i in range(MAX_ITERATIONS):
            reasoning_trace_list.append(f"\n**--- Iteration {i+1}/{MAX_ITERATIONS} ---**")
            logger.debug(f"ReAct Iteration {i+1}/{MAX_ITERATIONS} for question: '{user_question[:50]}...'")
            
            llm_generated_text = query_ollama(current_interaction_log, model_name=REACT_LLM_MODEL_NAME)

            if not llm_generated_text or llm_generated_text.startswith("ERROR_OLLAMA"):
                error_message = f"LLM Communication Error (ReAct): {llm_generated_text}"
                logger.error(error_message)
                # self._add_error_to_memory(error_message) # Don't add LLM errors to "Recent Errors" for prompt
                reasoning_trace_list.append(f"<span style='color:red;'>{error_message}</span>")
                return "Agent Error: LLM communication failed for ReAct logic.", "".join(reasoning_trace_list)

            current_interaction_log += llm_generated_text # Append LLM's Thought/Action/FinalAnswer
            reasoning_trace_list.append(f"**LLM Output (Raw):**\n```text\n{llm_generated_text}\n```\n")

            # Check for Final Answer
            final_answer_match = re.search(
                r"Final Answer\s*:\s*(.*?)(?=\n\s*(Thought|Action|Observation|$)|\Z)",
                llm_generated_text,
                re.DOTALL | re.IGNORECASE
            )
            if final_answer_match:
                final_answer = final_answer_match.group(1).strip()
                # Placeholder check (allow if answer explicitly says "not found" or similar)
                if re.search(r"[\[<][^>\]]+[\]>]", final_answer) and \
                   not any(kw in final_answer.lower() for kw in ["not found", "not contain", "unable to find", "document does not state"]):
                    observation_for_llm = (
                        "Error: Final Answer seems to contain placeholders (e.g., '[details]'). "
                        "Provide specific values/details directly from observations, or explicitly state if information was not found after diligent search."
                    )
                    self._add_error_to_memory("Final Answer rejected due to placeholders.") # Generic error for memory
                    reasoning_trace_list.append(
                        f"<span style='color:red;'>Rejected Final Answer: `{final_answer}` (contains placeholders without clear 'not found' statement).</span>\n"
                    )
                    current_interaction_log += f"\nObservation: {observation_for_llm}\nThought:"
                    continue
                logger.info(f"Final Answer extracted by ReAct agent: {final_answer}")
                reasoning_trace_list.append(f"**Final Answer Extracted:** `{final_answer}`")
                return final_answer, "".join(reasoning_trace_list)

            # Parse Action (take the first valid one)
            action_lines = re.findall(
                r"^\s*Action\s*:\s*([a-zA-Z0-9_]+)\s*\[(.*?)]\s*$", # Input `(.*?)` captures content within brackets
                llm_generated_text,
                re.MULTILINE | re.IGNORECASE
            )

            tool_name_to_execute = None
            tool_input_to_execute = None # Can be empty string
            parsed_action_for_trace = "<span style='color:grey;'>No valid action parsed this turn.</span>"
            observation_for_llm = "" # This will hold the tool's output or an error

            if action_lines:
                tn_candidate, ti_candidate = action_lines[0]
                tn_candidate = tn_candidate.strip()
                # ti_candidate needs careful handling if it was parsed with DOTALL for `Action: Tool[Input spans
                # lines]`. For `MULTILINE` without DOTALL on the input part, it shouldn't span.
                # Assuming `ti_candidate` is the content between brackets on that line.
                ti_candidate = ti_candidate.strip() # Input between brackets

                if tn_candidate.lower() in ["toolname", "input", "actiontool"]:
                    observation_for_llm = f"Error: LLM provided a placeholder tool name '{tn_candidate}'. Use an actual tool name from 'Available Tools'."
                else:
                    tool_name_to_execute = tn_candidate
                    tool_input_to_execute = ti_candidate
                    parsed_action_for_trace = f"`{tool_name_to_execute}[{tool_input_to_execute}]`"
            
            if observation_for_llm: # e.g., placeholder tool name error
                self._add_error_to_memory(observation_for_llm)
                reasoning_trace_list.append(f"<span style='color:red;'>Invalid Action Parsed: {observation_for_llm}</span>\n")
                current_interaction_log += f"\nObservation: {observation_for_llm}\nThought:"
                continue # Loop to next LLM call with this error as observation

            reasoning_trace_list.append(f"**Action Parsed:** {parsed_action_for_trace}\n")

            current_action_str = f"{tool_name_to_execute}[{tool_input_to_execute}]" if tool_name_to_execute else "None"
            if tool_name_to_execute and current_action_str == last_action:
                observation_for_llm = "Error: The exact same action and input were repeated. This indicates a loop. You MUST change your plan: try different input, a different tool, or provide a Final Answer if no other options."
                # This specific error is critical for the LLM to see to break loops
            last_action = current_action_str

            if observation_for_llm: # e.g. repeated action error
                 self._add_error_to_memory(observation_for_llm) # Add this critical error
                 # Fall through to append this observation and get next thought
            elif tool_name_to_execute and tool_input_to_execute is not None:
                if tool_name_to_execute in self.tools:
                    tool_to_use = self.tools[tool_name_to_execute]
                    try:
                        logger.info(f"Executing tool: {tool_name_to_execute} with input: '{str(tool_input_to_execute)[:100]}...'")
                        observation_from_tool = tool_to_use.execute(str(tool_input_to_execute)) # Ensure input is str
                        observation_for_llm = observation_from_tool
                        logger.debug(f"Tool {tool_name_to_execute} observation: {str(observation_for_llm)[:200]}...")
                        # If observation is an error from the tool, add it to error memory
                        if "Error:" in observation_from_tool: # Basic check
                             self._add_error_to_memory(f"Tool '{tool_name_to_execute}' reported: {observation_from_tool}")
                    except Exception as e:
                        logger.exception(f"Agent: Tool Execution Error for {tool_name_to_execute}:")
                        tool_exec_error = f"Agent-level Error executing {tool_name_to_execute}: {type(e).__name__} - {str(e)}"
                        observation_for_llm = tool_exec_error
                        self._add_error_to_memory(tool_exec_error)
                else:
                    unknown_tool_error = f"Error: Unknown tool '{tool_name_to_execute}'. Valid tools are: {', '.join(self.tools.keys())}."
                    observation_for_llm = unknown_tool_error
                    self._add_error_to_memory(unknown_tool_error)
            else: # No tool name/input parsed, and not a Final Answer, and no prior error this turn
                if "Final Answer:" not in llm_generated_text: # Should not happen if parsing logic is right
                    no_action_error = "Error: LLM did not provide a `Final Answer:` or a valid `Action: ToolName[input]`. One is required."
                    observation_for_llm = no_action_error
                    self._add_error_to_memory(no_action_error)

            # Append Observation to interaction log and trace
            if observation_for_llm:
                current_interaction_log += f"\nObservation: {observation_for_llm}"
                color = "red" if "Error:" in observation_for_llm else ("orange" if "Warning:" in observation_for_llm or "Could not find" in observation_for_llm else "green")
                reasoning_trace_list.append(
                    f"**Observation:** <span style='color:{color};'>{observation_for_llm}</span>\n"
                )
            else: # Only if LLM only gave a Thought and nothing else (should be rare with current prompt and stop tokens)
                no_obs_msg = "No specific observation was generated by a tool or error condition this turn. LLM should proceed."
                current_interaction_log += f"\nObservation: {no_obs_msg}" # Still need an Observation line
                reasoning_trace_list.append(f"**Observation:** <span style='color:grey;'>{no_obs_msg}</span>\n")

            current_interaction_log += "\nThought:" # Prompt LLM for the next thought

            if len(current_interaction_log) > 7500: # Slightly increased context check
                reasoning_trace_list.append(
                    "<span style='color:orange;'>Warning: Interaction log is very long, approaching context limit. Aim for Final Answer.</span>\n"
                )
        
        # Max iterations reached
        max_iter_msg = (
            f"Agent Error: Max iterations ({MAX_ITERATIONS}) reached. Unable to complete the request. "
            "This may be due to a complex query, the LLM getting stuck, or a tool repeatedly failing. "
            "Please review the reasoning trace. You could try rephrasing the question or breaking it into smaller parts."
        )
        logger.warning(f"Max iterations reached for user question: '{user_question[:50]}...'")
        reasoning_trace_list.append(f"<br><span style='color:red; font-weight:bold;'>{max_iter_msg}</span>")
        return max_iter_msg, "".join(reasoning_trace_list)