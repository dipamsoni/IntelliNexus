# app.py
import streamlit as st # Keep this early
import os
import json # For persistent chat history
import shutil # To potentially delete directory
import tools # For tool descriptions, accessing DOC_PATH, and init function call might be via this module path
import core_initializer # For central initialization
import logging # Should be configured early
from llm_interface import MODEL_NAME # For display in UI

# --- Move st.set_page_config() HERE as the first Streamlit command ---
st.set_page_config(layout="wide", page_title="Enhanced ReAct Agent")

# --- Basic Logging Setup (after set_page_config) ---
# Configure logging - This setup should apply to imported modules like tools, core_initializer too
logging.basicConfig(
    level=logging.INFO, # Adjust level as needed (DEBUG, INFO, WARNING, ERROR)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Output logs to console/stderr
    ]
)
logger = logging.getLogger(__name__) # Logger for this app.py module

logger.info("--- app.py: SCRIPT EXECUTION STARTED (after page_config) ---")

# --- Constants ---
CHAT_HISTORY_FILE = "persistent_chat_history.json"

# --- Initialization State Flags ---
# Check/Initialize Session State variables AFTER set_page_config
if 'systems_fully_initialized_flag' not in st.session_state:
    st.session_state.systems_fully_initialized_flag = False

# --- Persistent Chat History Loading (AFTER set_page_config) ---
# Define a variable to hold any early error message from history loading
chat_history_load_error_message = None

if 'chat_history' not in st.session_state:
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            if os.path.isdir(CHAT_HISTORY_FILE):
                # This path exists but IS a directory, which is wrong.
                logger.error(f"CRITICAL: '{CHAT_HISTORY_FILE}' is a directory, not a file. Please delete it manually and restart.")
                # Provide a clear message to the user in the UI
                chat_history_load_error_message = (
                    f"**Error:** The path for chat history (`{CHAT_HISTORY_FILE}`) "
                    f"exists but is a directory, not a file. Please manually delete "
                    f"this directory and restart the application."
                )
                st.session_state.chat_history = [] # Start fresh in this session
            else:
                # Path exists and is not a directory, try to read it as a file.
                with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                    st.session_state.chat_history = json.load(f)
                logger.info(f"Loaded chat history from {CHAT_HISTORY_FILE}")
        else:
            # File does not exist, start fresh
            st.session_state.chat_history = []
            logger.info(f"No persistent chat history file found ({CHAT_HISTORY_FILE}), starting fresh.")
    except PermissionError as pe:
         logger.error(f"Permission Error loading chat history from {CHAT_HISTORY_FILE}: {pe}. Starting fresh.", exc_info=True)
         chat_history_load_error_message = (
             f"**Warning:** Could not load previous chat history due to permission error "
             f"(`{CHAT_HISTORY_FILE}`). Please check file/folder permissions. Starting with a fresh history."
         )
         st.session_state.chat_history = [] # Ensure it's initialized
    except json.JSONDecodeError as je:
         logger.error(f"Error decoding JSON loading chat history from {CHAT_HISTORY_FILE}: {je}. File might be corrupt. Starting fresh.", exc_info=True)
         chat_history_load_error_message = (
             f"**Warning:** Could not parse previous chat history (`{CHAT_HISTORY_FILE}`). "
             f"The file might be corrupt. Starting with a fresh history."
         )
         st.session_state.chat_history = [] # Ensure it's initialized
    except Exception as e:
        # Catch other potential errors during loading
        logger.error(f"Unexpected Error loading chat history from {CHAT_HISTORY_FILE}: {e}. Starting fresh.", exc_info=True)
        chat_history_load_error_message = (
             f"**Warning:** An unexpected error occurred while loading previous chat history (`{CHAT_HISTORY_FILE}`). "
             f"Error: {e}. Starting with a fresh history."
         )
        st.session_state.chat_history = [] # Ensure it's initialized

# --- Display any crucial early error messages (AFTER set_page_config) ---
if chat_history_load_error_message:
    st.warning(chat_history_load_error_message) # Display the warning/error if loading failed


# --- Helper function to check for required files ---
# Can now contain st.error as set_page_config was called
def check_required_files():
    """Checks for essential data files and displays an error if any are missing."""
    missing_files = []
    try:
        doc_path_to_check = tools.DOC_PATH # Get path from tools module
        if not os.path.exists(doc_path_to_check):
            missing_files.append(f"{doc_path_to_check} (Policy Document)")
    except AttributeError:
        missing_files.append(f"Configuration error: tools.DOC_PATH not defined.")
        logger.error("AttributeError: tools.DOC_PATH not found during file check.")

    if not os.path.exists("report.csv"):
        missing_files.append("report.csv (Sales Data)")

    if missing_files:
        st.error(
            f"**Required Data Files Missing (expected in/relative to '{os.getcwd()}'):**\n\n" +
            "\n".join([f"- {mf}" for mf in missing_files]) +
            "\n\nPlease create them or ensure they are in the correct location and paths are correctly configured."
        )
        logger.error(f"Missing required files check failed: {', '.join(missing_files)}")
        return False
    logger.info("Required files check passed.")
    return True

# --- Page Title & Markdown (Display Elements - Fine Here) ---
st.title("🧠 Enhanced ReAct Agent (Self-Reflection, Persistent History, LangChain QA)")
logger.info("--- app.py: Streamlit UI Title Rendered ---")


# --- Perform File Check and Stop if Necessary ---
if not check_required_files():
    st.stop()

# --- Initialize Core Systems (Agent, QA Engine via core_initializer) ---
if not st.session_state.systems_fully_initialized_flag:
    logger.info("--- app.py: Core systems not yet initialized, calling core_initializer.initialize_all_systems() ---")
    # Display spinner during potentially long initialization
    with st.spinner("Initializing agent and Q&A engine... This may take time (esp. first FAISS index creation)."):
        if core_initializer.initialize_all_systems():
            st.session_state.systems_fully_initialized_flag = True
            # Log success, message displayed later to avoid being overwritten
            logger.info("--- app.py: Core systems initialization reported success. ---")
        else:
            st.error("Critical backend systems failed to initialize during startup. Please check console logs for details. The application might not function correctly.")
            logger.error("--- app.py: Core systems initialization reported failure. ---")
            # st.session_state.systems_fully_initialized_flag remains False


# --- Final Check and Stop/Success Message ---
agent_ready = st.session_state.systems_fully_initialized_flag and core_initializer.GLOBAL_AGENT_INSTANCE is not None
if agent_ready:
    st.success("Agent and supporting systems are ready!")
    st.markdown(f"""
    Using **{MODEL_NAME}** (ReAct LLM) & LangChain. Policy Doc: `{tools.DOC_PATH}`. Report: `report.csv`. Embed Model: `{tools.EMBEDDING_MODEL_NAME}`.
    Chat history uses `{CHAT_HISTORY_FILE}`. Agent includes self-reflection logic.
    """)
else:
    st.error("Core systems could not be initialized properly. Please check error messages above and detailed console logs. Ensure required services like Ollama are running.")
    st.stop() # Stop further app execution

# --- Sidebar ---
with st.sidebar:
    st.header("Agent Controls")
    if st.button("Clear Chat & Agent Memory"):
        st.session_state.chat_history = [] # Clear in-memory history
        if os.path.exists(CHAT_HISTORY_FILE):
            try:
                # Handle file or directory for deletion robustly
                if os.path.isfile(CHAT_HISTORY_FILE):
                     os.remove(CHAT_HISTORY_FILE)
                     logger.info(f"Deleted persistent chat history file: {CHAT_HISTORY_FILE}")
                elif os.path.isdir(CHAT_HISTORY_FILE):
                     shutil.rmtree(CHAT_HISTORY_FILE) # Use shutil to remove directory
                     logger.info(f"Deleted persistent chat history directory: {CHAT_HISTORY_FILE}")
            except Exception as e:
                logger.error(f"Error deleting persistent chat history path '{CHAT_HISTORY_FILE}': {e}")
                st.warning(f"Could not delete persistent chat history at '{CHAT_HISTORY_FILE}'.")

        if core_initializer.GLOBAL_AGENT_INSTANCE:
            core_initializer.GLOBAL_AGENT_INSTANCE.error_memory.clear()
        st.success("In-memory chat history cleared & agent error memory reset. Attempted to delete persistent history path.")
        # Note: FAISS index on disk is not touched by this reset.
        st.rerun() # Rerun to reflect cleared history

    st.header("Agent Information")
    st.markdown(f"**ReAct LLM (Ollama):** `{MODEL_NAME}`")
    # Use try-except for tool attributes in case tools didn't init fully
    try:
        st.markdown(f"**Document QA LLM (Ollama):** `{tools.QA_LLM_MODEL}`")
        st.markdown(f"**Embedding Model (Ollama):** `{tools.EMBEDDING_MODEL_NAME}`")
    except AttributeError:
         st.markdown("**Model names:** *(unavailable)*")

    st.markdown("**Available Tools:**")
    try:
        if tools.TOOLS_AVAILABLE:
            for tool_name, tool_instance in tools.TOOLS_AVAILABLE.items():
                st.markdown(f"- `{tool_name}`: {tool_instance.description}")
        else:
            st.markdown("*Tools list unavailable.*")
    except AttributeError:
         st.markdown("*Tools configuration error.*")

    st.markdown("---")
    st.caption(f"Data files should be in/relative to: `{os.getcwd()}`.")
    try:
         st.caption(f"Using Policy Document: `{tools.DOC_PATH}`")
         st.caption(f"Using FAISS Index: `{tools.FAISS_INDEX_PATH}`")
    except AttributeError:
        st.caption("*(Configuration paths unavailable)*")


# --- Main Chat Interface ---
st.header("Chat with the Agent")

# Display chat history (already loaded or initialized into session_state)
# Add safety check for chat history structure
for i, chat_item in enumerate(st.session_state.chat_history):
    if isinstance(chat_item, (list, tuple)) and len(chat_item) >= 2:
        query, response = chat_item[0], chat_item[1]
        trace_html = chat_item[2] if len(chat_item) > 2 else None # Optional trace
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(response, unsafe_allow_html=True)
            if trace_html:
                with st.expander(f"Reasoning Trace for Query {i+1}"):
                    st.markdown(trace_html, unsafe_allow_html=True)
    else:
        logger.warning(f"Skipping display of unexpected chat history item format at index {i}: {type(chat_item)}")
        st.warning(f"Skipping display of malformed chat history item {i}.")


user_query = st.chat_input("Ask the agent a question...")

if user_query:
    if not core_initializer.GLOBAL_AGENT_INSTANCE:
        st.error("Agent is not available. Cannot process query.")
        st.stop()

    st.chat_message("user").markdown(user_query)
    with st.spinner(f"🤖 The agent is processing your request..."):
        context_for_llm_str = ""
        if st.session_state.chat_history:
            history_for_prompt = []
            max_turns_for_llm_context = 3 # How many past Q/A pairs to include in prompt
            num_entries = len(st.session_state.chat_history)
            start_index = max(0, num_entries - max_turns_for_llm_context)

            # Select and format recent history for the LLM prompt
            for item in st.session_state.chat_history[start_index:]:
                 if isinstance(item, (list, tuple)) and len(item) >= 2:
                      q, a = item[0], item[1]
                      history_for_prompt.append(f"User: {q}\nAgent: {a}")

            if history_for_prompt:
                context_for_llm_str = "Recent Conversation History (chronological order):\n" + "\n\n".join(history_for_prompt)


        final_answer, reasoning_trace_html = core_initializer.GLOBAL_AGENT_INSTANCE.run(
            user_query,
            conversation_history_str=context_for_llm_str
        )

        # Append to in-memory history (essential for display)
        st.session_state.chat_history.append([user_query, final_answer, reasoning_trace_html])

        # Save updated history to persistent file
        try:
            with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(st.session_state.chat_history, f, indent=2)
            logger.info(f"Saved chat history update to {CHAT_HISTORY_FILE}")
        except PermissionError as pe:
             logger.error(f"Permission error saving chat history update to {CHAT_HISTORY_FILE}: {pe}")
             st.warning("Could not save chat history update persistently due to permissions.")
        except Exception as e:
            logger.error(f"Error saving chat history update to {CHAT_HISTORY_FILE}: {e}", exc_info=True)
            st.warning(f"Could not save chat history update persistently: {e}")

        # Display the new response in the UI using st.rerun() to refresh chat display
        # st.experimental_rerun() Deprecated
        st.rerun() # Preferred way to refresh UI after state changes like appending to chat history


# Add a placeholder at the end if needed, e.g., for layout
# st.empty()