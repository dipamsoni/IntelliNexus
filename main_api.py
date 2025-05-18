# main_api.py
from fastapi import FastAPI, HTTPException, Body, status
from pydantic import BaseModel
import logging
import uvicorn
import os # For OLLAMA_BASE_URL later if needed directly here

# Centralized initialization for QA system and Agent
import core_initializer
# The following are implicitly used by core_initializer but good to be aware of
# import tools
# from agent import ReActAgent

# Configure logging for the API - must be done before other modules that use logging
logging.basicConfig(
    level=logging.INFO, # Or DEBUG for more verbosity
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Output to console, Docker will catch this
    ]
)
logger = logging.getLogger(__name__) # Logger for this main_api.py module

app = FastAPI(
    title="ReAct Agent API",
    description="API interface for the ReAct Agent with Document Q&A capabilities.",
    version="0.1.0"
)

# --- API State ---
# core_initializer.initialize_all_systems() will set up core_initializer.GLOBAL_AGENT_INSTANCE
# and tools.POLICY_QA_CHAIN
# We just need a flag to indicate if startup initialization was successful overall.
api_systems_ready = False

@app.on_event("startup")
async def startup_event():
    """
    Event handler for API startup. Initializes all core systems.
    """
    global api_systems_ready
    logger.info("API application starting up...")
    logger.info("Calling core_initializer.initialize_all_systems() to set up backend systems...")
    if core_initializer.initialize_all_systems():
        api_systems_ready = True
        logger.info("API: Core systems initialized successfully and are ready.")
    else:
        api_systems_ready = False
        logger.error("API: CRITICAL - Core systems FAILED to initialize during startup. API may not function correctly.")
        # Depending on requirements, you might want to prevent FastAPI from fully starting
        # or make health check fail prominently. For now, it will start but endpoints will check `api_systems_ready`.

# --- Pydantic Models for Request and Response ---
class ChatQuery(BaseModel):
    user_query: str
    # Client is responsible for managing and sending relevant conversation history.
    # The API itself can be stateless regarding conversation turns.
    conversation_history_str: str | None = ""

class ChatResponse(BaseModel):
    final_answer: str
    reasoning_trace_html: str | None = None # Optional: for debugging or advanced clients

class HealthStatus(BaseModel):
    status: str
    message: str
    qa_system_status: str | None = None
    agent_status: str | None = None

# --- API Endpoints ---
@app.post("/chat",
          response_model=ChatResponse,
          summary="Process a user query with the ReAct agent",
          status_code=status.HTTP_200_OK)
async def chat_with_agent_endpoint(query: ChatQuery = Body(...)):
    """
    Accepts a user query and optional conversation history.
    The agent processes the query using its ReAct loop and tools (including Document QA).
    Returns the final answer and an optional HTML reasoning trace.
    """
    if not api_systems_ready or core_initializer.GLOBAL_AGENT_INSTANCE is None:
        logger.error("/chat endpoint called but systems are not ready.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent backend systems are not ready or failed to initialize. Please try again later or check server logs."
        )

    # More specific check for PolicyDocumentQA readiness (optional, as agent should handle tool errors)
    import tools # Ensure module is accessible
    if not tools.POLICY_QA_CHAIN or \
       (isinstance(tools.POLICY_QA_CHAIN, str) and tools.POLICY_QA_CHAIN.startswith("ERROR:")):
        logger.warning("/chat: PolicyDocumentQA tool appears to be in an error state. Query may be affected if it needs this tool.")
        # Could return a specific warning in the response or let the agent try and report tool failure.

    logger.info(f"API /chat endpoint: Received query (first 100 chars): '{query.user_query[:100]}...'")
    try:
        # Use the globally initialized agent instance from core_initializer
        final_answer, reasoning_trace_html = core_initializer.GLOBAL_AGENT_INSTANCE.run(
            query.user_query,
            conversation_history_str=query.conversation_history_str or ""
        )
        logger.info(f"API /chat endpoint: Agent processed query. Answer (first 100 chars): '{final_answer[:100]}...'")
        return ChatResponse(final_answer=final_answer, reasoning_trace_html=reasoning_trace_html)
    except Exception as e:
        logger.exception(f"API /chat endpoint: An unexpected error occurred during agent execution for query: '{query.user_query[:100]}'")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred while processing your request: {str(e)}"
        )

@app.get("/health",
         response_model=HealthStatus,
         summary="Check the health of the API and its backend systems")
async def health_check_endpoint():
    """
    Provides the operational status of the API and its core components.
    """
    import tools # ensure tools module is accessible for POLICY_QA_CHAIN status
    qa_status_str = "Not Initialized or Error"
    if tools.POLICY_QA_CHAIN and not (isinstance(tools.POLICY_QA_CHAIN, str) and tools.POLICY_QA_CHAIN.startswith("ERROR:")):
        qa_status_str = "Operational"
    elif isinstance(tools.POLICY_QA_CHAIN, str): # It's an error string
        qa_status_str = tools.POLICY_QA_CHAIN

    agent_status_str = "Not Initialized"
    if core_initializer.GLOBAL_AGENT_INSTANCE:
        agent_status_str = "Operational"
    
    if api_systems_ready and agent_status_str == "Operational" and qa_status_str == "Operational":
        return HealthStatus(status="ok", message="All systems nominal.", qa_system_status=qa_status_str, agent_status=agent_status_str)
    else:
        # Determine overall status based on readiness flag. Individual components might provide more detail.
        overall_status = "error" if not api_systems_ready else "degraded"
        return HealthStatus(status=overall_status,
                            message="One or more backend systems may not be fully operational.",
                            qa_system_status=qa_status_str,
                            agent_status=agent_status_str)

if __name__ == "__main__":
    # This allows running the API directly for local development: `python main_api.py`
    # Uvicorn will serve the FastAPI application.
    # Ensure OLLAMA_BASE_URL is set if Ollama isn't at http://localhost:11434,
    # or rely on the default in tools.py/llm_interface.py for local execution.
    logger.info("Starting Uvicorn server for local development on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)