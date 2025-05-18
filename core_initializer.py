# core_initializer.py
import logging
import tools # This will make tools.POLICY_QA_CHAIN available after tools.initialize_policy_qa_system()
from agent import ReActAgent # Assuming agent.py is in the same directory

# Configure a basic logger for this module if it's used independently or early
# However, if app.py or main_api.py calls basicConfig first, that will be the one used.
# For simplicity, let's assume the main entry point (app.py or main_api.py) configures logging.
logger = logging.getLogger(__name__)

GLOBAL_AGENT_INSTANCE = None

def initialize_all_systems():
    """
    Initializes all core backend systems: QA system (FAISS index etc.) and the ReAct Agent.
    This function is designed to be idempotent.
    Returns True if all systems are successfully initialized or already initialized, False otherwise.
    Sets the GLOBAL_AGENT_INSTANCE if successful.
    """
    global GLOBAL_AGENT_INSTANCE
    logger.info("Core Initializer: Attempting to initialize/verify all backend systems...")

    # 1. Initialize/Verify Policy QA System
    qa_system_ok = False
    try:
        logger.info("Core Initializer: Calling tools.initialize_policy_qa_system()...")
        # initialize_policy_qa_system should be idempotent and handle load/create of FAISS
        if tools.initialize_policy_qa_system():
            # After the call, check the actual state of the chain
            if tools.POLICY_QA_CHAIN and not \
               (isinstance(tools.POLICY_QA_CHAIN, str) and tools.POLICY_QA_CHAIN.startswith("ERROR:")):
                logger.info("Core Initializer: Policy QA system successfully initialized/verified.")
                qa_system_ok = True
            else:
                logger.error(f"Core Initializer: Policy QA system init function seemed to succeed, but chain state is invalid or error: {tools.POLICY_QA_CHAIN}")
                # qa_system_ok remains False
        else:
            # initialize_policy_qa_system returned False, implying an error occurred
            logger.error(f"Core Initializer: Policy QA system initialization failed. Reported status: {tools.POLICY_QA_CHAIN}")
            # qa_system_ok remains False
    except Exception as e:
        logger.exception("Core Initializer: An unexpected exception occurred during Policy QA system setup:")
        tools.POLICY_QA_CHAIN = f"ERROR: Exception during init - {str(e)}" # Ensure error state is set
        # qa_system_ok remains False

    if not qa_system_ok:
        logger.error("Core Initializer: Halting further initialization as Policy QA system failed.")
        return False # Critical system failed

    # 2. Initialize/Verify ReAct Agent
    agent_ok = False
    if GLOBAL_AGENT_INSTANCE is None: # Only initialize if not already done
        try:
            logger.info("Core Initializer: Initializing ReAct Agent...")
            GLOBAL_AGENT_INSTANCE = ReActAgent()
            logger.info("Core Initializer: ReAct Agent initialized successfully.")
            agent_ok = True
        except Exception as e:
            logger.exception("Core Initializer: Fatal Error Initializing ReAct Agent:")
            # agent_ok remains False
    elif isinstance(GLOBAL_AGENT_INSTANCE, ReActAgent):
        logger.info("Core Initializer: ReAct Agent was already initialized.")
        agent_ok = True
    else: # Should not happen if GLOBAL_AGENT_INSTANCE is only set to ReActAgent or None
        logger.error("Core Initializer: GLOBAL_AGENT_INSTANCE has an unexpected type.")


    if qa_system_ok and agent_ok:
        logger.info("Core Initializer: All backend systems initialized/verified successfully.")
        return True
    else:
        logger.error("Core Initializer: One or more backend systems failed to initialize/verify.")
        return False