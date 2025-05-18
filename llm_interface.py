# llm_interface.py
import requests
import json
import logging
import os

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"
MODEL_NAME = "mistral" # Model for the ReAct agent's thinking process
# Other models (for embeddings, for LangChain QA) are defined in tools.py

def query_ollama(prompt_text, model_name=MODEL_NAME): # Allow specifying model
    payload = {
        "model": model_name,
        "prompt": prompt_text,
        "stream": False,
        "options": {
            "temperature": 0.05, # Low for ReAct consistency
            "num_predict": 400,  # Allow enough tokens for Thought + Action or simple Final Answer
            "stop": [
                "Observation:", "\nObservation:", "Observation:\n",
                "\nThought:", # Stop if LLM tries to start a new thought cycle prematurely
            ]
        }
    }
    try:
        # logger.debug(f"Sending prompt to Ollama ({model_name}) (last 500 chars): ...{prompt_text[-500:]}")
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120) # Increased timeout
        response.raise_for_status()
        response_data = response.json()
        generated_text = response_data.get("response", "").strip()
        # logger.debug(f"Ollama ({model_name}) raw response (first 300 chars): {generated_text[:300]}...")

        obs_pos = generated_text.find("Observation:")
        if obs_pos != -1 and generated_text.count("Action:") == 0 and generated_text.count("Final Answer:") == 0 :
            thought_pos = generated_text.rfind("Thought:")
            if thought_pos == -1 or obs_pos < thought_pos + 10: # Heuristic: Observation: likely hallucinated if very early
                generated_text = generated_text[:obs_pos].strip()
                logger.warning("Manually truncated 'Observation:' likely hallucinated by LLM at the beginning of its response.")

        return generated_text

    except requests.exceptions.Timeout:
        error_msg = f"Ollama request ({model_name}) timed out after 120 seconds."
        logger.error(error_msg)
        return f"ERROR_OLLAMA_TIMEOUT: {error_msg}"
    except requests.exceptions.RequestException as e:
        error_msg = f"Error connecting to Ollama ({model_name}): {e}."
        logger.error(error_msg)
        return f"ERROR_OLLAMA_CONNECTION: {error_msg}"
    except json.JSONDecodeError as e:
        error_msg = f"Error decoding JSON from Ollama ({model_name}): {e}. Raw response: {response.text}"
        logger.error(error_msg)
        return f"ERROR_OLLAMA_JSON: {error_msg}"