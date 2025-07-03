# run_agent.py
# A-1 Definitive Application Entry Point
# -----------------------------------------------------------------------------
# This script is the single, correct way to launch the Prometheus Agent.
# It solves all Python path and import resolution issues by adding the
# project root to the system path and then using direct, absolute imports
# to the necessary functions from their source files. This is the most robust
# and unambiguous method possible.
#
# To Run:
# 1. Open a terminal in this file's directory (the project root).
# 2. Make sure your virtual environment is active.
# 3. Execute: python run_agent.py
# -----------------------------------------------------------------------------

import os
import sys
import asyncio
import logging

# --- A-1 Path Injection ---
# This ensures that the project root is on the Python path, allowing the
# 'Prometheus_Agent' package to be found and imported.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- A-1 DEFINITIVE, DIRECT IMPORTS ---
# We bypass the package's __init__ and import the specific functions we need
# directly from their source files. This is the most robust method.
from Prometheus_Agent.PrometheusAgent import main_async, setup_environment, log_uncaught_exceptions

from dotenv import load_dotenv


def main():
    """
    Orchestrates the complete application startup sequence.
    """
    # Load environment variables from .env file in the project root.
    load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

    # Set up logging and environment checks by calling the directly imported function.
    setup_environment()

    # Register the global exception handler by calling the directly imported function.
    sys.excepthook = log_uncaught_exceptions

    # Determine the LLM provider and API key.
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    logging.info(f"Detected LLM_PROVIDER: '{provider}'")

    key_to_use = os.getenv("OPENAI_API_KEY")
    if provider == "openai" and not key_to_use:
        logging.critical("FATAL: LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not found in .env file.")
        sys.exit(1)

    # Launch the asynchronous main application loop by calling the directly imported function.
    try:
        asyncio.run(main_async(api_key=key_to_use))
    except KeyboardInterrupt:
        logging.info("Application shutting down due to user interrupt (Ctrl+C).")
    except Exception as e:
        logging.critical("A fatal, unhandled error occurred in the main execution block.", exc_info=e)
        sys.exit(1)


if __name__ == "__main__":
    main()
