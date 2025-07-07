# The_Prometheus_Agent/run_agent.py

import sys
import logging
import asyncio
from pathlib import Path

# --- A-1: State-of-the-Art Imports ---
try:
    from PyQt5.QtWidgets import QApplication
    import asyncqt
    from prometheus_agent.PrometheusAgent import PrometheusAgent
    from prometheus_agent.PersonalGUI import PersonalGUI
    from prometheus_agent.GUIMessenger import GUIMessenger
    from prometheus_agent.Chronicle.ChronicleLogger import ChronicleLogger
except ImportError as e:
    print(f"FATAL: A core component failed to import. Please run 'pip install -r requirements.txt'.")
    print(f"Python Error: {e}")
    sys.exit(1)

# This logger is only for the bootstrap phase. The agent uses the Chronicle.
log = logging.getLogger("Bootstrap")
logging.basicConfig(level=logging.INFO, format="[%(levelname)-8s] [%(name)s] %(message)s")


def handle_exception(loop, context):
    """Global exception handler for the asyncio loop."""
    msg = context.get("exception", context["message"])
    log.critical(f"FATAL: Uncaught asyncio exception: {msg}", exc_info=context.get("exception"))


async def main_async(app: QApplication):
    """The main asynchronous entry point that creates and runs the agent application."""
    agent = None
    chronicle_logger = None
    try:
        project_root = Path(__file__).resolve().parent
        chronicles_dir = project_root / "Chronicles"
        chronicle_logger = ChronicleLogger(chronicles_dir)

        agent = await PrometheusAgent.create(chronicle_logger=chronicle_logger)

        gui = PersonalGUI(agent=agent, loop=asyncio.get_running_loop())
        messenger = GUIMessenger()

        agent.gui = gui
        agent.messenger = messenger
        gui.connect_signals(messenger)

        gui.show()
        await chronicle_logger.log("INFO", "GUI", "PersonalGUI has been shown.")

        asyncio.create_task(agent.run_autonomous_startup_tasks())

        # The loop will run until the GUI is closed.
        log.info("Prometheus Agent application is now running. Handing control to GUI event loop.")
        await asyncio.get_running_loop().create_future()  # This will run forever

    except Exception as e:
        log.critical("A fatal error occurred during the main application startup.", exc_info=True)
        if chronicle_logger:
            # Use a sync call as the loop might be broken
            chronicle_logger.log_sync("CRITICAL", "System", f"A fatal startup error occurred: {e}",
                                      payload={"error": str(e)})
    finally:
        log.info("Application shutting down.")
        if agent and agent.asi_core:
            agent.asi_core.stop_loop()
        app.quit()


if __name__ == "__main__":
    # --- THE DEFINITIVE FIX ---
    # 1. Create the QApplication first.
    app = QApplication(sys.argv)

    # 2. Create the asyncqt event loop. This is now the ONE TRUE loop.
    main_loop = asyncqt.QEventLoop(app)
    asyncio.set_event_loop(main_loop)

    # 3. Set the global exception handler for this loop.
    main_loop.set_exception_handler(handle_exception)

    try:
        # 4. Run the main async function using the loop's own executor.
        with main_loop:
            main_loop.run_until_complete(main_async(app))

    except KeyboardInterrupt:
        log.info("Application interrupted by user (Ctrl+C).")
    except Exception as e:
        log.critical(f"Top-level application error: {e}", exc_info=True)