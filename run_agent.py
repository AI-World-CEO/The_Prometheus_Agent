# The_Prometheus_Agent/run_agent.py

import sys
import logging
import asyncio
from pathlib import Path

# --- A-1: Environment Setup ---
try:
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # --- Corrected Imports ---
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer
    import asyncqt
    from prometheus_agent.PrometheusAgent import PrometheusAgent
    from prometheus_agent.PersonalGUI import PersonalGUI

except ImportError as e:
    print(f"FATAL: A core component failed to import. Please run 'pip install -r requirements.txt'.")
    print(f"Python Error: {e}")
    sys.exit(1)


def setup_logging_and_paths():
    """Configures logging and ensures necessary directories exist."""
    for dirname in ["Logs", "Corpus", "Archives"]:
        (project_root / dirname).mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] [%(name)s:%(lineno)d] %(message)s",
        handlers=[
            logging.FileHandler(project_root / "Logs" / "prometheus_agent.log", mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    sys.excepthook = lambda exctype, value, tb: logging.critical("FATAL: Uncaught exception.",
                                                                 exc_info=(exctype, value, tb))

    logging.info("=" * 60)
    logging.info("Prometheus Agent Environment Setup Initialized")
    logging.info("=" * 60)


def apply_asyncqt_monkey_patch():
    """Applies a stability patch to asyncqt for Windows compatibility."""
    if sys.platform != "win32" or hasattr(asyncqt.QEventLoop, '_patched'):
        return

    def _patched_call_later(self, delay, callback, *args, context=None):
        milliseconds = max(0, int(delay * 1000))
        handle = asyncio.Handle(callback, args, self, context=context)
        QTimer.singleShot(milliseconds, lambda: handle._run() if not handle.cancelled() else None)
        return handle

    asyncqt.QEventLoop.call_later = _patched_call_later
    setattr(asyncqt.QEventLoop, '_patched', True)
    logging.info("Applied robust stability monkey-patch to asyncqt.QEventLoop.")


async def main():
    """The main asynchronous entry point that creates and runs the agent."""
    setup_logging_and_paths()
    apply_asyncqt_monkey_patch()

    app = QApplication.instance() or QApplication(sys.argv)
    main_loop = asyncqt.QEventLoop(app)
    asyncio.set_event_loop(main_loop)

    try:
        agent = PrometheusAgent()

        # --- THE FIX: Use the correct keyword argument 'agent' ---
        # The PersonalGUI class expects the agent to be passed with the name 'agent'.
        gui = PersonalGUI(agent=agent, loop=main_loop)

        # Link the GUI and its signals back to the agent instance
        agent.gui = gui
        agent.status_signal = gui.signals.status_updated

        gui.show()
        logging.info("Prometheus Agent application is now running. Handing control to GUI event loop.")
        asyncio.create_task(agent.run_autonomous_startup_tasks())

        with main_loop:
            sys.exit(await main_loop.run_forever())

    except (SystemExit, ValueError, FileNotFoundError) as e:
        logging.critical(f"Agent initialization failed: {e}")
    except Exception:
        logging.critical("A fatal error occurred during the main application startup.", exc_info=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Application interrupted by user (Ctrl+C).")