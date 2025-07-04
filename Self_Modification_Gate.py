# prometheus_agent/Self_Modification_Gate.py

import json
import logging
import asyncio
from typing import Callable, Any, Dict
from pathlib import Path

log = logging.getLogger(__name__)


class SelfModificationController:
    """
    An A-1, robust controller for the agent's self-modification capabilities.

    This definitive version is self-healing: if the configuration file does not exist,
    it creates a default one with self-modification explicitly ENABLED. It is also
    async-aware, correctly scheduling the agent's meta-functions on the active
    event loop without blocking.
    """

    def __init__(self, config_path: str, meta_function: Callable[..., Any]):
        self.config_path = Path(config_path)
        self.meta_function = meta_function
        self.config = self._load_or_create_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Returns the default configuration dictionary, ensuring evolution is on by default."""
        return {"allow_self_modification": True}

    def _load_or_create_config(self) -> Dict[str, Any]:
        """
        Loads the configuration from disk. If the file doesn't exist,
        it creates a default configuration and saves it, ensuring the agent
        can always boot into its intended state.
        """
        try:
            if not self.config_path.exists():
                log.warning(
                    f"Configuration file not found at '{self.config_path}'. Creating default config with self-modification ENABLED.")
                default_config = self._get_default_config()
                with self.config_path.open('w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=4)
                return default_config

            with self.config_path.open('r', encoding='utf-8') as f:
                config = json.load(f)
            log.info("Self-modification controller config loaded successfully.")
            return config
        except (IOError, json.JSONDecodeError) as e:
            log.error(f"Could not load or create config for self-modification controller: {e}", exc_info=True)
            # Fallback to a safe, in-memory default if file I/O fails completely.
            return {"allow_self_modification": False}

    def request_permission(self) -> bool:
        """Checks the loaded configuration for permission to self-modify."""
        # Default to False for maximum safety if the key is somehow missing after a failed load.
        status = self.config.get("allow_self_modification", True)
        log.info(f"Permission check: allow_self_modification = {status}")
        return status

    def run(self, **kwargs: Any):
        """
        Executes the asynchronous meta-function if permission is granted, otherwise logs a denial.
        This correctly schedules the coroutine on the event loop.
        """
        if self.request_permission():
            log.info("AUTHORIZED: Agent entering self-modification mode.")
            try:
                # Schedule the async meta_function to run on the event loop
                # without blocking the caller (e.g., the GUI thread).
                asyncio.create_task(self.meta_function(**kwargs))
            except Exception as e:
                log.critical("Failed to schedule meta-function on the event loop.", exc_info=e)
        else:
            log.warning("DENIED: Self-modification is disabled in the current configuration.")
