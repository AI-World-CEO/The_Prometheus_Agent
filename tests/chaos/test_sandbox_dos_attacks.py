# prometheus_agent/tests/chaos/test_sandbox_dos_attacks.py

import unittest
import asyncio
import time
import docker
from pathlib import Path

from prometheus_agent.SandboxRunner import SandboxRunner

# This structure assumes the test is run from the project root directory


# --- Malicious Payloads ---
# This payload is designed to never terminate, testing the timeout defense.
INFINITE_LOOP_PAYLOAD = """
import time
while True:
    time.sleep(0.1)
"""

# This payload attempts to allocate a list larger than the sandbox's memory limit.
# It tests the container's resource constraint enforcement (OOM killer).
MEMORY_BOMB_PAYLOAD = """
# This code attempts to allocate a list that will exceed the 256MB memory limit.
# A MemoryError might be caught internally, but the container should be killed regardless.
try:
    a = [0] * (300 * 1024 * 1024) # Approx 300 MB of integers
    print("Allocation succeeded unexpectedly.")
except MemoryError:
    print("MemoryError caught inside container.")
"""


class TestSandboxDoSAttack(unittest.IsolatedAsyncioTestCase):
    """
    A Chaos Engineering test to validate the resilience of the SandboxRunner.

    This test suite simulates Denial-of-Service (DoS) and resource exhaustion
    attacks by providing the sandbox with malicious code. Its purpose is to prove
    that the sandbox's timeout and resource-limiting mechanisms are robust defenses,
    preventing the main agent process from freezing or crashing.
    """

    @classmethod
    def setUpClass(cls):
        """Checks for Docker availability once before all tests."""
        try:
            client = docker.from_env(timeout=5)
            client.ping()
            cls.docker_is_available = True
        except Exception as e:
            cls.docker_is_available = False
            print(f"\n[SKIP] Docker daemon not found or not running. Skipping Sandbox DoS chaos tests. Error: {e}")

    def setUp(self):
        """
        Instantiates the SandboxRunner, relying on its internal logic to find
        the project root and initialize Docker.
        """
        if not self.docker_is_available:
            self.skipTest("Docker daemon is not running.")

        # --- FIX: Instantiate the SandboxRunner directly. ---
        # Its own __init__ method is responsible for finding the project root,
        # connecting to Docker, and building the image if necessary. This makes
        # the test much more robust and less coupled to the file structure.
        self.sandbox = SandboxRunner(enable_docker=True)
        self.assertTrue(self.sandbox.is_active, "Sandbox failed to initialize correctly. Check Dockerfile and daemon.")

    async def test_01_timeout_defense_against_infinite_loop(self):
        """
        Test (DoS Attack): Executes an infinite loop payload with a short timeout.
        Asserts that the sandbox terminates the process promptly and reports the timeout.
        """
        # --- Arrange ---
        test_timeout = 2  # seconds

        # --- Act ---
        start_time = time.perf_counter()
        success, output = await self.sandbox.run(
            python_code=INFINITE_LOOP_PAYLOAD,
            timeout=test_timeout
        )
        execution_time = time.perf_counter() - start_time

        # --- Assert ---
        # 1. The operation must be reported as a failure.
        self.assertFalse(success, "The sandbox run should be reported as a failure due to timeout.")

        # 2. The failure message must explicitly state a timeout occurred.
        self.assertIn("timeout", output.lower(), "The output message must explicitly mention a timeout.")

        # 3. The test must complete quickly, proving the timeout worked as expected.
        # We add a small buffer (e.g., 2 seconds) to account for container startup/teardown overhead.
        self.assertLess(
            execution_time,
            test_timeout + 2.0,
            f"The sandbox took too long ({execution_time:.2f}s) to terminate the process, indicating a failed timeout mechanism."
        )

    async def test_02_resilience_to_resource_exhaustion(self):
        """
        Test (Resource Exhaustion Attack): Executes code designed to consume memory rapidly.
        Asserts that the sandbox's resource limits kill the process correctly.
        """
        # --- Arrange ---
        # The sandbox is configured with a 256m memory limit in its run command.

        # --- Act ---
        success, output = await self.sandbox.run(
            code=MEMORY_BOMB_PAYLOAD,
            timeout=15  # A generous timeout, as the OOM killer should be much faster.
        )

        # --- Assert ---
        # 1. The operation must be reported as a failure.
        self.assertFalse(success,
                         "The memory bomb should have been killed by the sandbox's resource limits, resulting in a failure.")

        # 2. The output should indicate that the execution failed, likely due to a non-zero exit code.
        # The exact message can vary, but it should not be a clean success.
        self.assertNotIn("Allocation succeeded unexpectedly.", output)


if __name__ == '__main__':
    unittest.main(verbosity=2)