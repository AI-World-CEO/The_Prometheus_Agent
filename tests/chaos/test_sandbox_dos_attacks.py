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
        Instantiates the SandboxRunner for each test and robustly finds the project root.
        """
        if not self.docker_is_available:
            self.skipTest("Docker daemon is not running.")

        try:
            # A-1 GOLD STANDARD FIX: Robustly find the project root from this file's location.
            # This is 3 levels up from prometheus_agent/tests/chaos/test_sandbox_dos_attack.py
            self.project_root = Path(__file__).resolve().parents[3]
            if not (self.project_root / "Dockerfile.sandbox").exists():
                # Fallback for different execution contexts
                self.project_root = Path.cwd()
                if not (self.project_root / "Dockerfile.sandbox").exists():
                    raise FileNotFoundError()
        except (IndexError, FileNotFoundError):
            self.fail("Could not determine the project root containing 'Dockerfile.sandbox'. The test cannot proceed.")

        # Instantiate the real SandboxRunner with a minimal config, pointing to the discovered project root.
        self.sandbox = SandboxRunner(
            config={"sandboxing": {"enable_docker": True}},
            project_root=self.project_root
        )
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
            timeout_seconds=test_timeout
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
            python_code=MEMORY_BOMB_PAYLOAD,
            timeout_seconds=15  # A generous timeout, as the OOM killer should be much faster.
        )

        # --- Assert ---
        # 1. The operation must be reported as a failure.
        self.assertFalse(success,
                         "The memory bomb should have been killed by the sandbox's resource limits, resulting in a failure.")

        # 2. The output should indicate a non-zero exit code, characteristic of being killed.
        # Docker's OOM killer typically results in exit code 137 (128 + 9 for SIGKILL).
        # We check for a general "exit code" message for robustness.
        self.assertIn("exit code", output.lower(), f"Expected a resource-limit failure message, but got: {output}")

        # 3. Specifically confirm the process was likely killed (exit code 137).
        # This is a stronger assertion for Linux-based Docker environments.
        self.assertIn("137", output,
                      f"The exit code should indicate an OOM kill (137), but was not found in output: {output}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
