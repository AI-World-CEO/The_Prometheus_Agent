# prometheus_agent/tests/integration/test_asi_cycle_with_real_fs.py

import unittest
import asyncio
import json
import tempfile
import shutil
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch
from contextlib import ExitStack

from prometheus_agent.PrometheusAgent import PrometheusAgent

# This structure assumes the test is run from the project root directory


# --- Mock LLM Responses ---
MOCK_LLM_RESPONSES = {
    "hypothesis": {
        "thought_process": "The mock module is a prime candidate for a version bump to test the full I/O and benchmark cycle.",
        "target_module_id": "MockModule",
        "change_hypothesis": "Increment the version number to test the full write-cycle."
    },
    "mutation": "# Mutated mock module code\nversion = 2.0\n"
}

# --- A realistic, discoverable unittest file for the temporary environment ---
# This is what the real ASI_Core's `unittest discover` command will find and run.
FAKE_UNITTEST_FILE_CONTENT = """
import unittest
import time

class RealBenchmarkTest(unittest.TestCase):
    def test_performance_simulation(self):
        # This test just needs to pass. The timing is controlled by mocking time.perf_counter.
        time.sleep(0.01) # Simulate some work
        self.assertTrue(True)
"""


class TestASICycleWithRealFS(unittest.IsolatedAsyncioTestCase):
    """
    A high-level integration test for the ASI_Core's self-modification cycle,
    interacting with a real (temporary) file system and a realistic benchmark.

    This test verifies the agent's ability to:
    1. Read and write files correctly in an isolated environment.
    2. Create backups of modified files as an audit trail.
    3. Execute the real `unittest discover` benchmark as a separate process.
    4. Use mocked timing to control the benchmark outcome.
    5. Make a correct, data-driven decision to apply or discard a mutation.
    """

    def setUp(self):
        """
        Set up a temporary file system structure that mimics the real project,
        including a valid test suite for the ASI_Core to discover.
        """
        self.temp_dir = Path(tempfile.mkdtemp(prefix="asi_live_fire_"))

        # --- Create the mock file structure ---
        self.mock_prometheus_agent_dir = self.temp_dir / "prometheus_agent"
        self.mock_module_path = self.mock_prometheus_agent_dir / "mock_module_to_modify.py"
        self.mock_tests_dir = self.mock_prometheus_agent_dir / "tests"

        self.mock_tests_dir.mkdir(parents=True, exist_ok=True)
        (self.mock_prometheus_agent_dir / "Logs").mkdir() # For agent init

        # Write the module that the agent will target for modification
        self.mock_module_path.write_text("# Original mock module code\nversion = 1.0\n", encoding='utf-8')

        # Write the realistic benchmark file that `unittest discover` will find
        (self.mock_tests_dir / "test_benchmark.py").write_text(FAKE_UNITTEST_FILE_CONTENT, encoding='utf-8')

        # Create a minimal config file needed for the agent to initialize
        self.mock_config = {
            "agent_name": "TestAgentIntegration", "version": "1.0",
            "llm_models": {"provider": "local", "power_model": "mock-model", "default_model": "mock-model"},
            "cognitive_toolkit": {}, "skill_routing": {"asi_hypothesis": "power_model", "asi_mutation": "power_model"},
            "asi_core": {
                "run_interval_seconds": 9999,
                "target_modules": {
                    # Use relative path from project root, as the real config does
                    "MockModule": str(self.mock_module_path.relative_to(self.temp_dir))
                }
            }
        }
        (self.mock_prometheus_agent_dir / "config.json").write_text(json.dumps(self.mock_config))


    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.temp_dir)

    @patch('time.perf_counter')
    @patch('prometheus_agent.Mutator.Mutator.generate', new_callable=AsyncMock)
    @patch('prometheus_agent.PrometheusAgent._reload_super_brain')
    @patch('prometheus_agent.PrometheusAgent._load_skill_map')
    async def test_successful_cycle_with_improvement(self, mock_load_skills, mock_reload_brain, mock_llm_generate, mock_time):
        """
        Tests the "golden path" where a generated mutation is benchmarked as a
        proven improvement and is correctly written to disk.
        """
        # --- 1. MOCK AND PATCH SETUP ---
        mock_reload_brain.return_value = {"mock_brain": "loaded"}
        mock_load_skills.return_value = {}
        mock_llm_generate.side_effect = [
            MOCK_LLM_RESPONSES["hypothesis"],
            MOCK_LLM_RESPONSES["mutation"]
        ]
        # Control benchmark timing: second run is faster (1.0s vs 2.0s).
        mock_time.side_effect = [100.0, 102.0, 200.0, 201.0]

        # Use ExitStack to robustly patch all necessary paths for full isolation
        with ExitStack() as stack:
            stack.enter_context(patch('prometheus_agent.PrometheusAgent.SOURCE_ROOT', self.mock_prometheus_agent_dir))
            stack.enter_context(patch('prometheus_agent.PrometheusAgent.PROJECT_ROOT', self.temp_dir))
            stack.enter_context(patch('prometheus_agent.Prometheus.ASI_Core.project_root', self.temp_dir))

            # --- 2. EXECUTION ---
            agent = PrometheusAgent()
            await agent.asi_core.run_single_cycle()

            # --- 3. ASSERTIONS ---
            # Verify the file on disk was successfully overwritten.
            final_code = self.mock_module_path.read_text(encoding='utf-8')
            self.assertIn("version = 2.0", final_code, "The file should have been overwritten with the mutated code.")

            # Verify the backup file was created and contains the original code.
            backup_path = self.mock_module_path.with_suffix(".py.bak")
            self.assertTrue(backup_path.exists(), "A .bak file must be created as a persistent audit trail.")
            backup_code = backup_path.read_text(encoding='utf-8')
            self.assertIn("version = 1.0", backup_code, "The backup file should contain the original code.")

            # Verify the benchmark was run twice (once for original, once for mutated).
            self.assertEqual(mock_time.call_count, 4, "time.perf_counter should be called twice per benchmark run.")

    @patch('time.perf_counter')
    @patch('prometheus_agent.Mutator.Mutator.generate', new_callable=AsyncMock)
    @patch('prometheus_agent.PrometheusAgent._reload_super_brain')
    @patch('prometheus_agent.PrometheusAgent._load_skill_map')
    async def test_discard_cycle_with_no_improvement(self, mock_load_skills, mock_reload_brain, mock_llm_generate, mock_time):
        """
        Tests the discard path where a generated mutation is correctly identified as
        not an improvement (a regression), and the original file is left untouched.
        """
        # --- 1. MOCK AND PATCH SETUP ---
        mock_reload_brain.return_value = {"mock_brain": "loaded"}
        mock_load_skills.return_value = {}
        mock_llm_generate.side_effect = [
            MOCK_LLM_RESPONSES["hypothesis"],
            MOCK_LLM_RESPONSES["mutation"]
        ]
        # Control benchmark timing: second run is slower (2.0s vs 1.0s).
        mock_time.side_effect = [100.0, 101.0, 200.0, 202.0]

        with ExitStack() as stack:
            stack.enter_context(patch('prometheus_agent.PrometheusAgent.SOURCE_ROOT', self.mock_prometheus_agent_dir))
            stack.enter_context(patch('prometheus_agent.PrometheusAgent.PROJECT_ROOT', self.temp_dir))
            stack.enter_context(patch('prometheus_agent.Prometheus.ASI_Core.project_root', self.temp_dir))

            # --- 2. EXECUTION ---
            agent = PrometheusAgent()
            await agent.asi_core.run_single_cycle()

            # --- 3. ASSERTIONS ---
            # The primary assertion is that the original file was NOT changed.
            final_code = self.mock_module_path.read_text(encoding='utf-8')
            self.assertIn("version = 1.0", final_code, "The original file should not have been modified.")

            # FIX: The backup file from the evaluation attempt SHOULD exist as an audit trail.
            # The ASI_Core creates the backup during evaluation and restores from it, but does not delete it.
            backup_path = self.mock_module_path.with_suffix(".py.bak")
            self.assertTrue(backup_path.exists(), "The backup from the evaluation attempt should still exist as an audit log.")
            # Verify its content to be sure
            backup_code = backup_path.read_text(encoding='utf-8')
            self.assertIn("version = 1.0", backup_code)


if __name__ == '__main__':
    unittest.main(verbosity=2)
