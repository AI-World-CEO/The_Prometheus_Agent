# prometheus_agent/tests/e2e/test_self_modification_integrity.py

import unittest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch
from contextlib import ExitStack

from prometheus_agent.PrometheusAgent import PrometheusAgent

# This structure assumes the test is run from the project root directory


# --- Mock LLM Responses ---
MOCK_LLM_RESPONSES = {
    "hypothesis": {
        "thought_process": "This is an E2E test. The goal is to prove the integrity of the self-modification loop. I will target the mock module for a version bump.",
        "target_module_id": "MockE2EModule",
        "change_hypothesis": "Increment the version number for test validation."
    },
    "mutation": "# E2E Test Mutated Code\n# This version is objectively better.\nversion = 2.0\n"
}

# --- A realistic, discoverable unittest file for the temporary environment ---
FAKE_UNITTEST_FILE_CONTENT = """
import unittest
import time

class RealBenchmarkTest(unittest.TestCase):
    def test_performance_simulation(self):
        # This test just needs to pass. The timing is controlled by mocking time.perf_counter.
        time.sleep(0.01)
        self.assertTrue(True)
"""


class TestSelfModificationIntegrityE2E(unittest.IsolatedAsyncioTestCase):
    """
    An End-to-End (E2E) test to audit the entire self-modification super-structure.

    This test verifies the successful collaboration of all components in an isolated
    environment, from hypothesis to file system I/O, using the real benchmark mechanism.
    """

    def setUp(self):
        """Creates a complete, temporary, and isolated project environment for the test."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="e2e_self_mod_"))

        # Mock project structure
        self.mock_prometheus_agent_dir = self.temp_dir / "prometheus_agent"
        self.mock_tests_dir = self.mock_prometheus_agent_dir / "tests"
        self.mock_tests_dir.mkdir(parents=True, exist_ok=True)
        (self.mock_prometheus_agent_dir / "Logs").mkdir()  # For agent init

        # Create the module that the agent will modify
        self.target_module_path = self.mock_prometheus_agent_dir / "module_to_evolve.py"
        self.target_module_path.write_text("# Original Code v1.0\nversion = 1.0\n", encoding='utf-8')

        # Create the REAL benchmark file in the temp environment
        (self.mock_tests_dir / "test_benchmark.py").write_text(FAKE_UNITTEST_FILE_CONTENT, encoding='utf-8')

        # Create a minimal, valid config file for the agent to load
        mock_config = {
            "agent_name": "TestAgentE2E", "version": "1.0",
            "llm_models": {"provider": "local", "power_model": "mock-model", "default_model": "mock-model"},
            "cognitive_toolkit": {}, "skill_routing": {"asi_hypothesis": "power_model", "asi_mutation": "power_model"},
            "sandboxing": {},
            "asi_core": {
                "run_interval_seconds": 9999,
                "target_modules": {
                    "MockE2EModule": str(self.target_module_path.relative_to(self.temp_dir))
                }
            }
        }
        (self.mock_prometheus_agent_dir / "config.json").write_text(json.dumps(mock_config))

    def tearDown(self):
        """Cleans up the temporary directory and all its contents."""
        shutil.rmtree(self.temp_dir)

    @patch('time.perf_counter')
    @patch('prometheus_agent.Mutator.Mutator.generate', new_callable=AsyncMock)
    @patch('prometheus_agent.PrometheusAgent._reload_super_brain')
    @patch('prometheus_agent.PrometheusAgent._load_skill_map')
    async def test_e2e_successful_modification_cycle(self, mock_load_skills, mock_reload_brain, mock_llm_generate,
                                                     mock_time):
        """
        Tests the 'golden path' where a mutation is generated, benchmarked as an
        improvement, and correctly applied, leaving a verifiable audit trail.
        """
        # --- 1. MOCK AND PATCH SETUP ---
        mock_reload_brain.return_value = {"mock_brain": "loaded"}
        mock_load_skills.return_value = {}
        mock_llm_generate.side_effect = [
            MOCK_LLM_RESPONSES["hypothesis"],
            MOCK_LLM_RESPONSES["mutation"]
        ]
        # Control benchmark timing: second run is faster.
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
            # 3.1: Verify the file on disk was successfully overwritten.
            final_code = self.target_module_path.read_text(encoding='utf-8')
            self.assertIn("version = 2.0", final_code)

            # 3.2: **CRITICAL**: Verify the backup file now exists and contains the original code.
            backup_path = self.target_module_path.with_suffix(".py.bak")
            self.assertTrue(backup_path.exists(), "A .bak file must be created as a persistent audit trail.")
            backup_code = backup_path.read_text(encoding='utf-8')
            self.assertIn("version = 1.0", backup_code)

            # 3.3: Verify agent's internal state was updated.
            self.assertEqual(agent.version, "1.1", "The agent's internal version should have been updated.")
            self.assertEqual(mock_llm_generate.call_count, 2, "LLM should be called for hypothesis and mutation.")

    @patch('time.perf_counter')
    @patch('prometheus_agent.Mutator.Mutator.generate', new_callable=AsyncMock)
    @patch('prometheus_agent.PrometheusAgent._reload_super_brain')
    @patch('prometheus_agent.PrometheusAgent._load_skill_map')
    async def test_e2e_discarded_modification_cycle(self, mock_load_skills, mock_reload_brain, mock_llm_generate,
                                                    mock_time):
        """
        Tests the 'regression path' where a mutation is correctly benchmarked as
        NOT an improvement and is discarded, leaving the original file untouched.
        """
        # --- 1. MOCK AND PATCH SETUP ---
        mock_reload_brain.return_value = {"mock_brain": "loaded"}
        mock_load_skills.return_value = {}
        mock_llm_generate.side_effect = [
            MOCK_LLM_RESPONSES["hypothesis"],
            MOCK_LLM_RESPONSES["mutation"]
        ]
        # Control benchmark timing: second run is slower.
        mock_time.side_effect = [100.0, 101.0, 200.0, 202.0]

        with ExitStack() as stack:
            stack.enter_context(patch('prometheus_agent.PrometheusAgent.SOURCE_ROOT', self.mock_prometheus_agent_dir))
            stack.enter_context(patch('prometheus_agent.PrometheusAgent.PROJECT_ROOT', self.temp_dir))
            stack.enter_context(patch('prometheus_agent.Prometheus.ASI_Core.project_root', self.temp_dir))

            # --- 2. EXECUTION ---
            agent = PrometheusAgent()
            await agent.asi_core.run_single_cycle()

            # --- 3. ASSERTIONS ---
            # 3.1: Verify the original file was NOT modified.
            final_code = self.target_module_path.read_text(encoding='utf-8')
            self.assertIn("version = 1.0", final_code)

            # 3.2: Verify the backup file STILL exists, showing what was tested.
            backup_path = self.target_module_path.with_suffix(".py.bak")
            self.assertTrue(backup_path.exists(), "The backup from the evaluation should still exist.")

            # 3.3: Verify agent state is unchanged.
            self.assertEqual(agent.version, "1.0", "The agent's version should not change on a discarded mutation.")


if __name__ == '__main__':
    unittest.main(verbosity=2)
