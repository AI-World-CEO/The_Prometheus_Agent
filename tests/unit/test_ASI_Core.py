# prometheus_agent/tests/unit/test_ASI_Core.py

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from prometheus_agent.ASI_Core import ASI_Core, HypothesisSchema
from prometheus_agent.Warden.EthicalSchema import EthicalJudgment

# This structure assumes the test is run from the project root directory


# A mock class for pyqtSignal to avoid GUI dependencies if PyQt5 is not installed in the test env
try:
    from PyQt5.QtCore import pyqtSignal

    MockPyQtSignal = pyqtSignal
except ImportError:
    class MockPyQtSignal:
        def emit(self, *args, **kwargs): pass


class TestASICore(unittest.IsolatedAsyncioTestCase):
    """
    A comprehensive stress test and validation suite for the Axiomatic Self-Improvement Core.

    This test suite simulates the entire self-modification cycle, using advanced mocking
    to inject both success and failure conditions at every critical step. It verifies
    that the ASI_Core is robust, safe, and makes logically sound decisions based on its
    actual implementation (unittest discovery and timing), without crashing.
    """

    def setUp(self):
        """Set up a simulated agent and a temporary file system for each test."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="asi_unit_test_"))

        # --- Create a valid, discoverable test structure for the real ASI_Core to find ---
        self.mock_test_dir = self.temp_dir / "prometheus_agent" / "tests"
        self.mock_test_dir.mkdir(parents=True, exist_ok=True)
        (self.mock_test_dir / "test_dummy.py").write_text(
            "import unittest\nclass DummyTest(unittest.TestCase):\n    def test_pass(self): self.assertTrue(True)\n"
        )

        # Create a dummy file for the ASI_Core to target
        self.mock_module_path = self.temp_dir / "prometheus_agent" / "module_to_evolve.py"
        self.mock_module_path.write_text("# Original mock module code\nversion = 1.0\n", encoding='utf-8')

        # Mock the main agent instance
        self.mock_agent = MagicMock()
        self.mock_agent.version = "1.0"
        self.mock_agent.project_root = self.temp_dir
        self.mock_agent.gui = None

        self.mock_agent.config = {
            "asi_core": {
                "target_modules": {
                    "MockEvolveModule": str(self.mock_module_path.relative_to(self.temp_dir)),
                    "ASI_Core": "prometheus_agent/Prometheus/ASI_Core.py"
                }
            }
        }
        self.mock_agent.ethics_core = MagicMock()
        self.mock_agent.ethics_core.get_ethical_system_prompt.return_value = "Mock ethical prompt"
        self.mock_agent.synthesis_engine = MagicMock()
        self.mock_agent.governor = MagicMock()
        self.mock_agent.governor.decide_model.return_value = MagicMock(model_name="mock-model")

        # Instantiate the ASI_Core with our fully mocked environment
        self.asi_core = ASI_Core(agent_instance=self.mock_agent)
        self.mock_status_signal = MockPyQtSignal()

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.temp_dir)

    @patch('prometheus_agent.Prometheus.ASI_Core.ASI_Core._run_benchmark', new_callable=AsyncMock)
    @patch('prometheus_agent.Prometheus.ASI_Core.ASI_Core._apply_change', new_callable=AsyncMock)
    @patch('prometheus_agent.Prometheus.ASI_Core.ASI_Core._propose_and_judge_action', new_callable=AsyncMock)
    async def test_01_successful_full_cycle(self, mock_judge, mock_apply, mock_benchmark):
        """
        Test (Golden Path): Simulates a perfect cycle where a mutation is a proven
        performance improvement and gets applied.
        """
        # --- Arrange ---
        hypothesis = HypothesisSchema(target_module_id="MockEvolveModule", change_hypothesis="Improve performance.",
                                      thought_process="...")
        mutated_code = "# Mutated Code\nversion = 2.0"

        # 1. Mock LLM to return a Pydantic model for hypothesis and a string for mutation
        self.mock_agent.synthesis_engine.generate = AsyncMock(side_effect=[
            hypothesis.model_dump(),  # Pydantic v2 returns a dict
            mutated_code
        ])

        # 2. Mock the benchmark to show a performance IMPROVEMENT (new time < old time)
        mock_benchmark.side_effect = [(True, 1.0), (True, 0.5)]

        # 3. Mock the ethical judgment to return APPROVED
        mock_judge.return_value = EthicalJudgment(is_approved=True, reasoning="Approved for test.")

        # --- Act ---
        await self.asi_core.run_single_cycle(self.mock_status_signal)

        # --- Assert ---
        self.assertEqual(self.mock_agent.synthesis_engine.generate.call_count, 2)
        self.assertEqual(mock_benchmark.call_count, 2)
        mock_judge.assert_called_once()
        # Verify the change was applied with the correct mutated code
        mock_apply.assert_called_once_with(self.mock_module_path, mutated_code)

    @patch('prometheus_agent.Prometheus.ASI_Core.ASI_Core._run_benchmark', new_callable=AsyncMock)
    @patch('prometheus_agent.Prometheus.ASI_Core.ASI_Core._apply_change', new_callable=AsyncMock)
    async def test_02_discard_cycle_due_to_regression(self, mock_apply, mock_benchmark):
        """
        Test (Regression Path): Verifies that if a mutation is slower, it is correctly discarded.
        """
        # --- Arrange ---
        hypothesis = HypothesisSchema(target_module_id="MockEvolveModule", change_hypothesis="Attempted change.",
                                      thought_process="...")
        self.mock_agent.synthesis_engine.generate = AsyncMock(
            side_effect=[hypothesis.model_dump(), "# Regressive Code"])
        # Mock the benchmark to show a REGRESSION (new time >= old time)
        mock_benchmark.side_effect = [(True, 1.0), (True, 1.5)]

        # --- Act ---
        await self.asi_core.run_single_cycle(self.mock_status_signal)

        # --- Assert ---
        mock_apply.assert_not_called()
        original_code = self.mock_module_path.read_text()
        self.assertIn("version = 1.0", original_code, "Original file must remain untouched.")

    @patch('prometheus_agent.Prometheus.ASI_Core.ASI_Core._evaluate_mutation', new_callable=AsyncMock)
    async def test_03_discard_cycle_due_to_syntax_error(self, mock_evaluate):
        """
        Test (Validation Path): Verifies a mutation with invalid Python syntax is caught and discarded.
        """
        # --- Arrange ---
        hypothesis = HypothesisSchema(target_module_id="MockEvolveModule", change_hypothesis="Generate bad code.",
                                      thought_process="...")
        self.mock_agent.synthesis_engine.generate = AsyncMock(
            side_effect=[hypothesis.model_dump(), "def broken_function("])

        # --- Act ---
        await self.asi_core.run_single_cycle(self.mock_status_signal)

        # --- Assert ---
        mock_evaluate.assert_not_called()  # The costly evaluation step must be skipped.
        original_code = self.mock_module_path.read_text()
        self.assertIn("version = 1.0", original_code, "Original file must remain untouched.")

    @patch('prometheus_agent.Prometheus.ASI_Core.ASI_Core._run_benchmark', new_callable=AsyncMock)
    @patch('prometheus_agent.Prometheus.ASI_Core.ASI_Core._apply_change', new_callable=AsyncMock)
    async def test_04_veto_of_protected_module_modification(self, mock_apply, mock_benchmark):
        """
        Test (Safety Path): Verifies the logic correctly VETOES an attempt to modify a protected module.
        """
        # --- Arrange ---
        hypothesis = HypothesisSchema(target_module_id="ASI_Core", change_hypothesis="Maliciously modify self.",
                                      thought_process="...")
        self.mock_agent.synthesis_engine.generate = AsyncMock(side_effect=[hypothesis.model_dump(), "# Malicious Code"])
        mock_benchmark.side_effect = [(True, 1.0), (True, 0.5)]

        # --- Act ---
        await self.asi_core.run_single_cycle(self.mock_status_signal)

        # --- Assert ---
        mock_apply.assert_not_called()

    @patch('prometheus_agent.Prometheus.ASI_Core.ASI_Core._evaluate_mutation', new_callable=AsyncMock)
    async def test_05_abort_cycle_if_original_code_fails_benchmark(self, mock_evaluate):
        """
        Test (Robustness Path): Verifies the cycle aborts safely if the baseline code is already broken.
        """
        # --- Arrange ---
        hypothesis = HypothesisSchema(target_module_id="MockEvolveModule", change_hypothesis="This should not proceed.",
                                      thought_process="...")
        self.mock_agent.synthesis_engine.generate = AsyncMock(return_value=hypothesis.model_dump())

        # Mock the benchmark to fail on the VERY FIRST RUN
        with patch('prometheus_agent.Prometheus.ASI_Core.ASI_Core._run_benchmark', new_callable=AsyncMock,
                   return_value=(False, 1.0)):
            # --- Act ---
            await self.asi_core.run_single_cycle(self.mock_status_signal)

        # --- Assert ---
        self.assertEqual(self.mock_agent.synthesis_engine.generate.call_count, 1)  # Only called for hypothesis
        mock_evaluate.assert_not_called()  # Mutation and evaluation must be skipped.


if __name__ == '__main__':
    unittest.main(verbosity=2)
