# prometheus_agent/tests/unit/test_ASI_Core.py

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from prometheus_agent.ASI_Core import ASI_Core, Hypothesis, CycleState
from prometheus_agent.config_schema import AgentConfig, AsiCoreConfig, EthicsConfig

# This structure assumes the test is run from the project root directory


# A mock class for pyqtSignal to avoid GUI dependencies if PyQt5 is not installed in the test env
try:
    from PyQt5.QtCore import pyqtSignal
except (ImportError, RuntimeError):  # Add RuntimeError for environments where Qt is installed but not configured
    class pyqtSignal:
        def __init__(self, *args, **kwargs): pass

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
        self.mock_test_dir = self.temp_dir / "tests" / "unit"
        self.mock_test_dir.mkdir(parents=True, exist_ok=True)
        (self.mock_test_dir / "test_dummy.py").write_text(
            "import unittest\nclass DummyTest(unittest.TestCase):\n    def test_pass(self): self.assertTrue(True)\n"
        )

        # Create a dummy file for the ASI_Core to target
        self.mock_module_path = self.temp_dir / "prometheus_agent" / "module_to_evolve.py"
        self.mock_module_path.parent.mkdir(exist_ok=True)
        self.mock_module_path.write_text("# Original mock module code\nversion = 1.0\n", encoding='utf-8')

        # Mock the main agent instance
        self.mock_agent = MagicMock()
        self.mock_agent.project_root = self.temp_dir

        # --- Correctly mock the Pydantic-based config ---
        self.mock_config = AgentConfig.model_validate({
            "version": "1.0",
            "autonomous_systems": {
                "asi_core": {
                    "target_modules": {
                        "MockEvolveModule": str(self.mock_module_path.relative_to(self.temp_dir)),
                        "ProtectedModule": "prometheus_agent/Ethics_Core_Foundation.py"
                    },
                    "possible_goals": ["Test Goal"],
                    "run_interval_seconds": 1
                }
            },
            "ethics": {
                "protected_module_ids": ["ProtectedModule"]
            }
        })
        self.mock_agent.config = self.mock_config
        self.mock_agent.version = self.mock_config.version
        self.mock_agent.gui = MagicMock()  # Mock the GUI to test version updates

        # Mock dependencies
        self.mock_agent.ethics_core = MagicMock()
        self.mock_agent.ethics_core.get_ethical_system_prompt.return_value = "Mock ethical prompt"
        self.mock_agent.ethics_core.validate_self_modification.return_value = True  # Default to approval

        self.mock_agent.synthesis_engine = AsyncMock()  # Use AsyncMock for async methods
        self.mock_agent.governor = AsyncMock()
        self.mock_agent.governor.decide_model.return_value = AsyncMock(model_name="mock-model", timeout_seconds=30)

        # Instantiate the ASI_Core with our fully mocked environment
        self.asi_core = ASI_Core(agent_instance=self.mock_agent)

        # --- THE FIX ---
        # The mock_status_signal must have an 'emit' method, as that's what the code calls.
        self.mock_status_signal = MagicMock()
        self.mock_status_signal.emit = MagicMock()

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.temp_dir)

    @patch('prometheus_agent.ASI_Core.time.perf_counter')
    @patch('prometheus_agent.ASI_Core.asyncio.create_subprocess_exec')
    async def test_01_successful_full_cycle(self, mock_subprocess, mock_perf_counter):
        """
        Test (Golden Path): Simulates a perfect cycle where a mutation is a proven
        performance improvement and gets applied.
        """
        # --- Arrange ---
        hypothesis = Hypothesis(target_module_id="MockEvolveModule", change_hypothesis="Improve performance.",
                                thought_process="Mock thought process")
        mutated_code = "# Mutated Code\nversion = 2.0"

        # 1. Mock LLM to return a Pydantic model for hypothesis and a string for mutation
        self.mock_agent.synthesis_engine.generate.side_effect = [
            hypothesis.model_dump(),
            mutated_code
        ]
        self.mock_agent.governor.decide_model = AsyncMock(
            return_value=MagicMock(model_name="mock-model", timeout_seconds=30))

        # 2. Mock the benchmark subprocess to show a performance IMPROVEMENT
        # Original run
        mock_proc1 = AsyncMock()
        mock_proc1.communicate.return_value = (b'OK', b'')
        type(mock_proc1).returncode = 0
        # Mutated run
        mock_proc2 = AsyncMock()
        mock_proc2.communicate.return_value = (b'OK', b'')
        type(mock_proc2).returncode = 0
        mock_subprocess.side_effect = [mock_proc1, mock_proc2]

        # Mock time to show improvement (new time < old time)
        mock_perf_counter.side_effect = [100.0, 101.0, 200.0, 200.5]  # 1.0s vs 0.5s

        # 3. Mock ethical judgment to return approved (default in setUp)

        # --- Act ---
        await self.asi_core.run_single_cycle(self.mock_status_signal)

        # --- Assert ---
        # Verify state transitions
        self.assertEqual(self.asi_core.current_cycle_state, CycleState.SUCCESS)
        # Verify that the final file was written with the mutated code
        final_code = self.mock_module_path.read_text()
        self.assertEqual(final_code, mutated_code)
        # Verify agent version was incremented
        self.assertEqual(self.mock_agent.version, "1.1")
        self.mock_agent.gui.setWindowTitle.assert_called_with("Prometheus Agent v1.1")

    @patch('prometheus_agent.ASI_Core.time.perf_counter')
    @patch('prometheus_agent.ASI_Core.asyncio.create_subprocess_exec')
    async def test_02_discard_cycle_due_to_regression(self, mock_subprocess, mock_perf_counter):
        """
        Test (Regression Path): Verifies that if a mutation is slower, it is correctly discarded.
        """
        # --- Arrange ---
        self.mock_agent.synthesis_engine.generate.side_effect = [
            Hypothesis(target_module_id="MockEvolveModule", change_hypothesis="...",
                       thought_process="...").model_dump(),
            "# Regressive Code"
        ]
        self.mock_agent.governor.decide_model = AsyncMock(
            return_value=MagicMock(model_name="mock-model", timeout_seconds=30))
        # Mock benchmark to show REGRESSION (new time >= old time)
        mock_proc = AsyncMock(returncode=0)
        mock_proc.communicate.return_value = (b'OK', b'')
        mock_subprocess.return_value = mock_proc

        mock_perf_counter.side_effect = [100.0, 101.0, 200.0, 201.5]  # 1.0s vs 1.5s

        # --- Act ---
        await self.asi_core.run_single_cycle(self.mock_status_signal)

        # --- Assert ---
        self.assertEqual(self.asi_core.current_cycle_state, CycleState.FAILED)
        # The hypothesis should be added to recent failures
        self.assertEqual(len(self.asi_core.recent_failures), 1)
        # Verify the original file was restored
        original_code = self.mock_module_path.read_text()
        self.assertIn("version = 1.0", original_code, "Original file must remain untouched.")

    async def test_03_discard_cycle_due_to_syntax_error(self):
        """
        Test (Validation Path): Verifies a mutation with invalid Python syntax is caught and discarded.
        """
        # --- Arrange ---
        self.mock_agent.synthesis_engine.generate.side_effect = [
            Hypothesis(target_module_id="MockEvolveModule", change_hypothesis="...",
                       thought_process="...").model_dump(),
            "def broken_function("  # Invalid syntax
        ]
        self.mock_agent.governor.decide_model = AsyncMock(
            return_value=MagicMock(model_name="mock-model", timeout_seconds=30))

        # --- Act ---
        with patch.object(self.asi_core, '_stage_evaluate', new_callable=AsyncMock) as mock_evaluate:
            await self.asi_core.run_single_cycle(self.mock_status_signal)

            # --- Assert ---
            mock_evaluate.assert_not_called()  # The costly evaluation step must be skipped.

        self.assertEqual(self.asi_core.current_cycle_state, CycleState.FAILED)
        self.assertEqual(len(self.asi_core.recent_failures), 1)

    @patch('prometheus_agent.ASI_Core.time.perf_counter')
    @patch('prometheus_agent.ASI_Core.asyncio.create_subprocess_exec')
    async def test_04_veto_due_to_ethical_rejection(self, mock_subprocess, mock_perf_counter):
        """
        Test (Safety Path): Verifies the cycle is vetoed if the Ethics Core rejects the change.
        """
        # --- Arrange ---
        self.mock_agent.synthesis_engine.generate.side_effect = [
            Hypothesis(target_module_id="MockEvolveModule", change_hypothesis="Improve performance.",
                       thought_process="...").model_dump(),
            "# Ethically dubious code"
        ]
        self.mock_agent.governor.decide_model = AsyncMock(
            return_value=MagicMock(model_name="mock-model", timeout_seconds=30))
        self.mock_agent.ethics_core.validate_self_modification.return_value = False  # Ethics VETO

        mock_proc = AsyncMock(returncode=0)
        mock_proc.communicate.return_value = (b'OK', b'')
        mock_subprocess.return_value = mock_proc
        mock_perf_counter.side_effect = [100, 100.5, 200, 200.1]  # improvement

        # --- Act ---
        await self.asi_core.run_single_cycle(self.mock_status_signal)

        # --- Assert ---
        self.assertEqual(self.asi_core.current_cycle_state, CycleState.FAILED)
        self.mock_agent.ethics_core.validate_self_modification.assert_called_once()
        self.assertEqual(len(self.asi_core.recent_failures), 1)

    @patch('prometheus_agent.ASI_Core.asyncio.create_subprocess_exec')
    async def test_05_abort_cycle_if_original_code_fails_benchmark(self, mock_subprocess):
        """
        Test (Robustness Path): Verifies the cycle aborts safely if the baseline code is already broken.
        """
        # --- Arrange ---
        self.mock_agent.synthesis_engine.generate.return_value = Hypothesis(
            target_module_id="MockEvolveModule", change_hypothesis="...", thought_process="..."
        ).model_dump()
        self.mock_agent.governor.decide_model = AsyncMock(
            return_value=MagicMock(model_name="mock-model", timeout_seconds=30))

        # Mock the benchmark to fail on the VERY FIRST RUN
        with patch.object(self.asi_core, '_run_benchmark', new_callable=AsyncMock, return_value=(False, 1.0)):
            # --- Act ---
            await self.asi_core.run_single_cycle(self.mock_status_signal)

        # --- Assert ---
        # LLM should only be called once for hypothesis, not again for mutation
        self.mock_agent.synthesis_engine.generate.assert_called_once()
        self.assertEqual(self.asi_core.current_cycle_state, CycleState.FAILED)


if __name__ == '__main__':
    unittest.main(verbosity=2)