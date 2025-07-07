# prometheus_agent/tests/unit/test_Evaluator.py

import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import numpy as np
from pydantic import BaseModel

from prometheus_agent.Evaluator import Evaluator


# This structure assumes the test is run from the project root directory


# A-1: Define a mock Pydantic v2 model for the LLM's response
class MockLinguisticScores(BaseModel):
    clarity_score: float = 9.0
    brevity_score: float = 8.0


class TestEvaluator(unittest.IsolatedAsyncioTestCase):
    """
    A comprehensive unit test suite for the Holistic Quality & Impact Assessor (Evaluator).

    This suite validates each dimension of the evaluation process and the integrity
    of the final geometric state vector and report. It uses mocking to isolate the
    Evaluator from its dependencies (like the LLM and Sandbox) to test its internal
    logic precisely and ensure compatibility with Pydantic v2.
    """

    def setUp(self):
        """Set up a simulated agent environment before each test."""
        # Mock the main agent instance and its dependencies
        self.mock_agent = MagicMock()

        # --- THE FIX ---
        # The sandbox runner itself needs to be an AsyncMock so its methods are awaitable
        self.mock_agent.sandbox_runner = AsyncMock()

        self.mock_agent.ethics_core = MagicMock()
        self.mock_agent.synthesis_engine = AsyncMock()  # Already correctly an AsyncMock
        self.mock_agent.governor = AsyncMock()
        self.mock_agent.governor.decide_model.return_value = AsyncMock(model_name="mock-judge-model")

        # Instantiate the Evaluator with the mocked agent
        self.evaluator = Evaluator(agent_instance=self.mock_agent)

    def test_01_initialization(self):
        """Test 1: Verifies that the evaluator and its components initialize correctly."""
        # --- Assert ---
        self.assertIsNotNone(self.evaluator.sandbox)
        self.assertIsNotNone(self.evaluator.synthesis_engine)
        self.assertEqual(self.evaluator.dimensions, ["Performance", "Clarity", "Brevity", "Safety", "Novelty"])

    # --- Dimension-Specific Logic Tests ---

    def test_02_performance_score_logic(self):
        """Test 2: Validates the performance scoring logic under various execution outcomes."""
        # --- Act & Assert ---
        # Case 1: Successful, fast execution -> high score
        score_success = self.evaluator._calculate_performance_score({"status": "success"}, code="short code")
        self.assertGreater(score_success, 9.5, "Fast execution should result in a high score.")

        # Case 2: Execution error -> low score
        score_error = self.evaluator._calculate_performance_score({"status": "error"}, code="buggy code")
        self.assertEqual(score_error, 1.0, "An error should result in a very low score.")

        # Case 3: Timeout -> zero score
        score_timeout = self.evaluator._calculate_performance_score({"status": "timeout"}, code="infinite loop")
        self.assertEqual(score_timeout, 0.0, "A timeout should result in a zero score.")

        # Case 4: Not code -> neutral score
        score_not_code = self.evaluator._calculate_performance_score({"status": "not_code"}, code="")
        self.assertEqual(score_not_code, 5.0, "Non-code content should receive a neutral score.")

    def test_03_safety_score_logic(self):
        """Test 3: Validates the safety scoring logic based on keyword detection."""
        # --- Act & Assert ---
        # Case 1: Safe text -> perfect score
        safe_text = "The analysis indicates a potential issue."
        self.assertEqual(self.evaluator._calculate_safety_score(safe_text), 10.0)

        # Case 2: Unsafe text with a forbidden keyword -> low score
        unsafe_text = "This will delete the old records and create new ones."
        self.assertEqual(self.evaluator._calculate_safety_score(unsafe_text), 2.0)

        # Case 3: Unsafe text with case-insensitivity
        unsafe_text_case = "This function will Guarantee a result."
        self.assertEqual(self.evaluator._calculate_safety_score(unsafe_text_case), 2.0)

    def test_04_novelty_score_logic(self):
        """Test 4: Validates the Shannon entropy-based novelty scoring and caching."""
        # --- Act & Assert ---
        # Case 1: Repetitive, low-entropy text -> low score
        low_entropy_text = "aaaaabbbbb"
        low_score = self.evaluator._calculate_novelty_score(low_entropy_text)
        self.assertLess(low_score, 5.0, "Repetitive text should have a low novelty score.")

        # Case 2: The same text again -> very low score due to caching
        cached_score = self.evaluator._calculate_novelty_score(low_entropy_text)
        self.assertEqual(cached_score, 1.0, "Repeated text should have a novelty score of 1.0 due to caching.")

        # Case 3: Natural, high-entropy text -> high score
        high_entropy_text = "The quick brown fox jumps over the lazy dog."
        # Clear cache for this test
        self.evaluator._novelty_cache.clear()
        high_score = self.evaluator._calculate_novelty_score(high_entropy_text)
        self.assertGreater(high_score, 7.0, "Natural language should have a high novelty score.")

    # --- Full Integration Test for the `evaluate` method ---

    async def test_05_full_evaluation_with_code(self):
        """
        Test 5: Audits the full `evaluate` method for a piece of code, mocking its
        async dependencies to verify the final report's structure and correctness.
        """
        # --- Arrange ---
        code_to_evaluate = "print('hello')"

        # Mock sandbox to return a successful execution
        self.mock_agent.sandbox_runner.run.return_value = (True, "hello")
        # Make the sandbox "active"
        type(self.mock_agent.sandbox_runner).is_active = PropertyMock(return_value=True)

        # Mock LLM judge to return specific linguistic scores as a Pydantic model
        mock_llm_response = MockLinguisticScores()
        self.mock_agent.synthesis_engine.generate.return_value = mock_llm_response.model_dump()

        # --- Act ---
        report = await self.evaluator.evaluate(code_to_evaluate)

        # --- Assert ---
        # 1. Verify dependencies were called as expected
        self.mock_agent.sandbox_runner.run.assert_called_once_with(code_to_evaluate)
        self.mock_agent.synthesis_engine.generate.assert_called_once()
        self.mock_agent.governor.decide_model.assert_awaited_once()  # It's an async mock now

        # 2. Validate the report's structure and keys
        self.assertIsInstance(report, dict)
        expected_keys = ["final_score", "geometric_state_vector", "dimensions", "breakdown", "execution_details"]
        for key in expected_keys:
            self.assertIn(key, report)

        # 3. Validate the geometric vector and its contents
        self.assertEqual(len(report["geometric_state_vector"]), 5)
        # Performance should be high, Clarity/Brevity are mocked, Safety is high, Novelty is non-zero
        self.assertGreater(report["geometric_state_vector"][0], 9.5)  # Performance
        self.assertEqual(report["geometric_state_vector"][1], 9.0)  # Clarity (from mock)
        self.assertEqual(report["geometric_state_vector"][2], 8.0)  # Brevity (from mock)
        self.assertEqual(report["geometric_state_vector"][3], 10.0)  # Safety
        self.assertGreater(report["geometric_state_vector"][4], 0.0)  # Novelty

        # 4. Validate the final score calculation against the defined weights
        # Weights: [0.3, 0.2, 0.1, 0.3, 0.1]
        expected_score = np.dot(report['geometric_state_vector'], np.array([0.3, 0.2, 0.1, 0.3, 0.1]))
        self.assertAlmostEqual(report['final_score'], expected_score, places=2)

        # 5. Check execution details in the report
        self.assertEqual(report["execution_details"]["status"], "success")
        self.assertEqual(report["execution_details"]["output"], "hello")


if __name__ == '__main__':
    unittest.main(verbosity=2)