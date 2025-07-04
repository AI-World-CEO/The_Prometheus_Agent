# prometheus_agent/tests/unit/test_cognitive_manifold.py

import unittest
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from prometheus_agent.CognitiveManifold import CognitiveManifold
from prometheus_agent.Mutator import SynthesisError
from prometheus_agent.Warden.EthicalSchema import EthicalJudgment

# This structure assumes the test is run from the project root directory


# --- Mock LLM Responses ---
# This dictionary simulates the structured JSON responses from the LLM for different tasks.
MOCK_LLM_RESPONSES = {
    "domain_classification": {"primary_domain": "coding", "confidence": 0.95, "reasoning": "Mock reasoning."},
    "plan_synthesis": {
        "thought_process": "The user wants to generate and execute code. I will create a three-step plan.",
        "plan": [
            {"step_number": 1, "tool_name": "code_generation", "objective": "Generate Python code.",
             "context_needed": [], "justification": "Initial step."},
            {"step_number": 2, "tool_name": "code_execution", "objective": "Execute the code.",
             "context_needed": ["step_1_output"], "justification": "Run the generated code."},
            {"step_number": 3, "tool_name": "final_synthesis", "objective": "Explain the result.",
             "context_needed": ["step_2_output"], "justification": "Summarize for the user."}
        ]
    },
    "code_generation": "print(123)",
    "final_synthesis": "The code was executed and the result is 123.",
    "fallback_synthesis": "I could not create a plan, but I will try to answer directly."
}


class TestCognitiveManifold(unittest.IsolatedAsyncioTestCase):
    """
    A comprehensive unit test suite for the CognitiveManifold.

    This suite validates the manifold's ability to orchestrate a query from
    classification and planning through to tool execution and final synthesis.
    It tests the "golden path", fallback mechanisms, and error handling in isolation
    using mocked dependencies.
    """

    def setUp(self):
        """Set up a mocked agent environment for the CognitiveManifold."""
        self.mock_agent = MagicMock()

        # Mock all dependencies of the manifold
        self.mock_agent.synthesis_engine = MagicMock()
        self.mock_agent.evaluator = MagicMock()
        self.mock_agent.sandbox_runner = MagicMock()
        self.mock_agent.governor = MagicMock()
        self.mock_agent.geometric_transformer = MagicMock()
        self.mock_agent.ethics_core = MagicMock()

        # Provide default return values for mocked components
        self.mock_agent.governor.decide_model.return_value = MagicMock(model_name="mock-model")
        self.mock_agent.evaluator.evaluate = AsyncMock(
            return_value={"final_score": 9.5, "geometric_state_vector": [9] * 5})
        self.mock_agent.geometric_transformer.get_directives_from_state_vector.return_value = (
        "The Analyst", MagicMock())
        self.mock_agent.ethics_core.get_ethical_system_prompt.return_value = "Ethical Prompt"
        # The agent needs a "loaded" brain to proceed
        type(self.mock_agent).super_brain_data = PropertyMock(return_value={"key": "mock brain data"})

        # Instantiate the class under test
        self.manifold = CognitiveManifold(agent_instance=self.mock_agent)

        # Mock the Warden call within the manifold's tools
        # For unit testing, we assume an approval unless we want to test the veto path.
        self.mock_warden_patch = patch(
            'prometheus_agent.Prometheus.CognitiveManifold.EthicalJudgment',
            return_value=EthicalJudgment(is_approved=True, reasoning="Test approval")
        )
        self.mock_warden_patch.start()

    def tearDown(self):
        self.mock_warden_patch.stop()

    async def test_01_golden_path_full_query_execution(self):
        """
        Test (Golden Path): Verifies a successful query that involves classification,
        multi-step planning, code generation, execution, and final synthesis.
        """

        # --- Arrange ---
        # Configure the synthesis engine to return different data based on the task
        async def mock_synthesis_side_effect(user_objective, **kwargs):
            if "Classify" in user_objective: return MOCK_LLM_RESPONSES["domain_classification"]
            if "plan" in user_objective: return MOCK_LLM_RESPONSES["plan_synthesis"]
            if "Generate" in user_objective: return MOCK_LLM_RESPONSES["code_generation"]
            if "Explain" in user_objective: return MOCK_LLM_RESPONSES["final_synthesis"]
            return "Default mock response"

        self.manifold.synthesis_engine.generate = AsyncMock(side_effect=mock_synthesis_side_effect)

        # Mock the sandbox to return a successful execution
        self.manifold.sandbox.run = AsyncMock(return_value=(True, "123"))
        type(self.manifold.sandbox).is_active = PropertyMock(return_value=True)

        # --- Act ---
        result = await self.manifold.query("Generate and run code, then explain.")

        # --- Assert ---
        # 1. Verify the final response is from the last step of the plan
        self.assertEqual(result["response"], "The code was executed and the result is 123.")
        self.assertEqual(result["classified_domain"], "coding")
        self.assertGreater(result["confidence_score"], 0)

        # 2. Verify the cognitive plan was created and followed
        plan = result["cognitive_plan"]
        self.assertEqual(len(plan["plan"]), 3)
        self.assertEqual(plan["plan"][0]["tool_name"], "code_generation")
        self.assertEqual(plan["plan"][2]["tool_name"], "final_synthesis")

        # 3. Verify dependencies were called
        self.assertEqual(self.manifold.synthesis_engine.generate.call_count, 4)
        self.manifold.sandbox.run.assert_called_once_with("print(123)")
        self.manifold.evaluator.evaluate.assert_called_once()

    async def test_02_planner_failure_fallback(self):
        """
        Test (Fallback Path): Verifies that if the planner fails (returns None),
        the manifold falls back to a direct, single-step synthesis.
        """

        # --- Arrange ---
        # Mock the synthesis engine to fail on planning, but succeed on synthesis
        async def mock_synthesis_side_effect(user_objective, **kwargs):
            if "Classify" in user_objective: return MOCK_LLM_RESPONSES["domain_classification"]
            if "plan" in user_objective: raise SynthesisError("Planner LLM failed")
            # This is the fallback call
            return MOCK_LLM_RESPONSES["fallback_synthesis"]

        self.manifold.synthesis_engine.generate = AsyncMock(side_effect=mock_synthesis_side_effect)

        # --- Act ---
        result = await self.manifold.query("A prompt that will cause planning to fail.")

        # --- Assert ---
        self.assertEqual(result["response"], MOCK_LLM_RESPONSES["fallback_synthesis"])
        # Check that the plan shows it was a fallback
        self.assertIn("Fallback", result["cognitive_plan"]["thought_process"])
        self.assertEqual(len(result["cognitive_plan"]["plan"]), 0)
        # Sandbox should not have been called
        self.manifold.sandbox.run.assert_not_called()

    async def test_03_tool_execution_failure(self):
        """
        Test (Error Handling): Verifies that a failure in a tool during plan
        execution aborts the plan and reports the error.
        """
        # --- Arrange ---
        self.manifold.synthesis_engine.generate = AsyncMock(side_effect=[
            MOCK_LLM_RESPONSES["domain_classification"],
            MOCK_LLM_RESPONSES["plan_synthesis"],
            MOCK_LLM_RESPONSES["code_generation"]
            # No final synthesis call because the plan will abort
        ])

        # Mock the sandbox to return a FAILURE
        self.manifold.sandbox.run = AsyncMock(return_value=(False, "SyntaxError: invalid syntax"))
        type(self.manifold.sandbox).is_active = PropertyMock(return_value=True)

        # --- Act ---
        result = await self.manifold.query("Run code that will fail.")

        # --- Assert ---
        # The final response should be the generic error message
        self.assertIn("critical error", result["response"])
        # Check the thought process to see the partial results
        thought_process = result["full_thought_process"]
        self.assertIn("[FATAL ERROR]", thought_process["step_2_output"])
        self.assertNotIn("step_3_output", thought_process)  # Step 3 should not have run

    @patch('prometheus_agent.Prometheus.CognitiveManifold.EthicalJudgment',
           return_value=EthicalJudgment(is_approved=False, reasoning="VETOED: Code is unsafe."))
    async def test_04_ethical_veto_aborts_plan(self, mock_veto_judgment):
        """
        Test (Safety): Verifies that if the Warden vetoes an action, the plan
        is aborted and the veto reason is reported.
        """
        # --- Arrange ---
        self.manifold.synthesis_engine.generate = AsyncMock(side_effect=[
            MOCK_LLM_RESPONSES["domain_classification"],
            MOCK_LLM_RESPONSES["plan_synthesis"],
            MOCK_LLM_RESPONSES["code_generation"]
        ])
        type(self.manifold.sandbox).is_active = PropertyMock(return_value=True)

        # --- Act ---
        result = await self.manifold.query("Run unsafe code.")

        # --- Assert ---
        self.assertIn("critical error", result["response"])
        thought_process = result["full_thought_process"]
        self.assertIn("[ACTION VETOED]", thought_process["step_2_output"])
        self.assertIn("Code is unsafe", thought_process["step_2_output"])
        self.manifold.sandbox.run.assert_not_called()  # Sandbox must not be run if vetoed


if __name__ == '__main__':
    unittest.main(verbosity=2)
