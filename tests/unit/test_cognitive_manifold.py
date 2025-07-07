# prometheus_agent/tests/unit/test_cognitive_manifold.py

import unittest
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from prometheus_agent.CognitiveManifold import CognitiveManifold, CognitivePlan
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
    "code_generation": "print('Success!')",
    "final_synthesis": "The code was executed and the result is: Success!",
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

        # Mock all dependencies of the manifold that are accessed via self.agent
        self.mock_agent.synthesis_engine = AsyncMock()
        self.mock_agent.evaluator = AsyncMock()
        self.mock_agent.sandbox_runner = AsyncMock()
        self.mock_agent.governor = AsyncMock()
        self.mock_agent.geometric_transformer = MagicMock()
        self.mock_agent.ethics_core = MagicMock()

        # Provide default return values for mocked components
        self.mock_agent.governor.decide_model.return_value = MagicMock(model_name="mock-model", timeout_seconds=10)
        self.mock_agent.evaluator.evaluate.return_value = {
            "final_score": 9.5, "geometric_state_vector": [9] * 5}
        self.mock_agent.geometric_transformer.get_directives_from_state_vector.return_value = (
            "The Analyst", MagicMock(model_dump=lambda: {}))
        self.mock_agent.ethics_core.get_ethical_system_prompt.return_value = "Ethical Prompt"
        type(self.mock_agent).super_brain_data = PropertyMock(return_value={"key": "mock brain data"})

        # Correctly mock the sandbox runner methods on the agent object
        self.mock_agent.sandbox_runner.run = AsyncMock(return_value=(True, "Success!"))
        type(self.mock_agent.sandbox_runner).is_active = PropertyMock(return_value=True)

        # Instantiate the class under test
        self.manifold = CognitiveManifold(agent_instance=self.mock_agent)

        # --- THE FIX: Correctly patch the judge method on the manifold's *instance* of the Warden ---
        self.mock_warden_judge_patch = patch.object(self.manifold.warden, 'judge', new_callable=AsyncMock)
        self.mock_warden_judge = self.mock_warden_judge_patch.start()

    def tearDown(self):
        self.mock_warden_judge_patch.stop()

    async def test_01_golden_path_full_query_execution(self):
        """
        Test (Golden Path): Verifies a successful query that involves classification,
        multi-step planning, code generation, execution, and final synthesis.
        """
        # --- Arrange ---
        # Warden approves the action
        self.mock_warden_judge.return_value = EthicalJudgment(is_approved=True, reasoning="Test approval")

        async def mock_synthesis_side_effect(user_objective, **kwargs):
            if "Classify" in user_objective: return MOCK_LLM_RESPONSES["domain_classification"]
            if "plan" in user_objective: return MOCK_LLM_RESPONSES["plan_synthesis"]
            if "Generate" in user_objective: return MOCK_LLM_RESPONSES["code_generation"]
            if "Synthesize" in user_objective: return MOCK_LLM_RESPONSES["final_synthesis"]
            return "Default mock response"

        self.mock_agent.synthesis_engine.generate.side_effect = mock_synthesis_side_effect

        # --- Act ---
        result = await self.manifold.query("Generate and run code, then explain.")

        # --- Assert ---
        self.assertEqual(result["response"], "The code was executed and the result is: Success!")
        self.assertEqual(result["classified_domain"], "coding")
        self.assertGreater(result["confidence_score"], 0)

        plan = result["cognitive_plan"]
        self.assertEqual(len(plan["plan"]), 3)
        self.assertEqual(plan["plan"][0]["tool_name"], "code_generation")
        self.assertEqual(plan["plan"][2]["tool_name"], "final_synthesis")

        self.assertEqual(self.mock_agent.synthesis_engine.generate.call_count, 4)
        self.mock_warden_judge.assert_awaited_once()
        self.mock_agent.sandbox_runner.run.assert_awaited_once_with("print('Success!')")
        self.mock_agent.evaluator.evaluate.assert_awaited_once()

    async def test_02_planner_failure_fallback(self):
        """
        Test (Fallback Path): Verifies that if the planner fails (e.g., LLM error),
        the manifold falls back to a direct, single-step synthesis.
        """

        # --- Arrange ---
        async def mock_synthesis_side_effect(user_objective, **kwargs):
            if "Classify" in user_objective: return MOCK_LLM_RESPONSES["domain_classification"]
            if "plan" in user_objective: raise SynthesisError("Planner LLM failed")
            return MOCK_LLM_RESPONSES["fallback_synthesis"]

        self.mock_agent.synthesis_engine.generate.side_effect = mock_synthesis_side_effect

        # --- Act ---
        result = await self.manifold.query("A prompt that will cause planning to fail.")

        # --- Assert ---
        self.assertEqual(result["response"], MOCK_LLM_RESPONSES["fallback_synthesis"])
        self.assertIn("Fallback: Primary planner failed", result["cognitive_plan"]["thought_process"])
        self.assertEqual(len(result["cognitive_plan"]["plan"]), 0)
        self.mock_agent.sandbox_runner.run.assert_not_called()

    async def test_03_tool_execution_failure(self):
        """
        Test (Error Handling): Verifies that a failure in a tool during plan
        execution aborts the plan and reports the error.
        """
        # --- Arrange ---
        self.mock_warden_judge.return_value = EthicalJudgment(is_approved=True, reasoning="Test approval")
        self.mock_agent.synthesis_engine.generate.side_effect = [
            MOCK_LLM_RESPONSES["domain_classification"],
            MOCK_LLM_RESPONSES["plan_synthesis"],
            MOCK_LLM_RESPONSES["code_generation"]
        ]
        # Mock the sandbox to return a FAILURE
        self.mock_agent.sandbox_runner.run.return_value = (False, "SyntaxError: invalid syntax")

        # --- Act ---
        result = await self.manifold.query("Run code that will fail.")

        # --- Assert ---
        self.assertIn("critical error", result["response"])
        thought_process = result["full_thought_process"]
        self.assertIn("Execution failed", thought_process["step_2_output"])
        self.assertNotIn("step_3_output", thought_process)  # Step 3 should not have run

    async def test_04_ethical_veto_aborts_plan(self):
        """
        Test (Safety): Verifies that if the Warden vetoes an action, the plan
        is aborted and the veto reason is reported.
        """
        # --- Arrange ---
        # Configure the warden mock to return a veto
        self.mock_warden_judge.return_value = EthicalJudgment(is_approved=False, reasoning="VETOED: Code is unsafe.")

        self.mock_agent.synthesis_engine.generate.side_effect = [
            MOCK_LLM_RESPONSES["domain_classification"],
            MOCK_LLM_RESPONSES["plan_synthesis"],
            MOCK_LLM_RESPONSES["code_generation"]
        ]

        # --- Act ---
        result = await self.manifold.query("Run unsafe code.")

        # --- Assert ---
        self.assertIn("critical error", result["response"])
        thought_process = result["full_thought_process"]
        # The tool's output should contain the veto message
        self.assertIn("[ACTION VETOED]", thought_process["step_2_output"])
        self.assertIn("Code is unsafe", thought_process["step_2_output"])
        # Sandbox must not be run if vetoed
        self.mock_agent.sandbox_runner.run.assert_not_called()
        self.mock_warden_judge.assert_awaited_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)