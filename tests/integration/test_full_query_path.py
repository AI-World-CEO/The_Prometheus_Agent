# prometheus_agent/tests/integration/test_full_query_path.py

import unittest
import asyncio
from unittest.mock import patch, AsyncMock

from prometheus_agent.PrometheusAgent import PrometheusAgent

# This structure assumes the test is run from the project root directory


# --- Mock LLM Responses ---
# This dictionary simulates the structured JSON responses from the LLM for different tasks.
# This allows us to control the "mind" of the agent for the duration of the test.

MOCK_LLM_RESPONSES = {
    # 1. The response when asked to classify the domain of the prompt
    "domain_classification": {
        "primary_domain": "coding",
        "confidence": 0.95,
        "reasoning": "The prompt contains keywords like 'Python', 'function', and 'calculate'."
    },
    # 2. The response when asked to create a plan
    "plan_synthesis": {
        "thought_process": "The user wants to generate code, execute it, and then explain it. I will create a three-step plan.",
        "plan": [
            {
                "step_number": 1,
                "tool_name": "code_generation",
                "objective": "Generate a Python function to calculate the factorial of 5.",
                "context_needed": ["initial_prompt"],
                "justification": "First, create the required code."
            },
            {
                "step_number": 2,
                "tool_name": "code_execution",
                "objective": "Execute the generated factorial function to get the result.",
                "context_needed": ["step_1_output"],
                "justification": "Next, run the code to get the answer."
            },
            {
                "step_number": 3,
                "tool_name": "final_synthesis",
                "objective": "Explain what a factorial is and state the result from the execution.",
                "context_needed": ["initial_prompt", "step_1_output", "step_2_output"],
                "justification": "Finally, synthesize all information into a coherent answer."
            }
        ]
    },
    # 3. The response when asked to generate code (tool call)
    "code_generation": "def factorial(n):\n    if n == 0: return 1\n    return n * factorial(n-1)\n\nresult = factorial(5)\nprint(result)",

    # 4. The response when asked for the final explanation (tool call)
    "final_synthesis": "A factorial (n!) is the product of all positive integers up to n. The code to calculate this was `def factorial...`. Based on the execution in the sandbox, the result of 5! is 120."
}


class TestFullQueryPath(unittest.IsolatedAsyncioTestCase):
    """
    An integration test to audit the integrity and performance of the
    Prometheus Agent's entire cognitive process, from user prompt to final response.

    This test validates the "cognitive chain":
    Manifold -> Governor -> Mutator -> Sandbox -> Evaluator -> GeometricTransformer -> Final Response
    """

    @patch('prometheus_agent.SandboxRunner.SandboxRunner.run', new_callable=AsyncMock)
    @patch('prometheus_agent.Mutator.Mutator.generate', new_callable=AsyncMock)
    @patch('prometheus_agent.PrometheusAgent._reload_super_brain')  # Prevent file access
    async def test_complex_multi_step_query_path(self, mock_reload_brain, mock_llm_generate, mock_sandbox_run):
        """
        Tests a complex query that requires the CognitiveManifold to classify, plan,
        generate code, execute it, and synthesize a final explanation.
        """
        # --- 1. MOCK CONFIGURATION ---
        mock_reload_brain.return_value = {"mock_brain": "loaded"}  # Ensure brain is "loaded"

        # This complex side effect simulates the LLM returning different structured
        # data depending on the task it's asked to perform.
        async def llm_side_effect(user_objective, **kwargs):
            if "classify" in user_objective:
                return MOCK_LLM_RESPONSES["domain_classification"]
            elif "create a step-by-step JSON plan" in user_objective:
                return MOCK_LLM_RESPONSES["plan_synthesis"]
            elif "Generate a Python function" in user_objective:
                return MOCK_LLM_RESPONSES["code_generation"]
            elif "Explain what a factorial is" in user_objective:
                return MOCK_LLM_RESPONSES["final_synthesis"]
            return {"thought_process": "Default mock response", "generated_content": "Default"}

        mock_llm_generate.side_effect = llm_side_effect

        # Configure the mock sandbox to return a successful execution result.
        mock_sandbox_run.return_value = (True, "120")

        # --- 2. TEST EXECUTION ---
        # Initialize the full agent. It will use mocks for file/network access.
        agent = PrometheusAgent()

        complex_prompt = "Please write a Python function to calculate 5!, run it, and tell me the result and what a factorial is."

        # Run the "reflexive thought" task, which triggers the full manifold query path
        start_time = asyncio.get_event_loop().time()
        result = await agent.reflexive_thought(complex_prompt)
        execution_time = asyncio.get_event_loop().time() - start_time

        print(f"\nEnd-to-end cognitive chain audit executed in {execution_time:.4f} seconds.")

        # --- 3. ASSERTIONS ---
        # 3.1: Validate the final output's structure and content
        self.assertIsNotNone(result)
        self.assertIn("response", result)
        self.assertIn("confidence_score", result)
        self.assertIn("persona_archetype", result)
        self.assertIn("cognitive_plan", result)

        # Check that the final response is the result of the LAST step in the chain
        self.assertIn("The result of 5! is 120", result["response"])

        # 3.2: Validate the integrity of the cognitive chain
        self.assertEqual(result["classified_domain"], "coding")
        cognitive_plan = result["cognitive_plan"]
        self.assertEqual(len(cognitive_plan['plan']), 3)
        self.assertEqual(cognitive_plan['plan'][0]['tool_name'], "code_generation")
        self.assertEqual(cognitive_plan['plan'][1]['tool_name'], "code_execution")
        self.assertEqual(cognitive_plan['plan'][2]['tool_name'], "final_synthesis")

        # 3.3: Validate that mocks were called as expected
        self.assertEqual(mock_llm_generate.call_count, 4,
                         "Expected LLM to be called for: 1. Domain Classification, 2. Planning, 3. Code-Gen, 4. Final Synthesis.")
        mock_sandbox_run.assert_called_once()
        # Check that the code generated in step 1 was the code executed in step 2
        executed_code = mock_sandbox_run.call_args[0][0]
        self.assertIn("def factorial(n):", executed_code)

        # 3.4: Validate that downstream components were engaged
        self.assertGreater(result['confidence_score'], 0, "Evaluator must have run to produce a confidence score.")
        self.assertIsNotNone(result['persona_archetype'], "GeometricTransformer must have run to produce a persona.")

        # 3.5: Assert performance in a mocked environment
        self.assertLess(execution_time, 1.0, "The entire non-blocking query path should be very fast with mocked I/O.")


if __name__ == '__main__':
    # This allows running the test directly from the command line
    unittest.main(verbosity=2)
