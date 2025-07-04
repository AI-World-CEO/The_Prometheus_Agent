# prometheus_agent/tests/chaos/test_llm_api_failure_resilience.py

import unittest
import asyncio
import json
from unittest.mock import patch, AsyncMock, MagicMock

# This structure assumes the test is run from the project root directory
from pydantic import BaseModel, Field


# Import all necessary exception types for mocking
from openai import APITimeoutError, APIStatusError
import httpx

from prometheus_agent.Mutator import Mutator, SynthesisError

# A mock response object required by the openai library for status errors
MOCK_500_RESPONSE = httpx.Response(status_code=500, json={"error": {"message": "Internal Server Error"}})


# A simple Pydantic model for testing JSON output
class MockSuccessSchema(BaseModel):
    thought_process: str
    generated_content: str


MOCK_JSON_SUCCESS_STR = MockSuccessSchema(thought_process="mock thought", generated_content="success").model_dump_json()
MOCK_OLLAMA_SUCCESS_BODY = json.dumps({"choices": [{"message": {"content": MOCK_JSON_SUCCESS_STR}}]})


class TestLLMApiFailureResilience(unittest.IsolatedAsyncioTestCase):
    """
    A Chaos Engineering test to validate the resilience of the Mutator component.

    This test simulates an unreliable network and a failing LLM API to ensure
    the Mutator's retry logic, error handling, and timeout mechanisms function
    as a robust defense against external service degradation for ALL supported providers.
    """

    def setUp(self):
        """Prepares a fast-retrying config for testing purposes."""
        # FIX: Use 3 attempts for a more realistic resilience test.
        # The Mutator interprets this as 3 total attempts (1 initial + 2 retries).
        self.test_toolkit_config = {"max_retries": 3, "request_timeout": 5}

    # =================================================================
    # ==                 TESTS FOR 'OPENAI' PROVIDER                 ==
    # =================================================================

    def get_openai_mutator(self):
        """Helper to create a Mutator instance configured for OpenAI."""
        model_config = {"provider": "openai", "openai_api_key": "sk-dummy-key-for-testing"}
        return Mutator(model_config=model_config, toolkit_config=self.test_toolkit_config)

    @patch('asyncio.sleep', new_callable=AsyncMock)
    @patch('openai.resources.chat.completions.Completions.create', new_callable=AsyncMock)
    async def test_openai_resilience_to_transient_failures(self, mock_openai_call, mock_sleep):
        """
        Test (OpenAI): Simulates a sequence of recoverable API errors, followed by success.
        Asserts that the Mutator retries correctly and eventually succeeds.
        """
        # --- Arrange ---
        mutator = self.get_openai_mutator()
        mock_openai_call.side_effect = [
            APITimeoutError("Request timed out"),  # First attempt fails
            APIStatusError("Internal Server Error", response=MOCK_500_RESPONSE, body=None),  # Second attempt fails
            # Third attempt succeeds
            MagicMock(choices=[MagicMock(message=MagicMock(content=MOCK_JSON_SUCCESS_STR))])
        ]

        # --- Act ---
        result = await mutator.generate("test", "gpt-4", output_schema=MockSuccessSchema, output_mode='json')

        # --- Assert ---
        self.assertIsNotNone(result)
        self.assertEqual(result.get("generated_content"), "success")
        self.assertEqual(mock_openai_call.call_count, 3, "Should have failed twice and succeeded on the third try.")
        self.assertEqual(mock_sleep.call_count, 2, "Should have slept twice between failed attempts.")

    @patch('asyncio.sleep', new_callable=AsyncMock)
    @patch('openai.resources.chat.completions.Completions.create', new_callable=AsyncMock)
    async def test_openai_graceful_failure_after_max_retries(self, mock_openai_call, mock_sleep):
        """
        Test (OpenAI): Simulates a persistent API failure.
        Asserts that the Mutator gives up after max retries and raises a SynthesisError.
        """
        # --- Arrange ---
        mutator = self.get_openai_mutator()
        mutator.max_retries = 3  # Explicit for clarity
        mock_openai_call.side_effect = APITimeoutError("Request timed out consistently")

        # --- Act & Assert ---
        with self.assertRaises(SynthesisError) as context:
            await mutator.generate("test", "gpt-4", output_schema=MockSuccessSchema, output_mode='json')

        # Validate call counts and exception details
        # FIX: The number of calls should equal max_retries (which means total attempts).
        self.assertEqual(mock_openai_call.call_count, mutator.max_retries, "Should have made exactly 3 attempts.")
        self.assertIn(f"failed after {mutator.max_retries} attempts", str(context.exception))
        self.assertIsInstance(context.exception.final_cause, APITimeoutError)

    # =================================================================
    # ==                  TESTS FOR 'LOCAL' PROVIDER                 ==
    # =================================================================

    def get_local_mutator(self):
        """Helper to create a Mutator instance configured for a local (Ollama) provider."""
        model_config = {"provider": "local", "local_api_base_url": "http://mock-ollama:11434"}
        return Mutator(model_config=model_config, toolkit_config=self.test_toolkit_config)

    @patch('asyncio.sleep', new_callable=AsyncMock)
    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_local_resilience_to_transient_failures(self, mock_httpx_post, mock_sleep):
        """
        Test (Local): Simulates a sequence of recoverable local API errors, followed by success.
        Asserts that the Mutator retries correctly.
        """
        # --- Arrange ---
        mutator = self.get_local_mutator()
        mock_httpx_post.side_effect = [
            httpx.ReadTimeout("Unable to read from server"),  # First attempt fails
            httpx.Response(status_code=503, reason_phrase="Service Unavailable"),  # Second attempt fails
            # Third attempt succeeds
            httpx.Response(status_code=200, text=MOCK_OLLAMA_SUCCESS_BODY)
        ]

        # --- Act ---
        result = await mutator.generate("test", "llama3", output_schema=MockSuccessSchema, output_mode='json')

        # --- Assert ---
        self.assertIsNotNone(result)
        self.assertEqual(result.get("generated_content"), "success")
        self.assertEqual(mock_httpx_post.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch('asyncio.sleep', new_callable=AsyncMock)
    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_local_graceful_failure_after_max_retries(self, mock_httpx_post, mock_sleep):
        """
        Test (Local): Simulates a persistent local API failure.
        Asserts that the Mutator gives up and raises a SynthesisError.
        """
        # --- Arrange ---
        mutator = self.get_local_mutator()
        mutator.max_retries = 3  # Explicit for clarity
        mock_httpx_post.side_effect = httpx.ConnectError("Connection refused")

        # --- Act & Assert ---
        with self.assertRaises(SynthesisError) as context:
            await mutator.generate("test", "llama3", output_schema=MockSuccessSchema, output_mode='json')

        # FIX: Validate the correct number of calls
        self.assertEqual(mock_httpx_post.call_count, mutator.max_retries, "Should have made exactly 3 attempts.")
        self.assertIn(f"failed after {mutator.max_retries} attempts", str(context.exception))

        # FIX: The final cause MUST be the original httpx exception, not a generic RuntimeError.
        self.assertIsInstance(context.exception.final_cause, httpx.ConnectError)
        self.assertIn("Connection refused", str(context.exception.final_cause))


if __name__ == '__main__':
    unittest.main(verbosity=2)
