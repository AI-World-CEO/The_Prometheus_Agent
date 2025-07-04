# prometheus_agent/tests/unit/test_mutator.py

import unittest
import asyncio
import json
from unittest.mock import patch, AsyncMock, MagicMock

from pydantic import BaseModel, Field

# This structure assumes the test is run from the project root directory


# Import all necessary exception types for mocking
from openai import APITimeoutError
import httpx

from prometheus_agent.Mutator import Mutator, SynthesisError


class MockOutputSchema(BaseModel):
    """A simple Pydantic schema for testing JSON output mode."""
    key: str = Field(..., description="A test key.")
    value: int = Field(..., description="A test value.")


# --- Reusable Mock Data ---
MOCK_JSON_PAYLOAD_STR = json.dumps({"key": "test", "value": 123})
MOCK_RAW_PAYLOAD_STR = "def new_function():\n    return True"
MOCK_OLLAMA_JSON_RESPONSE = json.dumps({"choices": [{"message": {"content": MOCK_JSON_PAYLOAD_STR}}]})


class TestMutator(unittest.IsolatedAsyncioTestCase):
    """
    A comprehensive unit test suite for the hyper-resilient Mutator component.

    This suite validates all core logic paths, including provider-specific API calls,
    the robust JSON repair mechanism, retry logic, and error handling, ensuring the
    Mutator is reliable and correct under various conditions.
    """

    def setUp(self):
        """Prepares a fast-retrying config and mock data for each test."""
        self.toolkit_config = {"max_retries": 3, "request_timeout": 5}  # 3 total attempts
        self.openai_config = {"provider": "openai", "openai_api_key": "sk-test"}
        self.local_config = {"provider": "local", "local_api_base_url": "http://localhost:11434"}

    # --- Initialization and Internal Logic Tests ---

    def test_init_openai_provider_success(self):
        """Test (Initialization): Verifies correct initialization for the OpenAI provider."""
        mutator = Mutator(self.openai_config, self.toolkit_config)
        self.assertEqual(mutator.provider, "openai")
        self.assertIsNotNone(mutator.client)

    def test_init_local_provider_url_normalization(self):
        """Test (Initialization): Ensures the local provider's base URL is correctly normalized."""
        mutator = Mutator({"provider": "local", "local_api_base_url": "http://localhost:11434/v1/"},
                          self.toolkit_config)
        self.assertEqual(mutator.base_url, "http://localhost:11434")

    def test_init_openai_provider_no_key_fails(self):
        """Test (Validation): Ensures initialization fails if OpenAI provider is chosen without an API key."""
        with self.assertRaisesRegex(ValueError, "FATAL: Provider is 'openai' but 'openai_api_key' is not set."):
            Mutator({"provider": "openai", "openai_api_key": None}, self.toolkit_config)

    def test_json_repair_and_cleaning(self):
        """Test (Core Feature): Validates the robust JSON extraction logic from messy LLM responses."""
        mutator = Mutator(self.local_config, self.toolkit_config)

        # --- Arrange ---
        messy_json_response = f"""
        Of course! Here is the JSON object you requested.
        ```json
        {MOCK_JSON_PAYLOAD_STR}
        ```
        Let me know if you need anything else!
        """

        # --- Act ---
        cleaned_json = mutator._clean_llm_response(messy_json_response, 'json')

        # --- Assert ---
        self.assertEqual(cleaned_json, MOCK_JSON_PAYLOAD_STR)
        # Verify it's now valid JSON
        self.assertEqual(json.loads(cleaned_json)['key'], 'test')

    # --- Provider-Specific API Call Tests ---

    @patch('asyncio.sleep', new_callable=AsyncMock)
    @patch('openai.resources.chat.completions.Completions.create', new_callable=AsyncMock)
    async def test_openai_generate_json_success(self, mock_openai_call, mock_sleep):
        """Test (OpenAI - Golden Path): Verifies successful generation of a validated JSON object."""
        mutator = Mutator(self.openai_config, self.toolkit_config)
        mock_openai_call.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content=MOCK_JSON_PAYLOAD_STR))])

        result = await mutator.generate("test", "gpt-4", output_schema=MockOutputSchema, output_mode='json')

        self.assertEqual(result, {"key": "test", "value": 123})
        mock_openai_call.assert_called_once()
        self.assertIn('json_object', mock_openai_call.call_args.kwargs['response_format']['type'])
        mock_sleep.assert_not_called()

    @patch('asyncio.sleep', new_callable=AsyncMock)
    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_local_generate_raw_success(self, mock_httpx_post, mock_sleep):
        """Test (Local - Golden Path): Verifies successful generation of raw text."""
        mutator = Mutator(self.local_config, self.toolkit_config)
        mock_response_text = json.dumps({"choices": [{"message": {"content": MOCK_RAW_PAYLOAD_STR}}]})
        mock_httpx_post.return_value = httpx.Response(200, text=mock_response_text)

        result = await mutator.generate("test", "llama3", output_mode='raw')

        self.assertEqual(result, MOCK_RAW_PAYLOAD_STR)
        mock_httpx_post.assert_called_once()
        mock_sleep.assert_not_called()

    # --- Resiliency and Error Handling Tests ---

    @patch('asyncio.sleep', new_callable=AsyncMock)
    @patch('openai.resources.chat.completions.Completions.create', new_callable=AsyncMock)
    async def test_openai_transient_failure_and_retry(self, mock_openai_call, mock_sleep):
        """Test (Resilience): Verifies the retry logic for the OpenAI provider with a transient failure."""
        mutator = Mutator(self.openai_config, self.toolkit_config)
        mock_openai_call.side_effect = [
            APITimeoutError("Request timed out on first attempt"),
            APITimeoutError("Request timed out on second attempt"),
            MagicMock(choices=[MagicMock(message=MagicMock(content=MOCK_JSON_PAYLOAD_STR))])
        ]

        result = await mutator.generate("test", "gpt-4", output_schema=MockOutputSchema, output_mode='json')

        self.assertEqual(result, {"key": "test", "value": 123})
        self.assertEqual(mock_openai_call.call_count, 3, "Should make 3 total attempts.")
        self.assertEqual(mock_sleep.call_count, 2, "Should sleep twice between retries.")

    @patch('asyncio.sleep', new_callable=AsyncMock)
    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_local_persistent_failure_raises_synthesis_error(self, mock_httpx_post, mock_sleep):
        """Test (Resilience): Verifies a persistent local failure correctly raises SynthesisError after all retries."""
        mutator = Mutator(self.local_config, self.toolkit_config)
        mock_httpx_post.side_effect = httpx.ConnectError("Connection refused repeatedly")

        with self.assertRaises(SynthesisError) as context:
            await mutator.generate("test", "llama3", output_mode='raw')

        # With max_retries=3, the loop runs for attempt 0, 1, 2.
        self.assertEqual(mock_httpx_post.call_count, 3, "Should make exactly 3 total attempts.")
        self.assertEqual(mock_sleep.call_count, 2, "Should sleep twice before the final attempt.")
        self.assertIsInstance(context.exception.final_cause, httpx.ConnectError)
        self.assertIn("Synthesis failed after 3 attempts", str(context.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)
