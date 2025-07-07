# prometheus_agent/tests/unit/test_mutator.py

import unittest
import asyncio
import json
from unittest.mock import patch, AsyncMock, MagicMock

from pydantic import BaseModel, Field

# This structure assumes the test is run from the project root directory

# Import all necessary exception types for mocking
try:
    from openai import APITimeoutError, APIError
    import httpx
except ImportError:
    # Create dummy classes if the libraries aren't installed in the test environment
    class APITimeoutError(Exception):
        pass


    class APIError(Exception):
        pass


    class httpx:
        class ConnectError(Exception):
            pass

        class ReadTimeout(Exception):
            pass

        class Request:
            def __init__(self, *args, **kwargs):
                pass

        class Response:
            def __init__(self, status_code, text="", request=None):
                self.status_code = status_code
                self.text = text
                self.request = request

            def json(self):
                return json.loads(self.text)

            def raise_for_status(self):
                if 400 <= self.status_code < 600:
                    raise httpx.ConnectError(f"Client error '{self.status_code}'")

from prometheus_agent.Mutator import Mutator, SynthesisError


class MockOutputSchema(BaseModel):
    """A simple Pydantic schema for testing JSON output mode."""
    key: str = Field(..., description="A test key.")
    value: int = Field(..., description="A test value.")


# --- Reusable Mock Data ---
MOCK_JSON_PAYLOAD_STR = json.dumps({"key": "test", "value": 123})
MOCK_RAW_PAYLOAD_STR = "def new_function():\n    return True"
MOCK_OLLAMA_RAW_RESPONSE = json.dumps({"choices": [{"message": {"content": MOCK_RAW_PAYLOAD_STR}}]})
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
        self.toolkit_config = {"max_retries": 3, "request_timeout": 5, "default_request_timeout": 5}
        self.openai_config = {"provider": "openai", "openai_api_key": "sk-test"}
        self.local_config = {"provider": "local", "local_api_base_url": "http://localhost:11434/v1"}

    # --- Initialization and Internal Logic Tests ---

    def test_init_openai_provider_success(self):
        """Test (Initialization): Verifies correct initialization for the OpenAI provider."""
        # --- THE FIX ---
        # The correct patch target is the actual import path of the class
        with patch('openai.AsyncOpenAI') as mock_openai_client:
            mutator = Mutator(self.openai_config, self.toolkit_config)
            self.assertEqual(mutator.provider, "openai")
            self.assertIsNotNone(mutator.client)
            mock_openai_client.assert_called_once()

    def test_init_local_provider_url_normalization(self):
        """Test (Initialization): Ensures the local provider's base URL is correctly normalized."""
        mutator = Mutator(self.local_config, self.toolkit_config)
        self.assertEqual(mutator.base_url, "http://localhost:11434")

    def test_init_openai_provider_no_key_fails(self):
        """Test (Validation): Ensures initialization fails if OpenAI provider is chosen without an API key."""
        with self.assertRaisesRegex(ValueError, "FATAL: Provider is 'openai' but 'openai_api_key' is not set."):
            Mutator({"provider": "openai", "openai_api_key": None}, self.toolkit_config)

    def test_json_repair_and_cleaning(self):
        """Test (Core Feature): Validates the robust JSON extraction logic from messy LLM responses."""
        mutator = Mutator(self.local_config, self.toolkit_config)
        messy_json_response = f"Sure, here you go:\n```json\n{MOCK_JSON_PAYLOAD_STR}\n```\nEnjoy!"
        cleaned_json = mutator._clean_llm_response(messy_json_response, 'json')
        self.assertEqual(cleaned_json, MOCK_JSON_PAYLOAD_STR)

    # --- Provider-Specific API Call Tests ---

    @patch('asyncio.sleep', new_callable=AsyncMock)
    @patch('openai.AsyncOpenAI', new_callable=MagicMock)  # Mock the class itself
    async def test_openai_generate_json_success(self, mock_openai_class, mock_sleep):
        """Test (OpenAI - Golden Path): Verifies successful generation of a validated JSON object."""
        # Create an instance of the mutator, which will create a mocked client instance
        mutator = Mutator(self.openai_config, self.toolkit_config)
        # Configure the 'create' method on the mocked client instance
        mutator.client.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(content=MOCK_JSON_PAYLOAD_STR))])
        )

        result = await mutator.generate("test", "gpt-4", output_schema=MockOutputSchema, output_mode='json')

        self.assertEqual(result, {"key": "test", "value": 123})
        mutator.client.chat.completions.create.assert_awaited_once()
        self.assertIn('json_object', mutator.client.chat.completions.create.call_args.kwargs['response_format']['type'])
        mock_sleep.assert_not_called()

    @patch('asyncio.sleep', new_callable=AsyncMock)
    @patch('httpx.AsyncClient')
    async def test_local_generate_raw_success(self, mock_async_client, mock_sleep):
        """Test (Local - Golden Path): Verifies successful generation of raw text."""
        mutator = Mutator(self.local_config, self.toolkit_config)

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = json.loads(MOCK_OLLAMA_RAW_RESPONSE)

        # Mock the async context manager
        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_async_client.return_value.__aenter__.return_value = mock_client_instance

        result = await mutator.generate("test", "llama3", output_mode='raw')

        self.assertEqual(result, MOCK_RAW_PAYLOAD_STR)
        mock_client_instance.post.assert_awaited_once()
        mock_sleep.assert_not_called()

    # --- Resiliency and Error Handling Tests ---

    @patch('asyncio.sleep', new_callable=AsyncMock)
    @patch('openai.AsyncOpenAI', new_callable=MagicMock)
    async def test_openai_transient_failure_and_retry(self, mock_openai_class, mock_sleep):
        """Test (Resilience): Verifies the retry logic for the OpenAI provider with a transient failure."""
        mutator = Mutator(self.openai_config, self.toolkit_config)
        mock_create = AsyncMock(side_effect=[
            APITimeoutError("Request timed out on first attempt"),
            MagicMock(choices=[MagicMock(message=MagicMock(content=MOCK_JSON_PAYLOAD_STR))])
        ])
        mutator.client.chat.completions.create = mock_create

        result = await mutator.generate("test", "gpt-4", output_schema=MockOutputSchema, output_mode='json')

        self.assertEqual(result, {"key": "test", "value": 123})
        self.assertEqual(mock_create.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)

    @patch('asyncio.sleep', new_callable=AsyncMock)
    @patch('httpx.AsyncClient')
    async def test_local_persistent_failure_raises_synthesis_error(self, mock_async_client, mock_sleep):
        """Test (Resilience): Verifies a persistent local failure correctly raises SynthesisError after all retries."""
        mutator = Mutator(self.local_config, self.toolkit_config)

        mock_client_instance = AsyncMock()
        mock_client_instance.post.side_effect = httpx.ConnectError("Connection refused repeatedly")
        mock_async_client.return_value.__aenter__.return_value = mock_client_instance

        with self.assertRaises(SynthesisError) as context:
            await mutator.generate("test", "llama3", output_mode='raw')

        self.assertEqual(mock_client_instance.post.call_count, self.toolkit_config["max_retries"])
        self.assertEqual(mock_sleep.call_count, self.toolkit_config["max_retries"] - 1)
        self.assertIsInstance(context.exception.final_cause, httpx.ConnectError)
        self.assertIn(f"Synthesis failed after {self.toolkit_config['max_retries']} attempts", str(context.exception))