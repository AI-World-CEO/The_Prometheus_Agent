# tests/unit/test_audio_engine.py

import unittest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

# --- Correct, Absolute Imports from the Main Source Package ---
from prometheus_agent.AudioEngine import AudioEngine


# --- THE FIX: High-Fidelity Mock of the External Error Class ---
# This is a functional "stunt double" for the real elevenlabs.APIError.
# It allows our tests to simulate realistic API failures without needing
# to import the actual library, keeping the test isolated.
class MockAPIError(Exception):
    """A high-fidelity mock of elevenlabs.client.APIError for isolated testing."""

    def __init__(self, message: str, request=None, response=None, body=None):
        super().__init__(message)
        self.message = message
        self.request = request or MagicMock()
        self.response = response or MagicMock()
        self.body = body or {}

    def __str__(self):
        return self.message


class TestAudioEngine(unittest.IsolatedAsyncioTestCase):
    """
    A comprehensive unit test suite for the AudioEngine.

    This suite validates the engine's ability to correctly handle different
    output modes (streaming, non-streaming, file saving) and to gracefully
    manage API errors and invalid inputs. All external dependencies are mocked.
    """

    def setUp(self):
        """Set up a mock audio stream for the tests."""

        async def mock_audio_stream_generator():
            yield b"chunk1"
            yield b"chunk2"
            yield b"chunk3"

        self.mock_stream_generator = mock_audio_stream_generator

    def test_01_inactive_engine_does_not_fail(self):
        """Tests that an engine initialized without an API key fails gracefully."""
        inactive_engine = AudioEngine(api_key=None)
        try:
            asyncio.run(inactive_engine.speak("test"))
        except Exception as e:
            self.fail(f"Inactive engine raised an unexpected exception: {e}")

    @patch('prometheus_agent.AudioEngine.stream')
    @patch('prometheus_agent.AudioEngine.AsyncElevenLabs.generate', new_callable=AsyncMock)
    async def test_02_speak_streaming_mode(self, mock_generate, mock_stream):
        """Test (Golden Path): Verifies streaming mode calls the correct functions."""
        engine = AudioEngine(api_key="sk-dummy")
        mock_generate.return_value = self.mock_stream_generator()
        await engine.speak("Hello world", use_streaming=True)

        mock_generate.assert_awaited_once_with(text='Hello world', voice='Rachel', model='eleven_multilingual_v2',
                                               stream=True)
        mock_stream.assert_called_once()
        self.assertTrue(hasattr(mock_stream.call_args[0][0], '__anext__'))

    @patch('prometheus_agent.AudioEngine.play')
    @patch('prometheus_agent.AudioEngine.AsyncElevenLabs.generate', new_callable=AsyncMock)
    async def test_03_speak_non_streaming_mode(self, mock_generate, mock_play):
        """Verifies non-streaming mode collects all chunks before playing."""
        engine = AudioEngine(api_key="sk-dummy")
        mock_generate.return_value = self.mock_stream_generator()
        await engine.speak("Hello world", use_streaming=False)

        mock_generate.assert_awaited_once()
        mock_play.assert_called_once_with(b"chunk1chunk2chunk3")

    @patch('prometheus_agent.AudioEngine.save')
    @patch('prometheus_agent.AudioEngine.AsyncElevenLabs.generate', new_callable=AsyncMock)
    async def test_04_speak_save_to_file_mode(self, mock_generate, mock_save):
        """Verifies file-saving mode collects chunks and calls save correctly."""
        engine = AudioEngine(api_key="sk-dummy")
        mock_generate.return_value = self.mock_stream_generator()
        output_file = "/tmp/test_output.mp3"
        await engine.speak("Hello world", output_path=output_file)

        mock_generate.assert_awaited_once()
        mock_save.assert_called_once_with(
            audio_data=b"chunk1chunk2chunk3",
            filename=output_file
        )

    @patch('prometheus_agent.AudioEngine.AsyncElevenLabs.generate', new_callable=AsyncMock)
    async def test_05_api_error_handling(self, mock_generate):
        """Tests that a simulated APIError is caught gracefully."""
        engine = AudioEngine(api_key="sk-dummy")
        # Configure the mock to raise our functional MockAPIError
        mock_generate.side_effect = MockAPIError(
            message="Simulated API failure",
            body={"detail": {"message": "Invalid API key"}}
        )

        try:
            await engine.speak("This will fail")
        except Exception as e:
            self.fail(f"Engine speak method raised an unhandled exception on APIError: {e}")

        mock_generate.assert_awaited_once()

    async def test_06_invalid_text_input(self):
        """Tests that providing empty or invalid text does not result in an API call."""
        engine = AudioEngine(api_key="sk-dummy")
        # We need a mock generate to assert it's NOT called
        engine.client.generate = AsyncMock()

        await engine.speak("")
        await engine.speak("   \n\t  ")
        await engine.speak(None)

        engine.client.generate.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
