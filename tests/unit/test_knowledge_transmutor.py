# prometheus_agent/tests/unit/test_knowledge_transmutor.py

import unittest
import asyncio
import tempfile
import shutil
import yaml
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from prometheus_agent.KnowledgeTransmutor import KnowledgeTransmuter, TransmutationReport
from prometheus_agent.Mutator import Mutator, SynthesisError

# This structure assumes the test is run from the project root directory


# Mock LLM response that the transmuter expects
MOCK_REFINED_YAML_CONTENT = {
    "thought_process": "This is a mock thought process for a refined document.",
    "generated_content": {
        "id": "mock-refined-id",
        "title": "Mock Refined Title",
        "content": "This is the refined content from the mock LLM."
    }
}


class TestKnowledgeTransmuter(unittest.IsolatedAsyncioTestCase):
    """
    A comprehensive unit test suite for the KnowledgeTransmuter component.

    This suite validates the core logic for the high-performance, concurrent
    transmutation of a text corpus into a structured YAML brain, using a mocked
    agent and a temporary file system for isolated and reliable testing.
    """

    def setUp(self):
        """Set up a temporary file system and a mocked agent environment."""
        # Create a top-level temporary directory for the entire test run
        self.temp_dir = Path(tempfile.mkdtemp(prefix="transmuter_test_"))

        # Create isolated Corpus and YAML_Brain directories inside the temp dir
        self.mock_corpus_path = self.temp_dir / "Corpus"
        self.mock_yaml_brain_path = self.temp_dir / "YAML_Brain"
        self.mock_corpus_path.mkdir()
        self.mock_yaml_brain_path.mkdir()

        # --- Mock the Agent and its dependencies ---
        self.mock_agent = MagicMock()
        self.mock_agent.synthesis_engine = MagicMock(spec=Mutator)
        self.mock_agent.governor = MagicMock()
        self.mock_agent.governor.decide_model.return_value = MagicMock(model_name="mock-model")

        # Instantiate the class we are testing
        self.transmuter = KnowledgeTransmuter(agent_instance=self.mock_agent, max_concurrent_tasks=2)

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.temp_dir)

    async def test_01_successful_transmutation_of_corpus(self):
        """
        Test (Golden Path): Verifies the successful transmutation of multiple valid
        text files into structured YAML files.
        """
        # --- Arrange ---
        # Create some valid source files in the mock corpus
        (self.mock_corpus_path / "file1.txt").write_text("Content of file 1.")
        (self.mock_corpus_path / "subdir").mkdir()
        (self.mock_corpus_path / "subdir" / "file2.txt").write_text("Content of file 2.")

        # Mock the mutator to always return a successful refinement
        self.transmuter.mutator.generate = AsyncMock(return_value=MOCK_REFINED_YAML_CONTENT)

        # --- Act ---
        report = await self.transmuter.transmute_corpus_to_yaml_brain(
            corpus_root_str=str(self.mock_corpus_path),
            yaml_brain_root_str=str(self.mock_yaml_brain_path)
        )

        # --- Assert ---
        # 1. Verify the report accuracy
        self.assertIsInstance(report, TransmutationReport)
        self.assertEqual(report.total_files_processed, 2)
        self.assertEqual(report.success_count, 2)
        self.assertEqual(report.failure_count, 0)
        self.assertEqual(len(report.failures), 0)

        # 2. Verify that the output files were created with the correct content
        output_file1 = self.mock_yaml_brain_path / "file1.yaml"
        output_file2 = self.mock_yaml_brain_path / "subdir" / "file2.yaml"
        self.assertTrue(output_file1.exists())
        self.assertTrue(output_file2.exists())

        with open(output_file1, 'r') as f:
            content = yaml.safe_load(f)
            self.assertEqual(content['title'], "Mock Refined Title")

        # 3. Verify the LLM was called for each file
        self.assertEqual(self.transmuter.mutator.generate.call_count, 2)

    async def test_02_handles_llm_failure_gracefully(self):
        """
        Test (Failure Path): Ensures that if the LLM fails for one file,
        it is correctly logged as a failure, but other files are still processed.
        """
        # --- Arrange ---
        (self.mock_corpus_path / "success.txt").write_text("This one will succeed.")
        (self.mock_corpus_path / "fail.txt").write_text("This one will fail.")

        # Mock the mutator to fail for the file named "fail.txt"
        async def mock_generate_with_failure(user_objective, **kwargs):
            if "fail.txt" in user_objective:
                raise SynthesisError("Simulated LLM API error")
            return MOCK_REFINED_YAML_CONTENT

        self.transmuter.mutator.generate = AsyncMock(side_effect=mock_generate_with_failure)

        # --- Act ---
        report = await self.transmuter.transmute_corpus_to_yaml_brain(
            corpus_root_str=str(self.mock_corpus_path),
            yaml_brain_root_str=str(self.mock_yaml_brain_path)
        )

        # --- Assert ---
        self.assertEqual(report.total_files_processed, 2)
        self.assertEqual(report.success_count, 1)
        self.assertEqual(report.failure_count, 1)
        self.assertEqual(len(report.failures), 1)
        self.assertEqual(report.failures[0].source_path.name, "fail.txt")
        self.assertIn("Simulated LLM API error", report.failures[0].reason)

        # Verify the successful file was created and the failed one was not
        self.assertTrue((self.mock_yaml_brain_path / "success.yaml").exists())
        self.assertFalse((self.mock_yaml_brain_path / "fail.yaml").exists())

    async def test_03_skips_empty_files(self):
        """
        Test (Edge Case): Verifies that empty text files are skipped and logged
        as failures without calling the LLM.
        """
        # --- Arrange ---
        (self.mock_corpus_path / "valid.txt").write_text("Some content.")
        (self.mock_corpus_path / "empty.txt").write_text("  \n\t  ")  # Whitespace only

        self.transmuter.mutator.generate = AsyncMock(return_value=MOCK_REFINED_YAML_CONTENT)

        # --- Act ---
        report = await self.transmuter.transmute_corpus_to_yaml_brain(
            corpus_root_str=str(self.mock_corpus_path),
            yaml_brain_root_str=str(self.mock_yaml_brain_path)
        )

        # --- Assert ---
        self.assertEqual(report.total_files_processed, 2)
        self.assertEqual(report.success_count, 1)
        self.assertEqual(report.failure_count, 1)
        self.assertEqual(report.failures[0].source_path.name, "empty.txt")
        self.assertEqual(report.failures[0].reason, "Source file is empty.")

        # The LLM should only have been called for the one valid file
        self.transmuter.mutator.generate.assert_called_once()

    async def test_04_handles_empty_corpus(self):
        """
        Test (Edge Case): Verifies that the function returns a zero-count report
        when the corpus directory is empty, without error.
        """
        # --- Act ---
        report = await self.transmuter.transmute_corpus_to_yaml_brain(
            corpus_root_str=str(self.mock_corpus_path),
            yaml_brain_root_str=str(self.mock_yaml_brain_path)
        )

        # --- Assert ---
        self.assertEqual(report.total_files_processed, 0)
        self.assertEqual(report.success_count, 0)
        self.assertEqual(report.failure_count, 0)


if __name__ == '__main__':
    unittest.main()
