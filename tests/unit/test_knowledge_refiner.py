# prometheus_agent/tests/unit/test_knowledge_refiner.py

import unittest
import asyncio
import tempfile
import shutil
import yaml
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from prometheus_agent.KnowledgeRefiner import KnowledgeRefiner
from prometheus_agent.Mutator import Mutator, SynthesisError

# This structure assumes the test is run from the project root directory


# --- Raw Unstructured Knowledge to be Ingested ---
RAW_KNOWLEDGE_TEXT = """
Title: The Principle of Abstraction in Software
Abstract: Abstraction is the process of hiding the complexity of a system while exposing only the necessary parts. It allows developers to create reusable components and manage large codebases effectively. A key benefit is the decoupling of interface from implementation.
"""

# --- Mock LLM Responses ---
MOCK_LLM_RESPONSES = {
    "placement": {
        "domain_path": "2_cognitive_architecture/2.2_architectural_principles",
        "file_name": "Axiom_2.2.2_principle_of_abstraction.yaml",
        "reasoning": "The text describes a core software architectural principle."
    },
    "refinement": """
id: arch-abs-001
title: The Principle of Abstraction
type: foundational-software-principle
domain: cognitive_architecture
abstract: |
  Abstraction is the process of hiding the complexity of a system while exposing only the necessary parts. It allows developers to create reusable components and manage large codebases effectively.
key-concepts:
  - name: Interface-Implementation Decoupling
    description: "The primary benefit of abstraction, allowing an implementation to change without affecting the components that use its interface."
  - name: Complexity Hiding
    description: "Simplifies interaction by concealing unnecessary details."
"""
}


class TestKnowledgeRefiner(unittest.IsolatedAsyncioTestCase):
    """
    A comprehensive unit test suite for the KnowledgeRefiner component.

    This suite validates the core logic for autonomous knowledge ingestion and
    continuous refinement, using a mocked agent and a temporary file system
    to ensure isolated, fast, and reliable testing.
    """

    def setUp(self):
        """Set up a temporary file system and a mocked agent environment."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="refiner_test_"))
        self.mock_yaml_brain_path = self.temp_dir / "YAML_Brain"
        self.mock_yaml_brain_path.mkdir()

        # Create a "golden standard" exemplar file for the refiner to use
        exemplar_dir = self.mock_yaml_brain_path / "2_cognitive_architecture" / "2.2_architectural_principles"
        exemplar_dir.mkdir(parents=True)
        (exemplar_dir / "Axiom_2.2.1_clean_architecture.yaml").write_text("id: example\ntitle: Example\n...")

        # --- Mock the Agent and its dependencies ---
        self.mock_agent = MagicMock()
        self.mock_agent.synthesis_engine = MagicMock(spec=Mutator)
        self.mock_agent.governor = MagicMock()
        self.mock_agent.governor.decide_model.return_value = MagicMock(model_name="mock-model")
        # Point the agent's path attribute to our temporary directory
        self.mock_agent.yaml_brain_path = self.mock_yaml_brain_path

        # Instantiate the class we are testing
        self.refiner = KnowledgeRefiner(agent_instance=self.mock_agent)

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.temp_dir)

    async def test_01_ingest_new_knowledge_success(self):
        """
        Test (Golden Path): Verifies the full, successful pipeline for ingesting
        new, unstructured text.
        """
        # --- Arrange ---
        # Configure the mock mutator to return placement and refinement data in order
        self.mock_agent.synthesis_engine.generate = AsyncMock(side_effect=[
            MOCK_LLM_RESPONSES["placement"],  # First call for placement
            MOCK_LLM_RESPONSES["refinement"]  # Second call for refinement
        ])

        # --- Act ---
        result_path = await self.refiner.ingest_new_knowledge(RAW_KNOWLEDGE_TEXT)

        # --- Assert ---
        # 1. Verify the final file path is correct
        expected_path = self.mock_yaml_brain_path / "2_cognitive_architecture" / "2.2_architectural_principles" / "Axiom_2.2.2_principle_of_abstraction.yaml"
        self.assertEqual(result_path, expected_path)

        # 2. Verify the file exists and its content is the refined YAML
        self.assertTrue(expected_path.exists())
        with open(expected_path, 'r') as f:
            content = yaml.safe_load(f)
        self.assertEqual(content['id'], 'arch-abs-001')
        self.assertIn("Interface-Implementation Decoupling", str(content))

        # 3. Verify the mutator was called twice with the correct modes
        self.assertEqual(self.mock_agent.synthesis_engine.generate.call_count, 2)
        # Check call arguments for mode
        self.assertEqual(self.mock_agent.synthesis_engine.generate.call_args_list[0].kwargs['output_mode'], 'json')
        self.assertEqual(self.mock_agent.synthesis_engine.generate.call_args_list[1].kwargs['output_mode'], 'raw')

    async def test_02_ingest_new_knowledge_placement_fails(self):
        """
        Test (Failure Path): Ensures that if the placement step fails,
        no file is created and the function returns None.
        """
        # --- Arrange ---
        # Simulate a failure during the first LLM call (placement)
        self.mock_agent.synthesis_engine.generate.side_effect = SynthesisError("LLM failed to determine placement.")

        # --- Act ---
        result_path = await self.refiner.ingest_new_knowledge(RAW_KNOWLEDGE_TEXT)

        # --- Assert ---
        self.assertIsNone(result_path)
        # Check that no new files were created in the YAML_Brain
        file_count = len(list(self.mock_yaml_brain_path.rglob("*.yaml")))
        self.assertEqual(file_count, 1, "No new file should be created if placement fails.")
        self.mock_agent.synthesis_engine.generate.assert_called_once()

    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_03_continuous_refinement_loop(self, mock_sleep):
        """
        Test (Background Task): Verifies the continuous refinement loop can
        identify and process a low-quality "bootstrapped" file.
        """
        # --- Arrange ---
        # Create a low-quality, bootstrapped file that the loop should identify
        bootstrapped_content = {
            "id": "bootstrapped-idea",
            "type": "knowledge_document_bootstrapped",
            "content": "A simple idea about recursion."
        }
        bootstrapped_file_path = self.mock_yaml_brain_path / "bootstrapped.yaml"
        with open(bootstrapped_file_path, 'w') as f:
            yaml.dump(bootstrapped_content, f)

        # Mock the LLM to return a refined version when called
        self.mock_agent.synthesis_engine.generate.return_value = "# Refined YAML\nid: refined-recursion\n..."

        # --- Act ---
        # We need to run the loop for a short time and then stop it
        self.refiner.is_running = True

        async def stop_loop_after_one_cycle():
            await asyncio.sleep(0.1)  # Let the loop run once
            self.refiner.stop_loop()

        # Run the loop and the stop command concurrently
        await asyncio.gather(
            self.refiner.start_continuous_refinement_loop(interval_seconds=0.01),
            stop_loop_after_one_cycle()
        )

        # --- Assert ---
        # 1. The sleep mock should have been called, indicating the loop ran.
        mock_sleep.assert_called()

        # 2. The LLM should have been called to refine the file.
        self.mock_agent.synthesis_engine.generate.assert_called_once()

        # 3. The content of the file should now be the refined version.
        final_content = bootstrapped_file_path.read_text()
        self.assertIn("# Refined YAML", final_content)


if __name__ == '__main__':
    unittest.main()
