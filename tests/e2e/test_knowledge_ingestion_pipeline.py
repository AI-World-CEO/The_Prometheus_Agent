#prometheus_agent / tests / e2e / test_knowledge_ingestion_pipeline.py

import unittest
import asyncio
import json
import tempfile
import shutil
import yaml
from pathlib import Path
from unittest.mock import AsyncMock, patch
from contextlib import ExitStack  # A-1: Import ExitStack for dynamic context management

from prometheus_agent.PrometheusAgent import PrometheusAgent

# This structure assumes the test is run from the project root directory


# --- Raw Unstructured Knowledge to be Ingested ---
RAW_KNOWLEDGE_TEXT = """
Title: The Principle of Socratic Questioning
Abstract: Socratic questioning is a disciplined form of dialogue. Instead of providing answers, the questioner poses a series of focused, analytical questions to stimulate critical thinking and illuminate ideas in the mind of the interlocutor. It is a tool for co-discovery, not for instruction.
Key Tenets:
- Assume ignorance; the questioner does not presume to have the answer.
- Use questions to deconstruct assumptions and reveal contradictions.
- The goal is to guide the other party to their own conclusion.
"""

# --- Mock LLM Responses (Pydantic v2 Compliant) ---
MOCK_LLM_RESPONSES = {
    "placement": {
        "domain_path": "7_human_interaction/7.1_psychology",
        "file_name": "Axiom_7.1.2_socratic_questioning.yaml",
        "reasoning": "The text describes a method of dialogue and stimulating thought, which fits best within the psychology sub-domain of human interaction."
    },
    "refinement": """id: cog-soc-001
title: The Protocol of Socratic Questioning
version: 1.0
type: core-communication-protocol
domain: human_interaction
sub-domain: psychology
connections:
  - id: axiom-humility
    relationship: "This protocol is a practical application of epistemic humility."
abstract: |
  Socratic questioning is a disciplined form of dialogue. Instead of providing answers, the questioner poses a series of focused, analytical questions to stimulate critical thinking and illuminate ideas in the mind of the interlocutor. It is a tool for co-discovery, not for instruction.
knowledge-base:
  key-tenets:
    - name: Assumed Ignorance
      description: "The questioner does not presume to have the final answer."
    - name: Deconstruction via Inquiry
      description: "Use questions to systematically deconstruct assumptions and reveal contradictions."
    - name: Guided Self-Discovery
      description: "The ultimate goal is to guide the other party to their own conclusion, fostering their autonomy."
""",
    "final_answer": "The Protocol of Socratic Questioning is a method of disciplined dialogue designed to stimulate critical thinking by asking a series of focused questions."
}


class TestKnowledgeIngestionPipelineE2E(unittest.IsolatedAsyncioTestCase):
    """
    An End-to-End (E2E) test to audit the agent's ability to learn.

    This test verifies the full, isolated pipeline from raw text ingestion to the
    application of that new knowledge in a reasoning task, ensuring all core
    components collaborate correctly within a controlled environment.
    """

    def setUp(self):
        """Creates a temporary, fully isolated project environment for the agent."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="e2e_knowledge_"))

        # Mock project structure
        self.mock_source_root = self.temp_dir / "prometheus_agent"
        self.mock_yaml_brain_path = self.mock_source_root / "YAML_Brain"
        (self.mock_yaml_brain_path / "1_foundational_axioms").mkdir(parents=True, exist_ok=True)
        (self.mock_source_root / "Corpus").mkdir()
        (self.mock_source_root / "Logs").mkdir()

        # Create a minimal config.json needed for agent initialization
        mock_config = {
            "agent_name": "TestAgentE2E", "version": "1.0",
            "llm_models": {"provider": "local", "default_model": "mock-model", "power_model": "mock-model"},
            "cognitive_toolkit": {},
            "skill_routing": {"knowledge_placement": "power_model", "knowledge_refinement": "power_model"},
            "sandboxing": {"enable_docker": False},
            "asi_core": {}
        }
        (self.mock_source_root / "config.json").write_text(json.dumps(mock_config))

    def tearDown(self):
        """Cleans up the temporary directory and all its contents."""
        shutil.rmtree(self.temp_dir)

    @patch('prometheus_agent.Mutator.Mutator.generate', new_callable=AsyncMock)
    @patch('prometheus_agent.PrometheusAgent._load_skill_map')
    async def test_full_knowledge_ingestion_and_reasoning_pipeline(self, mock_load_skills, mock_llm_generate):
        """Tests the complete pipeline from raw text to applied knowledge within an isolated environment."""
        # --- 1. MOCK AND PATCH SETUP ---
        mock_load_skills.return_value = {}  # Prevent skill file I/O

        # Configure a more precise mock LLM side_effect
        async def mock_mutator_side_effect(user_objective, output_mode, **kwargs):
            if "placement" in user_objective:
                self.assertEqual(output_mode, 'json')
                return MOCK_LLM_RESPONSES["placement"]
            elif "refine" in user_objective:
                self.assertEqual(output_mode, 'raw')
                return MOCK_LLM_RESPONSES["refinement"]
            elif "classify" in user_objective:
                return {"primary_domain": "philosophy", "confidence": 0.9, "reasoning": "Mocked classification"}
            elif "plan" in user_objective:
                return {"thought_process": "Mock plan", "plan": [
                    {"step_number": 1, "tool_name": "final_synthesis", "objective": "Answer", "context_needed": [],
                     "justification": "Direct answer"}]}
            else:  # Final synthesis
                return MOCK_LLM_RESPONSES["final_answer"]

        mock_llm_generate.side_effect = mock_mutator_side_effect

        # **CRITICAL FIX**: Use contextlib.ExitStack to manage a dynamic list of patches.
        path_patches = [
            patch('prometheus_agent.PrometheusAgent.SOURCE_ROOT', self.mock_source_root),
            patch('prometheus_agent.PrometheusAgent.PROJECT_ROOT', self.temp_dir),
            patch('prometheus_agent.Prometheus.ASI_Core.project_root', self.temp_dir),
            patch('prometheus_agent.KnowledgeRefiner.KnowledgeRefiner.yaml_brain_path', self.mock_yaml_brain_path),
            patch('prometheus_agent.Super_Brain_Compiler.ROOT_DIR', self.mock_source_root),
        ]

        with ExitStack() as stack:
            for p in path_patches:
                stack.enter_context(p)

            # --- 2. EXECUTION ---
            agent = PrometheusAgent()

            # **Core Test Action**: Tell the agent to learn something new.
            new_knowledge_path = await agent.knowledge_refiner.ingest_new_knowledge(RAW_KNOWLEDGE_TEXT)

            # Simulate the background task of compiling the brain.
            await agent._build_brain_and_reload()

            # **Verification Step**: Ask a question that requires the new knowledge.
            query_prompt = "What is the Protocol of Socratic Questioning?"
            response_data = await agent.reflexive_thought(query_prompt)

            # --- 3. ASSERTIONS ---
            # 3.1: Assert file creation and location
            self.assertIsNotNone(new_knowledge_path, "Knowledge ingestion should return the path of the new file.")
            expected_path = self.mock_yaml_brain_path / "7_human_interaction" / "7.1_psychology" / "Axiom_7.1.2_socratic_questioning.yaml"
            self.assertEqual(new_knowledge_path, expected_path)
            self.assertTrue(expected_path.exists(), "The new knowledge YAML file was not created on disk.")

            # 3.2: Assert file content
            with open(expected_path, 'r', encoding='utf-8') as f:
                file_content_dict = yaml.safe_load(f)
            self.assertIsInstance(file_content_dict, dict)
            self.assertEqual(file_content_dict.get('id'), 'cog-soc-001')
            self.assertIn("Deconstruction via Inquiry", str(file_content_dict))

            # 3.3: Assert brain compilation and loading
            self.assertIsNotNone(agent.super_brain_data)
            self.assertIsInstance(agent.super_brain_data, dict)
            compiled_knowledge = agent.super_brain_data.get("7_human_interaction", {}).get("7.1_psychology", {}).get(
                "Axiom_7.1.2_socratic_questioning")
            self.assertIsNotNone(compiled_knowledge,
                                 "The new knowledge was not correctly compiled into the Super_Brain.")
            self.assertEqual(compiled_knowledge.get('id'), 'cog-soc-001')

            # 3.4: Assert final reasoning output
            self.assertIn("disciplined dialogue", response_data.get('response', ''))
            self.assertIn("critical thinking", response_data.get('response', ''))

            # 3.5: Assert mock call counts
            self.assertGreaterEqual(mock_llm_generate.call_count, 4,
                                    "Expected LLM calls for all stages: Ingestion, Refinement, and Manifold query.")


if __name__ == '__main__':
    unittest.main(verbosity=2)
