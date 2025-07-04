# prometheus_agent/tests/unit/test_cognitive_governor.py

import unittest
from unittest.mock import MagicMock

from prometheus_agent.CognitiveGovernor import CognitiveGovernor, ModelDecision


# This structure assumes the test is run from the project root directory



class TestCognitiveGovernor(unittest.TestCase):
    """
    A comprehensive unit test suite for the CognitiveGovernor.

    This suite validates the governor's hierarchical decision-making logic,
    ensuring it correctly selects the appropriate model based on a cascading
    set of rules: domain-specific skills, global overrides, domain defaults,
    and the final global default. All dependencies are mocked for fast,
    isolated, and deterministic testing.
    """

    def setUp(self):
        """Set up a mock agent instance with a well-defined config and skill_map."""
        self.mock_agent = MagicMock()

        # Define a mock configuration, mimicking config.json
        self.mock_agent.config = {
            "llm_models": {
                "provider": "local",
                "default_model": "phi3:latest",
                "power_model": "llama3:8b-instruct",
                "fast_model": "phi3:medium",
            },
            "skill_routing": {
                "evaluator_judge": "fast_model",
                "knowledge_transmutation": "power_model"
            }
        }

        # Define a mock skill map, mimicking the compiled YAML_Brain skills
        self.mock_agent.skill_map = {
            "coding": {
                "default_model": "power_model",
                "skills_by_name": {
                    "manifold_code_gen": {
                        "name": "manifold_code_gen",
                        "model": "power_model",  # Explicitly uses the powerful model
                        "description": "Generates a complete, runnable Python script."
                    }
                }
            },
            "metacognition": {
                "default_model": "default_model",  # This domain prefers the standard model
                "skills_by_name": {}  # No specific skill overrides in this domain
            }
        }

        # Instantiate the class under test
        self.governor = CognitiveGovernor(agent_instance=self.mock_agent)

    def test_01_initialization_and_alias_map_creation(self):
        """Tests successful initialization and correct creation of the model alias map."""
        self.assertIsNotNone(self.governor)
        self.assertEqual(self.governor.model_alias_map["power_model"], "llama3:8b-instruct")
        self.assertEqual(self.governor.model_alias_map["default_model"], "phi3:latest")

    def test_02_initialization_failure_with_incomplete_config(self):
        """Tests that initialization raises a ValueError if the config is missing required model aliases."""
        incomplete_config_agent = MagicMock()
        incomplete_config_agent.config = {"llm_models": {"default_model": "phi3"}}  # Missing power and fast models
        incomplete_config_agent.skill_map = {}

        with self.assertRaisesRegex(ValueError, "must define non-empty values for 'default_model', 'power_model', and 'fast_model'"):
            CognitiveGovernor(agent_instance=incomplete_config_agent)

    def test_03_precedence_1_domain_specific_skill_model(self):
        """Tests that a model defined for a specific skill in its domain is chosen first."""
        # --- Act ---
        decision = self.governor.decide_model(skill="manifold_code_gen", domain="coding")

        # --- Assert ---
        self.assertIsInstance(decision, ModelDecision)
        # Should resolve 'power_model' to its concrete name
        self.assertEqual(decision.model_name, "llama3:8b-instruct")
        self.assertIn("Domain-Specific Skill", decision.reason)

    def test_04_precedence_2_global_skill_override_model(self):
        """Tests that a global override from config.json is used if no domain-specific skill model exists."""
        # The 'knowledge_transmutation' skill is not in any skill_map, but is in the global skill_routing config

        # --- Act ---
        decision = self.governor.decide_model(skill="knowledge_transmutation", domain="some_other_domain")

        # --- Assert ---
        # Should resolve 'power_model' to its concrete name
        self.assertEqual(decision.model_name, "llama3:8b-instruct")
        self.assertIn("Global Skill Override", decision.reason)

    def test_05_precedence_3_domain_default_model(self):
        """Tests that a domain's default model is used if no specific or global rule applies."""
        # The 'knowledge_placement' skill is not in the metacognition skill_map or global routing.
        # Therefore, the 'metacognition' domain's default ('default_model') should be used.

        # --- Act ---
        decision = self.governor.decide_model(skill="knowledge_placement", domain="metacognition")

        # --- Assert ---
        # Should resolve the domain's 'default_model' to its concrete name
        self.assertEqual(decision.model_name, "phi3:latest")
        self.assertIn("Domain Default", decision.reason)

    def test_06_precedence_4_global_default_model(self):
        """Tests that the global default model is used as the final fallback."""
        # This skill and domain exist nowhere in the configuration.

        # --- Act ---
        decision = self.governor.decide_model(skill="completely_unknown_skill", domain="non_existent_domain")

        # --- Assert ---
        # Should resolve the global 'default_model' alias
        self.assertEqual(decision.model_name, "phi3:latest")
        self.assertIn("Global Default", decision.reason)

    def test_07_resolve_alias_logic(self):
        """Directly tests the internal alias resolution logic."""
        # --- Act & Assert ---
        # Test resolving a known alias
        self.assertEqual(self.governor._resolve_alias("fast_model"), "phi3:medium")

        # Test resolving a key that is not an alias (should return itself)
        self.assertEqual(self.governor._resolve_alias("gpt-4-turbo"), "gpt-4-turbo")

        # Test resolving a non-existent key (should return itself)
        self.assertEqual(self.governor._resolve_alias("non_existent_alias"), "non_existent_alias")


if __name__ == '__main__':
    unittest.main(verbosity=2)
