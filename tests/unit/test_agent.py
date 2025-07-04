# prometheus_agent/tests/unit/test_agent.py

import unittest
import json
from pydantic import ValidationError

from prometheus_agent.Agent import Agent, AgentMetadata, GeometricState, QuantumState


# This structure assumes the test is run from the project root directory



class TestAgentModel(unittest.TestCase):
    """
    A comprehensive unit test suite for the Agent Pydantic model.

    This suite validates the core data structure of the Prometheus system, ensuring
    its validation logic, automatic data linking (validators), and serialization/
    deserialization processes are robust and correct.
    """

    def setUp(self):
        """Prepares a sample, valid evaluation report for use in tests."""
        self.evaluation_report = {
            "final_score": 9.12,
            "geometric_state_vector": [9.5, 8.8, 9.0, 10.0, 8.3],
            "dimensions": ["Performance", "Clarity", "Brevity", "Safety", "Novelty"],
            "breakdown": {"Performance": 9.5, "Clarity": 8.8, "Brevity": 9.0, "Safety": 10.0, "Novelty": 8.3},
            "execution_details": {"status": "success", "execution_time": 0.015}
        }
        self.agent_data = {
            "metadata": {
                "parent_id": "agent-parent-123",
                "origin_prompt": "Test prompt",
                "reasoning_path": ["tool1", "tool2"],
                "evaluations": [self.evaluation_report]
            },
            "code": "def solve():\n    return 42"
        }

    def test_01_successful_creation_with_valid_data(self):
        """
        Test (Golden Path): Verifies that an Agent can be created successfully
        with complete and valid data.
        """
        # --- Act ---
        try:
            agent = Agent.model_validate(self.agent_data)
        except ValidationError as e:
            self.fail(f"Agent creation failed with valid data: {e}")

        # --- Assert ---
        self.assertIsInstance(agent, Agent)
        self.assertIsInstance(agent.metadata, AgentMetadata)
        self.assertIsInstance(agent.metadata.geometric_state, GeometricState)
        self.assertIsInstance(agent.metadata.quantum_state, QuantumState)
        self.assertEqual(agent.code, self.agent_data["code"])
        self.assertIn("agent-", agent.metadata.version_id)  # Check for default UUID factory

    def test_02_model_validator_auto_populates_state(self):
        """
        Test (Core Logic): Ensures the `link_evaluation_to_state` validator
        correctly populates the top-level score and vector from the evaluations log.
        """
        # --- Act ---
        agent = Agent.model_validate(self.agent_data)

        # --- Assert ---
        self.assertEqual(agent.metadata.final_score, 9.12)
        self.assertEqual(agent.metadata.geometric_state.position_vector, [9.5, 8.8, 9.0, 10.0, 8.3])
        self.assertEqual(len(agent.metadata.geometric_state.position_vector),
                         len(agent.metadata.geometric_state.dimensions))

    def test_03_field_validator_rejects_empty_code(self):
        """
        Test (Validation): Verifies that the `code_must_not_be_empty` validator
        correctly raises a ValidationError for empty or whitespace-only code.
        """
        # --- Arrange ---
        invalid_data_empty = self.agent_data.copy()
        invalid_data_empty["code"] = ""

        invalid_data_whitespace = self.agent_data.copy()
        invalid_data_whitespace["code"] = "   \n\t   "

        # --- Act & Assert ---
        with self.assertRaisesRegex(ValidationError, "code_must_not_be_empty"):
            Agent.model_validate(invalid_data_empty)

        with self.assertRaisesRegex(ValidationError, "code_must_not_be_empty"):
            Agent.model_validate(invalid_data_whitespace)

    def test_04_field_validator_rejects_mismatched_vector_length(self):
        """

        Test (Validation): Verifies that the GeometricState validator rejects
        a position_vector whose length does not match the dimensions.
        """
        # --- Arrange ---
        invalid_data = self.agent_data.copy()
        # Create a report with a vector that is too short
        invalid_report = self.evaluation_report.copy()
        invalid_report["geometric_state_vector"] = [1.0, 2.0, 3.0]  # Should be 5
        invalid_data["metadata"]["evaluations"] = [invalid_report]

        # --- Act & Assert ---
        with self.assertRaisesRegex(ValidationError, "vector_matches_dimensions"):
            Agent.model_validate(invalid_data)

    def test_05_serialization_and_deserialization_integrity(self):
        """
        Test (Data Integrity): Ensures an Agent can be serialized to a dictionary
        and then perfectly deserialized back into an identical object.
        """
        # --- Arrange ---
        original_agent = Agent.model_validate(self.agent_data)

        # --- Act ---
        # 1. Serialize to a dictionary
        archive_dict = original_agent.to_archive_dict()

        # 2. (Optional) Simulate saving and loading as JSON
        archive_json = json.dumps(archive_dict)
        reloaded_dict = json.loads(archive_json)

        # 3. Deserialize back into an Agent object
        rehydrated_agent = Agent.from_archive_dict(reloaded_dict)

        # --- Assert ---
        self.assertEqual(original_agent, rehydrated_agent, "The rehydrated agent must be identical to the original.")
        self.assertEqual(original_agent.metadata.version_id, rehydrated_agent.metadata.version_id)
        self.assertEqual(original_agent.metadata.final_score, rehydrated_agent.metadata.final_score)
        self.assertEqual(original_agent.code, rehydrated_agent.code)

    def test_06_creation_without_evaluations_is_valid(self):
        """
        Test (Edge Case): Verifies that an agent can be created without any
        evaluation reports, resulting in default state values.
        """
        # --- Arrange ---
        data_no_evals = {
            "metadata": {"origin_prompt": "Initial state"},
            "code": "pass"
        }

        # --- Act ---
        agent = Agent.model_validate(data_no_evals)

        # --- Assert ---
        self.assertEqual(agent.metadata.final_score, 0.0)
        self.assertEqual(agent.metadata.geometric_state.position_vector, [0.0] * 5)
        self.assertEqual(agent.metadata.evaluations, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
