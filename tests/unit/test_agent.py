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
    deserialization processes are robust and correct, especially with Pydantic v2.
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
        self.assertIn("agent-", agent.metadata.version_id)

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
        # --- THE FIX: Use a less brittle regex that captures the essence of the error ---
        with self.assertRaisesRegex(ValidationError, "Agent 'code' field cannot be empty"):
            Agent.model_validate(invalid_data_empty)

        with self.assertRaisesRegex(ValidationError, "Agent 'code' field cannot be empty"):
            Agent.model_validate(invalid_data_whitespace)

    def test_04_field_validator_rejects_mismatched_vector_length(self):
        """
        Test (Validation): Verifies that the GeometricState validator rejects
        a position_vector whose length does not match the dimensions.
        """
        # --- Arrange ---
        # This data is invalid because the vector has 3 elements but should have 5.
        # This will trigger the validator within the nested GeometricState model.
        invalid_geometric_state_data = {
            "dimensions": ["Performance", "Clarity", "Brevity", "Safety", "Novelty"],
            "position_vector": [1.0, 2.0, 3.0] # Mismatched length
        }

        # --- Act & Assert ---
        # --- THE FIX: Test the nested model directly and use a more general regex ---
        # Pydantic v2's error messages are more structured. We check for the core message.
        with self.assertRaisesRegex(ValidationError, "Length of position_vector must match"):
            GeometricState.model_validate(invalid_geometric_state_data)

    def test_05_serialization_and_deserialization_integrity(self):
        """
        Test (Data Integrity): Ensures an Agent can be serialized to a dictionary
        and then perfectly deserialized back into an identical object.
        """
        # --- Arrange ---
        original_agent = Agent.model_validate(self.agent_data)

        # --- Act ---
        # Use Pydantic's recommended serialization methods for v2
        archive_dict = original_agent.model_dump(mode='json')
        archive_json_str = original_agent.model_dump_json()

        # Deserialize from both the dict and the JSON string to be thorough
        rehydrated_from_dict = Agent.model_validate(archive_dict)
        rehydrated_from_json = Agent.model_validate_json(archive_json_str)

        # --- Assert ---
        self.assertEqual(original_agent, rehydrated_from_dict, "Rehydrated agent from dict must be identical.")
        self.assertEqual(original_agent, rehydrated_from_json, "Rehydrated agent from json must be identical.")
        self.assertEqual(original_agent.metadata.version_id, rehydrated_from_json.metadata.version_id)
        self.assertEqual(original_agent.metadata.final_score, rehydrated_from_json.metadata.final_score)
        self.assertEqual(original_agent.code, rehydrated_from_json.code)

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