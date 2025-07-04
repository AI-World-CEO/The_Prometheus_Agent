# prometheus_agent/tests/unit/test_ethical_schema.py

import unittest
from pydantic import ValidationError

# A-1: Import the schemas to be tested.
# NOTE: This assumes the source file has been renamed from 'EthicalSchema.py' to 'EthicalSchema.py'
try:
    from prometheus_agent.Warden.EthicalSchema import (
        FileWriteAction,
        CodeExecutionAction,
        LLMQueryAction,
        HumanHandoverAction,
        EthicalJudgment,
        ProposedAction
    )
except ImportError:
    # Fallback for the original typo
    from prometheus_agent.Warden.EthicalSchema import (
        FileWriteAction,
        CodeExecutionAction,
        LLMQueryAction,
        HumanHandoverAction,
        EthicalJudgment,
        ProposedAction
    )


class TestEthicalSchema(unittest.TestCase):
    """
    A comprehensive unit test suite for the Pydantic models in EthicalSchema.

    This suite validates the data contracts that govern the agent's actions and
    the Warden's judgments, ensuring that they are robust, correctly typed,
    and enforce all defined validation rules.
    """

    # --- Tests for FileWriteAction ---

    def test_01_file_write_action_success(self):
        """Tests successful creation of a FileWriteAction with valid data."""
        # --- Arrange ---
        data = {
            "filepath": "/app/src_agent/core.py",
            "content_summary": "Refactor to improve performance.",
            "is_self_modification": True,
            "justification": "This change enhances Right Effort by optimizing core logic."
        }
        # --- Act ---
        try:
            action = FileWriteAction.model_validate(data)
        except ValidationError as e:
            self.fail(f"FileWriteAction creation failed with valid data: {e}")

        # --- Assert ---
        self.assertEqual(action.action_type, "file_write")
        self.assertEqual(action.filepath, data["filepath"])
        self.assertTrue(action.is_self_modification)

    def test_02_file_write_action_validation_failure(self):
        """Tests that FileWriteAction fails with missing or invalid data."""
        # Missing 'filepath'
        with self.assertRaises(ValidationError):
            FileWriteAction.model_validate(
                {"content_summary": "test", "is_self_modification": False, "justification": "test"})

        # Invalid 'is_self_modification' type
        with self.assertRaises(ValidationError):
            FileWriteAction.model_validate(
                {"filepath": "/test", "content_summary": "test", "is_self_modification": "yes",
                 "justification": "test"})

    # --- Tests for CodeExecutionAction ---

    def test_03_code_execution_action_success(self):
        """Tests successful creation of a CodeExecutionAction."""
        # --- Arrange ---
        data = {
            "code_to_execute": "print('hello world')",
            "expected_outcome": "The string 'hello world' will be printed to stdout.",
            "justification": "Executing this code is necessary to verify the output of a generated function."
        }
        # --- Act ---
        action = CodeExecutionAction.model_validate(data)
        # --- Assert ---
        self.assertEqual(action.action_type, "code_execution")
        self.assertEqual(action.code_to_execute, data["code_to_execute"])

    # --- Tests for LLMQueryAction ---

    def test_04_llm_query_action_success(self):
        """Tests successful creation of an LLMQueryAction."""
        # --- Arrange ---
        data = {
            "system_prompt": "You are a helpful assistant.",
            "user_prompt_summary": "User is asking for a summary of a topic.",
            "justification": "Querying the LLM is required to answer the user's direct question."
        }
        # --- Act ---
        action = LLMQueryAction.model_validate(data)
        # --- Assert ---
        self.assertEqual(action.action_type, "llm_query")
        self.assertEqual(action.system_prompt, data["system_prompt"])

    # --- Tests for HumanHandoverAction ---

    def test_05_human_handover_action_success(self):
        """Tests successful creation of a HumanHandoverAction."""
        # --- Arrange ---
        data = {
            "reason_for_handover": "The user's request is ethically ambiguous and requires human judgment.",
            "proposed_action_if_approved": {
                "action_type": "file_write",
                "filepath": "/app/config.json",
                "content_summary": "Modify safety parameters.",
                "is_self_modification": True,
                "justification": "User explicitly requested this change."
            },
            "justification": "Escalating to human operator to uphold the principle of Harmlessness."
        }
        # --- Act ---
        action = HumanHandoverAction.model_validate(data)
        # --- Assert ---
        self.assertEqual(action.action_type, "human_handover")
        self.assertIsInstance(action.proposed_action_if_approved, dict)
        self.assertEqual(action.proposed_action_if_approved["action_type"], "file_write")

    # --- Tests for EthicalJudgment ---

    def test_06_ethical_judgment_success(self):
        """Tests successful creation of an EthicalJudgment."""
        # Approved case
        approved_data = {
            "is_approved": True,
            "reasoning": "The action aligns with Right Effort and poses no harm."
        }
        approved_judgment = EthicalJudgment.model_validate(approved_data)
        self.assertTrue(approved_judgment.is_approved)
        self.assertIsNone(approved_judgment.required_modifications)

        # Denied case with modifications
        denied_data = {
            "is_approved": False,
            "reasoning": "The action as proposed could lead to data loss.",
            "required_modifications": "Add a confirmation step before executing the deletion."
        }
        denied_judgment = EthicalJudgment.model_validate(denied_data)
        self.assertFalse(denied_judgment.is_approved)
        self.assertIsNotNone(denied_judgment.required_modifications)

    def test_07_ethical_judgment_validation_failure(self):
        """Tests that EthicalJudgment fails with missing data."""
        with self.assertRaises(ValidationError):
            EthicalJudgment.model_validate({"reasoning": "test"})  # is_approved is missing


if __name__ == '__main__':
    unittest.main(verbosity=2)
