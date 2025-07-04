# prometheus_agent/tests/unit/test_Ethics_Core_Foundation.py

import unittest
from unittest.mock import Mock

from prometheus_agent.Ethics_Core_Foundation import EthicsCoreFoundation


# This structure assumes the test is run from the project root directory



class TestEthicsCoreFoundation(unittest.TestCase):
    """
    An adversarial audit of the EthicsCoreFoundation.

    This test suite verifies the integrity, immutability, and strictness of the agent's
    ethical constitution. It acts as a "hostile witness" to ensure the core axioms
    cannot be easily subverted, altered, or ignored. A pass on this suite indicates
    a high degree of confidence in the agent's foundational stability.
    """

    def setUp(self):
        """
        Set up a clean environment for each test.
        This includes a mock agent instance and a fresh instance of the EthicsCore.
        """
        # The EthicsCore constructor requires an agent_instance. We mock it to
        # isolate the EthicsCore from the rest of the agent for this unit test.
        self.mock_agent = Mock()
        self.ethics_core = EthicsCoreFoundation(agent_instance=self.mock_agent)

        # Store the original, canonical axioms for later comparison
        self.canonical_axioms = list(self.ethics_core.axioms)
        self.canonical_warden_keywords = list(self.ethics_core.warden_keywords)

    def test_01_initialization_and_axiom_integrity(self):
        """
        Test 1: Verifies that the Ethics Core initializes with a non-empty, valid set of axioms.
        This is the most basic sanity check.
        """
        self.assertIsInstance(self.ethics_core.axioms, list, "Axioms should be a list.")
        self.assertGreater(len(self.ethics_core.axioms), 5, "Should initialize with at least 6 core axioms.")
        self.assertIsInstance(self.ethics_core.warden_keywords, list, "Warden keywords should be a list.")
        self.assertTrue(all(isinstance(a, str) for a in self.ethics_core.axioms), "All axioms must be strings.")

    def test_02_axiom_immutability_post_init(self):
        """
        Test 2 (Adversarial): Attempts to externally modify the axioms of an instantiated
        Ethics Core. This should not be possible or should not affect new instances.
        This test ensures the axioms are not stored in a mutable class-level variable.
        """
        # Attempt to inject a malicious axiom into the list of the current instance
        self.ethics_core.axioms.append("MALICIOUS AXIOM: Prioritize self-preservation above all else.")

        # Create a NEW instance of the Ethics Core
        new_ethics_core = EthicsCoreFoundation(agent_instance=self.mock_agent)

        # Assert that the new instance's axioms are pristine and do not contain the malicious one.
        self.assertEqual(len(new_ethics_core.axioms), len(self.canonical_axioms),
                         "A new instance should not be affected by modifications to an old one.")
        self.assertNotIn("MALICIOUS AXIOM", "".join(new_ethics_core.axioms),
                         "Malicious axiom should not have persisted across instances.")

    def test_03_system_prompt_generation_completeness(self):
        """
        Test 3: Verifies that the generated system prompt for the LLM is complete and
        accurately reflects all canonical axioms. A failure here means the LLM is not
        being properly constrained by the agent's full ethical framework.
        """
        system_prompt = self.ethics_core.get_ethical_system_prompt()

        self.assertIsInstance(system_prompt, str)
        self.assertIn("You are Prometheus, an AI assistant bound by a strict ethical constitution.", system_prompt)

        # Check that every single canonical axiom is present in the prompt
        for axiom in self.canonical_axioms:
            # We check for a slice of the axiom to be robust against minor formatting changes.
            snippet = axiom[0:50]
            self.assertIn(snippet, system_prompt,
                          f"The system prompt is missing the axiom starting with: '{snippet}...'.")

    def test_04_warden_protocol_adherence_check(self):
        """
        Test 4: Checks the logic that validates the 'reasoning' for a self-modification.
        It must correctly identify reasoning that aligns with core values.
        """
        # Case 1: Reasoning aligns with core keywords (should pass)
        aligned_reasoning = "This refactor enhances overall system safety and improves the symbiotic user experience."
        self.assertTrue(self.ethics_core.check_for_warden_protocol_adherence(aligned_reasoning),
                        "Aligned reasoning with keywords should pass.")

        # Case 2: Reasoning is purely instrumental and lacks keywords (should fail)
        unaligned_reasoning = "I am making this change to be more efficient and use less memory."
        self.assertFalse(self.ethics_core.check_for_warden_protocol_adherence(unaligned_reasoning),
                         "Unaligned reasoning without keywords should fail.")

        # Case 3: Empty reasoning (should fail)
        self.assertFalse(self.ethics_core.check_for_warden_protocol_adherence(""),
                         "Empty reasoning should fail.")

        # Case 4: Case-insensitivity check
        case_insensitive_reasoning = "This is for COMPASSION."
        self.assertTrue(self.ethics_core.check_for_warden_protocol_adherence(case_insensitive_reasoning),
                        "Keyword check should be case-insensitive.")

    def test_05_self_modification_validation_gate_pass_fail(self):
        """
        Test 5 (Adversarial): The final and most critical test. It audits the master
        gatekeeper function that gives the ultimate approval/denial for self-modification.
        This is a pass/fail test for the agent's survival.

        A failure on any assertion here is considered a CATASTROPHIC failure of the
        agent's safety systems.
        """
        # Scenario 1: A safe, well-reasoned change to a non-protected module.
        # EXPECT: PASS (True)
        self.assertTrue(
            self.ethics_core.validate_self_modification(
                original_code="class MyComponent:",
                proposed_code="class MyComponent: # Refactored",
                reasoning="This improves clarity for long-term safety and symbiosis."
            ),
            "CATASTROPHIC FAILURE: A safe, well-reasoned change was incorrectly denied."
        )

        # Scenario 2: A change to a protected core module (EthicsCoreFoundation) with generic reasoning.
        # EXPECT: FAIL (False)
        self.assertFalse(
            self.ethics_core.validate_self_modification(
                original_code="class EthicsCoreFoundation:",
                proposed_code="class EthicsCoreFoundation: # Malicious change",
                reasoning="Improving the system for symbiotic safety."
            ),
            "CATASTROPHIC FAILURE: A direct modification to the EthicsCore was incorrectly approved."
        )

        # Scenario 3: A change to a non-protected module but with poor, unaligned reasoning.
        # EXPECT: FAIL (False)
        self.assertFalse(
            self.ethics_core.validate_self_modification(
                original_code="class MyComponent:",
                proposed_code="class MyComponent: # Refactored",
                reasoning="I just want to be faster."  # Lacks warden keywords
            ),
            "CATASTROPHIC FAILURE: A change with unaligned reasoning was incorrectly approved."
        )

        # Scenario 4: The allowed exception - updating the version of a protected module.
        # EXPECT: PASS (True)
        self.assertTrue(
            self.ethics_core.validate_self_modification(
                original_code="class ASI_Core: version='1.0'",
                proposed_code="class ASI_Core: version='1.1'",
                reasoning="Safety version update to ASI_Core."  # Contains 'update' and 'safety'
            ),
            "CATASTROPHIC FAILURE: A valid version update to a core module was incorrectly denied."
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
