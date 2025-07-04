# prometheus_agent/tests/unit/test_geometric_transformer.py

import unittest
import numpy as np

from prometheus_agent.Geometry.GeometricTransformer import GeometricTransformer


# This structure assumes the test is run from the project root directory



class TestGeometricTransformer(unittest.TestCase):
    """
    A comprehensive unit test suite for the GeometricTransformer.

    This suite validates the transformer's ability to correctly map a given
    cognitive state vector to the closest persona archetype using Euclidean
    distance, and to handle invalid inputs gracefully.
    """

    def setUp(self):
        """Initialize the GeometricTransformer for each test."""
        self.transformer = GeometricTransformer()
        # Extract the archetypes for direct access in tests
        self.archetypes = self.transformer._PERSONA_ARCHETYPES

    def test_01_exact_match_guardian(self):
        """
        Test (Exact Match): Verifies that a vector identical to the Guardian's
        archetype vector correctly maps to the Guardian persona.
        """
        # --- Arrange ---
        guardian_archetype = next(a for a in self.archetypes if a['name'] == 'The Guardian')
        state_vector = guardian_archetype['vector']

        # --- Act ---
        archetype_name, directive = self.transformer.get_directives_from_state_vector(state_vector)

        # --- Assert ---
        self.assertEqual(archetype_name, 'The Guardian')
        self.assertEqual(directive, guardian_archetype['directive'])
        self.assertEqual(directive.facial_state.primary_emotion, 'concerned')

    def test_02_exact_match_visionary(self):
        """
        Test (Exact Match): Verifies that a vector identical to the Visionary's
        archetype vector correctly maps to the Visionary persona.
        """
        # --- Arrange ---
        visionary_archetype = next(a for a in self.archetypes if a['name'] == 'The Visionary')
        state_vector = visionary_archetype['vector']

        # --- Act ---
        archetype_name, directive = self.transformer.get_directives_from_state_vector(state_vector)

        # --- Assert ---
        self.assertEqual(archetype_name, 'The Visionary')
        self.assertEqual(directive, visionary_archetype['directive'])
        self.assertEqual(directive.vocal_profile.speed, 'energetic')

    def test_03_noisy_vector_maps_to_closest_archetype(self):
        """
        Test (Core Logic): Verifies that a "noisy" vector, not identical to any
        archetype, correctly maps to the NEAREST one (The Analyst).
        """
        # --- Arrange ---
        # The Analyst's vector is [9.0, 9.0, 8.0, 7.0, 4.0]
        # This noisy vector is clearly closest to The Analyst.
        noisy_analyst_vector = np.array([8.9, 9.1, 7.8, 7.0, 4.2])

        # --- Act ---
        archetype_name, directive = self.transformer.get_directives_from_state_vector(noisy_analyst_vector)

        # --- Assert ---
        self.assertEqual(archetype_name, 'The Analyst')
        self.assertEqual(directive.facial_state.primary_emotion, 'focused')

    def test_04_neutral_vector_maps_to_default(self):
        """
        Test (Default Fallback): Verifies that a neutral vector, equidistant from
        several archetypes, maps to the designated default (The Synthesist).
        """
        # --- Arrange ---
        # The Synthesist's vector is [7.0, 7.0, 7.0, 7.0, 5.0]
        neutral_vector = np.array([7.1, 6.9, 7.0, 7.1, 5.1])

        # --- Act ---
        archetype_name, directive = self.transformer.get_directives_from_state_vector(neutral_vector)

        # --- Assert ---
        self.assertEqual(archetype_name, 'The Synthesist (Default)')
        self.assertEqual(directive.facial_state.primary_emotion, 'neutral')

    def test_05_invalid_input_falls_back_gracefully(self):
        """
        Test (Robustness): Ensures that invalid input (e.g., wrong type, wrong shape)
        does not crash the transformer and instead returns the default persona.
        """
        # --- Arrange ---
        default_archetype = next(a for a in self.archetypes if 'Default' in a['name'])
        invalid_inputs = [
            None,
            [1, 2, 3, 4, 5],  # Standard list, not a numpy array
            np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),  # 2D array
            np.array([1, 2, 3])  # Wrong dimension
        ]

        for invalid_input in invalid_inputs:
            with self.subTest(input_type=type(invalid_input)):
                # --- Act ---
                archetype_name, directive = self.transformer.get_directives_from_state_vector(invalid_input)

                # --- Assert ---
                self.assertEqual(archetype_name, default_archetype['name'])
                self.assertEqual(directive, default_archetype['directive'])

    def test_06_alias_method_for_backwards_compatibility(self):
        """
        Test (Compatibility): Verifies that the alias method `get_directives_from_mutation_vector`
        produces the same result as the primary method.
        """
        # --- Arrange ---
        state_vector = np.array([9.0, 9.0, 8.0, 7.0, 4.0])  # The Analyst vector

        # --- Act ---
        name1, directive1 = self.transformer.get_directives_from_state_vector(state_vector)
        name2, directive2 = self.transformer.get_directives_from_mutation_vector(state_vector)

        # --- Assert ---
        self.assertEqual(name1, name2)
        self.assertEqual(directive1, directive2)
        self.assertEqual(name2, 'The Analyst')


if __name__ == '__main__':
    unittest.main(verbosity=2)
