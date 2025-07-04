# prometheus_agent/tests/unit/test_persona_directives.py

import unittest
from pydantic import ValidationError

from prometheus_agent.Geometry.persona_directives import FacialState, VocalProfile, SomaticState, PersonaDirective


# This structure assumes the test is run from the project root directory




class TestPersonaDirectives(unittest.TestCase):
    """
    A comprehensive unit test suite for the Somatic Protocol Pydantic models.

    This suite validates that the data contracts for the agent's persona are
    robust, correctly typed, and enforce all defined validation rules, ensuring
    data integrity between the cognitive core and the GUI/avatar.
    """

    # --- FacialState Tests ---

    def test_01_facial_state_defaults(self):
        """Tests that FacialState instantiates with correct default values."""
        # --- Act ---
        state = FacialState()
        # --- Assert ---
        self.assertEqual(state.primary_emotion, 'neutral')
        self.assertEqual(state.intensity, 0.5)
        self.assertEqual(state.eye_movement, 'stable')
        self.assertEqual(state.microexpression_frequency, 0.2)

    def test_02_facial_state_custom_values(self):
        """Tests successful creation of FacialState with valid custom values."""
        # --- Arrange ---
        data = {
            "primary_emotion": "focused",
            "intensity": 0.8,
            "eye_movement": "direct_contact",
            "microexpression_frequency": 0.1
        }
        # --- Act ---
        state = FacialState.model_validate(data)
        # --- Assert ---
        self.assertEqual(state.primary_emotion, "focused")
        self.assertEqual(state.intensity, 0.8)

    def test_03_facial_state_validation_failure(self):
        """Tests that FacialState raises ValidationError for out-of-bounds or invalid enum values."""
        # Invalid primary_emotion
        with self.assertRaises(ValidationError):
            FacialState(primary_emotion="angry")

        # Intensity too high
        with self.assertRaises(ValidationError):
            FacialState(intensity=1.1)

        # Intensity too low
        with self.assertRaises(ValidationError):
            FacialState(intensity=-0.1)

        # Invalid eye_movement
        with self.assertRaises(ValidationError):
            FacialState(eye_movement="erratic")

    # --- VocalProfile Tests ---

    def test_04_vocal_profile_defaults(self):
        """Tests that VocalProfile instantiates with correct default values."""
        # --- Act ---
        profile = VocalProfile()
        # --- Assert ---
        self.assertEqual(profile.speed, 'normal')
        self.assertEqual(profile.pitch, 'neutral')
        self.assertEqual(profile.inflection_strength, 0.5)

    def test_05_vocal_profile_validation_failure(self):
        """Tests that VocalProfile raises ValidationError for invalid values."""
        # Invalid speed
        with self.assertRaises(ValidationError):
            VocalProfile(speed="slow")

        # Inflection strength out of bounds
        with self.assertRaises(ValidationError):
            VocalProfile(inflection_strength=2.0)

    # --- SomaticState Tests ---

    def test_06_somatic_state_defaults(self):
        """Tests that SomaticState instantiates with correct default values."""
        # --- Act ---
        state = SomaticState()
        # --- Assert ---
        self.assertEqual(state.breathing_rate, 1.0)
        self.assertEqual(state.gesture_rhythm, 'calm')

    def test_07_somatic_state_validation_failure(self):
        """Tests that SomaticState raises ValidationError for invalid values."""
        # Breathing rate too high
        with self.assertRaises(ValidationError):
            SomaticState(breathing_rate=2.1)

        # Invalid gesture_rhythm
        with self.assertRaises(ValidationError):
            SomaticState(gesture_rhythm="frantic")

    # --- PersonaDirective (Container) Tests ---

    def test_08_persona_directive_default_composition(self):
        """Tests that the top-level PersonaDirective correctly composes default sub-models."""
        # --- Act ---
        directive = PersonaDirective()
        # --- Assert ---
        self.assertIsInstance(directive.facial_state, FacialState)
        self.assertIsInstance(directive.vocal_profile, VocalProfile)
        self.assertIsInstance(directive.somatic_state, SomaticState)
        self.assertEqual(directive.facial_state.primary_emotion, 'neutral')
        self.assertEqual(directive.vocal_profile.speed, 'normal')

    def test_09_persona_directive_custom_composition(self):
        """Tests creation of a PersonaDirective with custom, nested models."""
        # --- Arrange ---
        custom_facial = FacialState(primary_emotion='inquisitive', intensity=0.9)
        custom_vocal = VocalProfile(speed='energetic', pitch='raised')

        # --- Act ---
        directive = PersonaDirective(
            facial_state=custom_facial,
            vocal_profile=custom_vocal
        )

        # --- Assert ---
        self.assertEqual(directive.facial_state.primary_emotion, 'inquisitive')
        self.assertEqual(directive.facial_state.intensity, 0.9)
        self.assertEqual(directive.vocal_profile.speed, 'energetic')
        # Check that unspecified models still get their defaults
        self.assertEqual(directive.somatic_state.breathing_rate, 1.0)

    def test_10_persona_directive_is_immutable(self):
        """
        Test (Core Feature): Verifies that the PersonaDirective is frozen and
        raises an error on attempted modification, ensuring its integrity.
        """
        # --- Arrange ---
        directive = PersonaDirective()

        # --- Act & Assert ---
        # Attempting to change an attribute on a frozen Pydantic model raises a TypeError
        with self.assertRaises(TypeError):
            directive.facial_state.intensity = 1.0

        with self.assertRaises(TypeError):
            directive.facial_state = FacialState(primary_emotion='concerned')


if __name__ == '__main__':
    unittest.main(verbosity=2)
