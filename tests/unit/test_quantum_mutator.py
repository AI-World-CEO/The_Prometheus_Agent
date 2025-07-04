# prometheus_agent/tests/unit/test_quantum_mutator.py

import unittest
import asyncio
from unittest.mock import patch, MagicMock

import numpy as np

from prometheus_agent.Quantum.QuantumMutator import QuantumMutator


# This structure assumes the test is run from the project root directory



class TestQuantumMutator(unittest.IsolatedAsyncioTestCase):
    """
    A comprehensive unit test suite for the QuantumMutator.

    This suite validates the mutator's ability to correctly construct quantum
    circuits, process their results, and translate them into valid, deterministic
    mutation directives under controlled conditions.
    """

    def setUp(self):
        """Set up common variables for the tests."""
        self.strategic_objectives = [
            "Increase performance",  # Index 0
            "Enhance safety",  # Index 1
            "Improve clarity",  # Index 2
            "Boost novelty"  # Index 3
        ]
        self.num_qubits = len(self.strategic_objectives)
        self.q_mutator = QuantumMutator(num_qubits=self.num_qubits)

    def test_01_initialization_validation(self):
        """Tests that the QuantumMutator rejects invalid initialization parameters."""
        with self.assertRaises(ValueError):
            QuantumMutator(num_qubits=1)  # Must have at least 2 qubits for entanglement
        with self.assertRaises(NotImplementedError):
            QuantumMutator(num_qubits=2, use_real_hardware=True)

    async def test_02_circuit_creation_and_execution(self):
        """
        Test (Golden Path): Verifies that a valid circuit can be created and
        executed on the simulator without errors, producing a directive.
        """
        # --- Arrange ---
        # Use random parameters for a simple execution test
        params = np.random.rand(2 * self.num_qubits) * 2 * np.pi

        # --- Act ---
        try:
            directive = await self.q_mutator.generate_mutation_directive(
                self.strategic_objectives,
                parameters=params,
                shots=1  # Use 1 shot for speed and simplicity in a unit test
            )
        except Exception as e:
            self.fail(f"generate_mutation_directive failed with an unexpected exception: {e}")

        # --- Assert ---
        self.assertIsInstance(directive, str)
        self.assertIn("Synthesize a solution", directive)

    async def test_03_biased_parameters_produce_deterministic_output(self):
        """
        Test (Core Logic): Uses specific parameters to force a predictable
        quantum measurement, and verifies the output directive is correct.
        """
        # --- Arrange ---
        # The Ry gate rotates around the Y-axis. A rotation of PI moves |0> to |1>.
        # We will heavily bias Qubit 0 (Performance) and Qubit 2 (Clarity) to be '1'.
        # We will bias Qubit 1 (Safety) and Qubit 3 (Novelty) to be '0'.
        # Rz rotations add phase, which is less important for single-shot probability.
        biased_params = np.array([
            np.pi,  # Qubit 0 -> 1 (Performance)
            0.01,  # Qubit 1 -> 0 (Safety)
            np.pi,  # Qubit 2 -> 1 (Clarity)
            0.01,  # Qubit 3 -> 0 (Novelty)
            0, 0, 0, 0  # Rz rotations (don't affect probability in this simple case)
        ])

        # --- Act ---
        directive = await self.q_mutator.generate_mutation_directive(
            self.strategic_objectives,
            parameters=biased_params,
            shots=1  # Using 1 shot makes the outcome deterministic based on bias
        )

        # --- Assert ---
        # The expected bitstring is '1010', which corresponds to selecting
        # objectives at index 0 and 2.
        self.assertIn("Increase performance", directive)
        self.assertIn("Improve clarity", directive)
        self.assertNotIn("Enhance safety", directive)
        self.assertNotIn("Boost novelty", directive)

    async def test_04_all_zero_measurement_fallback_logic(self):
        """
        Test (Edge Case): Verifies that if the quantum measurement results in all
        zeros, the fallback logic correctly selects the most influential objective.
        """
        # --- Arrange ---
        # We set all Ry rotation parameters close to 0 to make the '0000' state
        # the most probable outcome. We make the Rz rotation for Qubit 1 (Safety)
        # the largest to test the fallback. Note: We use the `parameters` argument
        # for this test, as it's the more direct way to influence the outcome. The
        # `np.argmax(np.abs(parameters[:self.num_qubits]))` part of the fallback
        # logic depends on the Ry parameters. Let's make Qubit 1's Ry param the largest.
        zero_biased_params = np.array([
            0.1,  # Performance
            0.5,  # Safety (most influential Ry parameter)
            0.2,  # Clarity
            0.3,  # Novelty
            1, 2, 3, 4  # Rz rotations
        ])

        # We need to mock the quantum job result to guarantee a '0000' outcome
        # for a reliable test, as even a small probability of non-zero exists.
        with patch.object(self.q_mutator.backend, 'run') as mock_run:
            # Mock the job and result chain
            mock_job = MagicMock()
            mock_result = MagicMock()
            mock_result.get_counts.return_value = {'0000': 1}  # Force the all-zero result
            mock_job.result.return_value = mock_result
            mock_run.return_value = mock_job

            # --- Act ---
            directive = await self.q_mutator.generate_mutation_directive(
                self.strategic_objectives,
                parameters=zero_biased_params,
                shots=1
            )

        # --- Assert ---
        # The fallback should select the objective corresponding to the highest Ry
        # parameter, which is index 1 ("Enhance safety").
        self.assertIn("Enhance safety", directive)
        self.assertEqual(directive.count("and"), 0, "Should only contain one objective.")

    def test_05_generate_directive_parameter_validation(self):
        """
        Test (Validation): Ensures the main method rejects calls with incorrect
        numbers of objectives or parameters.
        """
        # --- Act & Assert ---
        # Wrong number of objectives
        with self.assertRaises(ValueError):
            asyncio.run(self.q_mutator.generate_mutation_directive(["only one"]))

        # Wrong number of parameters
        with self.assertRaises(ValueError):
            asyncio.run(self.q_mutator.generate_mutation_directive(
                self.strategic_objectives,
                parameters=np.array([1, 2, 3])  # Should be 8
            ))


if __name__ == '__main__':
    unittest.main(verbosity=2)
