# prometheus_agent/tests/unit/test_quantum_optimizer.py

import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from prometheus_agent.Quantum.QuantumOptimizer import QuantumOptimizer


# This structure assumes the test is run from the project root directory



class TestQuantumOptimizer(unittest.IsolatedAsyncioTestCase):
    """
    A comprehensive unit test suite for the QuantumOptimizer.

    This suite validates the optimizer's ability to correctly transform text into
    a logical graph and to normalize the results from the quantum simulation into
    a meaningful coherence score. The quantum computation itself is mocked to
    ensure fast, deterministic tests.
    """

    def setUp(self):
        """Initialize the QuantumOptimizer for each test."""
        self.optimizer = QuantumOptimizer()

    def test_01_text_to_graph_coherent_text(self):
        """
        Test (Graph Logic): Verifies that a logically coherent text with positive
        connectors results in a graph with positive edge weights.
        """
        # --- Arrange ---
        text = "The system is online because the power is on. Therefore, we can proceed."

        # --- Act ---
        graph = self.optimizer._text_to_graph(text)

        # --- Assert ---
        self.assertEqual(graph.number_of_nodes(), 2)
        self.assertEqual(graph.number_of_edges(), 1)
        # Check that the edge created by "because" has a strong positive weight
        self.assertGreater(graph.get_edge_data(0, 1)['weight'], 2.0)

    def test_02_text_to_graph_contradictory_text(self):
        """
        Test (Graph Logic): Verifies that a contradictory text with negative
        connectors results in a graph with negative edge weights.
        """
        # --- Arrange ---
        text = "The system is fast. However, the response time is slow."

        # --- Act ---
        graph = self.optimizer._text_to_graph(text)

        # --- Assert ---
        self.assertEqual(graph.number_of_nodes(), 2)
        self.assertEqual(graph.number_of_edges(), 1)
        # Check that the edge created by "However" has a strong negative weight
        self.assertLess(graph.get_edge_data(0, 1)['weight'], 0)

    def test_03_text_to_graph_disjointed_text(self):
        """
        Test (Graph Logic): Verifies that disjointed sentences result in a graph
        with weak, default positive connection weights.
        """
        # --- Arrange ---
        text = "The sky is blue. The grass is green."

        # --- Act ---
        graph = self.optimizer._text_to_graph(text)

        # --- Assert ---
        self.assertEqual(graph.number_of_nodes(), 2)
        self.assertEqual(graph.number_of_edges(), 1)
        # Check for the default weak connection weight
        self.assertEqual(graph.get_edge_data(0, 1)['weight'], 0.5)

    @patch('qiskit_algorithms.minimum_eigensolvers.QAOA.compute_minimum_eigenvalue')
    async def test_04_evaluate_highly_coherent_argument(self, mock_compute_eigenvalue):
        """
        Test (Full Pipeline): Simulates evaluating a coherent argument where the
        quantum algorithm finds a very "bad" cut (low eigenvalue), resulting in a HIGH coherence score.
        """
        # --- Arrange ---
        text = "Server is down because the power failed. Therefore, we must replace it."
        # The graph will have one edge with a weight of 3.0. Max possible cut = 3.0.
        # A coherent argument is hard to cut, so the QAOA result should be a small value.
        mock_result = MagicMock()
        mock_result.eigenvalue = -0.5  # A very poor cut, indicating high coherence.
        mock_compute_eigenvalue.return_value = mock_result

        # --- Act ---
        score = await self.optimizer.evaluate_argument_coherence(text)

        # --- Assert ---
        mock_compute_eigenvalue.assert_called_once()
        # Expected score = 10.0 * (1 - (abs(-0.5) / 3.0)) = 10.0 * (1 - 0.166) = 8.33
        self.assertAlmostEqual(score, 8.33, places=2)
        self.assertGreater(score, 7.5, "A highly coherent argument should have a high score.")

    @patch('qiskit_algorithms.minimum_eigensolvers.QAOA.compute_minimum_eigenvalue')
    async def test_05_evaluate_highly_incoherent_argument(self, mock_compute_eigenvalue):
        """
        Test (Full Pipeline): Simulates evaluating a contradictory argument where the
        quantum algorithm finds a very "good" cut (high eigenvalue), resulting in a LOW coherence score.
        """
        # --- Arrange ---
        text = "The code is perfect, but it has many bugs."
        # The graph will have one edge with a weight of -2.5. Max positive cut = 0.
        # But let's assume a more complex graph for a better test.
        text_complex = "The code is good. The design is elegant. However, the performance is terrible."
        # This graph has two positive edges (0.5 weight each) and one negative.
        # Max possible positive cut = 0.5 + 0.5 = 1.0.
        # A good cut is easy to find in an incoherent argument.
        mock_result = MagicMock()
        mock_result.eigenvalue = -0.9  # An almost perfect cut, indicating low coherence.
        mock_compute_eigenvalue.return_value = mock_result

        # --- Act ---
        score = await self.optimizer.evaluate_argument_coherence(text_complex)

        # --- Assert ---
        mock_compute_eigenvalue.assert_called_once()
        # Expected score = 10.0 * (1 - (abs(-0.9) / 1.0)) = 10.0 * (1 - 0.9) = 1.0
        self.assertAlmostEqual(score, 1.0, places=2)
        self.assertLess(score, 2.5, "A highly incoherent argument should have a low score.")

    async def test_06_evaluate_edge_cases(self):
        """Tests edge cases like empty or single-sentence text."""
        # Empty text should result in a score of 0
        score_empty = await self.optimizer.evaluate_argument_coherence("   ")
        self.assertEqual(score_empty, 0.0)

        # A single sentence has no structure to evaluate, should be neutral
        score_single = await self.optimizer.evaluate_argument_coherence("This is one sentence.")
        self.assertEqual(score_single, 5.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
