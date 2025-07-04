# tests/unit/test_visualizer.py

import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

# --- Correct, Absolute Imports from the Main Source Package ---
# These paths are now correct for the new project structure.
from prometheus_agent.Agent import Agent
from prometheus_agent.ArchivesManager import ArchiveManager
from prometheus_agent.Visualizer import Visualizer


# Helper to create a mock Pydantic v2-compliant Agent for testing
def create_mock_agent(score: float, vector: list, agent_id: str, parent_id: str = None) -> Agent:
    """Helper function to create a validated Agent object with specific data."""
    eval_report = {"final_score": score, "geometric_state_vector": vector}
    agent_data = {
        "metadata": {
            "version_id": agent_id,
            "parent_id": parent_id,
            "origin_prompt": "Test Agent",
            "evaluations": [eval_report],
        },
        "code": "pass"
    }
    return Agent.model_validate(agent_data)


class TestVisualizer(unittest.IsolatedAsyncioTestCase):
    """
    A comprehensive unit test suite for the Visualizer component.

    This suite validates the data processing and plotting logic of the Visualizer
    in isolation, using a mocked ArchiveManager and patched plotting libraries
    to ensure fast, reliable, and deterministic testing.
    """

    def setUp(self):
        """Set up a mocked ArchiveManager and instantiate the Visualizer."""
        self.mock_archive_manager = MagicMock(spec=ArchiveManager)
        # Pass the mocked manager to the Visualizer
        self.visualizer = Visualizer(archive_manager=self.mock_archive_manager)

        # Create a sample population of agents for the tests
        self.agent1 = create_mock_agent(score=5.0, vector=[5, 5, 5, 5, 5], agent_id="agent-001")
        self.agent2 = create_mock_agent(score=7.5, vector=[7, 8, 7, 8, 7], agent_id="agent-002", parent_id="agent-001")
        self.agent3 = create_mock_agent(score=9.0, vector=[9, 8, 9, 9, 9], agent_id="agent-003", parent_id="agent-001")
        self.test_population = [self.agent1, self.agent2, self.agent3]

    @patch('webbrowser.open')
    @patch('plotly.graph_objects.Figure.write_html')
    async def test_01_visualize_lineage_success(self, mock_write_html, mock_webbrowser):
        """
        Test (Golden Path): Verifies that visualize_lineage correctly processes
        agents, builds a graph, and calls the plotting function with valid data.
        """
        # --- Arrange ---
        self.mock_archive_manager.query_agents = AsyncMock(return_value=self.test_population)

        # --- Act ---
        await self.visualizer.visualize_lineage()

        # --- Assert ---
        self.mock_archive_manager.query_agents.assert_called_once()
        mock_write_html.assert_called_once()
        mock_webbrowser.assert_called_once()

        fig = mock_write_html.call_args[0][0]
        self.assertEqual(len(fig.data), 2, "Figure should have one edge trace and one node trace.")
        node_trace = fig.data[1]
        self.assertEqual(len(node_trace.x), 3, "Node trace should have data for 3 agents.")
        self.assertIn("agent-001", node_trace.text[0])
        self.assertIn("Score: 9.00", node_trace.text[2])

    @patch('webbrowser.open')
    @patch('plotly.graph_objects.Figure.write_html')
    async def test_02_visualize_lineage_no_agents(self, mock_write_html, mock_webbrowser):
        """
        Test (Edge Case): Ensures visualize_lineage handles an empty agent list
        gracefully without error.
        """
        # --- Arrange ---
        self.mock_archive_manager.query_agents = AsyncMock(return_value=[])

        # --- Act ---
        await self.visualizer.visualize_lineage()

        # --- Assert ---
        self.mock_archive_manager.query_agents.assert_called_once()
        mock_write_html.assert_not_called()
        mock_webbrowser.assert_not_called()

    # The patch path must now reflect the new absolute import path
    @patch('webbrowser.open')
    @patch('plotly.graph_objects.Figure.write_html')
    @patch('prometheus_agent.Visualizer.PCA')
    async def test_03_visualize_cognitive_space_success(self, mock_pca, mock_write_html, mock_webbrowser):
        """
        Test (Golden Path): Verifies visualize_cognitive_space correctly extracts
        vectors, performs dimensionality reduction, and plots the result.
        """
        # --- Arrange ---
        self.mock_archive_manager.query_agents = AsyncMock(return_value=self.test_population)

        mock_pca_instance = MagicMock()
        mock_pca_instance.fit_transform.return_value = np.array([[0, 1], [1, 0], [0.5, 0.5]])
        mock_pca.return_value = mock_pca_instance

        # --- Act ---
        await self.visualizer.visualize_cognitive_space(method='PCA')

        # --- Assert ---
        self.mock_archive_manager.query_agents.assert_called_once()
        mock_pca.assert_called_once_with(n_components=2)
        mock_pca_instance.fit_transform.assert_called_once()
        mock_write_html.assert_called_once()
        mock_webbrowser.assert_called_once()

        fig = mock_write_html.call_args[0][0]
        scatter_trace = fig.data[0]
        self.assertEqual(len(scatter_trace.x), 3)
        self.assertEqual(scatter_trace.x[0], 0)
        self.assertEqual(scatter_trace.y[1], 0)
        self.assertIn("Vector: [9. 8. 9. 9. 9.]", scatter_trace.text[2])

    @patch('webbrowser.open')
    @patch('plotly.graph_objects.Figure.write_html')
    async def test_04_visualize_cognitive_space_insufficient_data(self, mock_write_html, mock_webbrowser):
        """
        Test (Edge Case): Ensures cognitive space visualization is aborted if there
        are not enough agents with geometric data.
        """
        # --- Arrange ---
        self.mock_archive_manager.query_agents = AsyncMock(return_value=[self.agent1])

        # --- Act ---
        await self.visualizer.visualize_cognitive_space()

        # --- Assert ---
        self.mock_archive_manager.query_agents.assert_called_once()
        mock_write_html.assert_not_called()
        mock_webbrowser.assert_not_called()


if __name__ == '__main__':
    # This setup allows you to run this test file directly, assuming your
    # IDE's test runner is configured to start from the project root.
    unittest.main()
