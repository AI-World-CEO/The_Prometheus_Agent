# tests/unit/test_visualizer.py

import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

# --- Correct, Absolute Imports from the Main Source Package ---
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
        self.visualizer = Visualizer(archive_manager=self.mock_archive_manager)

        # Create a sample population of agents for the tests
        self.agent1 = create_mock_agent(score=5.0, vector=[5, 5, 5, 5, 5], agent_id="agent-001")
        self.agent2 = create_mock_agent(score=7.5, vector=[7, 8, 7, 8, 7], agent_id="agent-002", parent_id="agent-001")
        self.agent3 = create_mock_agent(score=9.0, vector=[9, 8, 9, 9, 9], agent_id="agent-003", parent_id="agent-001")
        self.test_population = [self.agent1, self.agent2, self.agent3]

    # --- THE FIX: The patch target is the Figure class itself, not its methods. ---
    @patch('prometheus_agent.Visualizer.go.Figure')
    @patch('prometheus_agent.Visualizer.webbrowser.open')
    async def test_01_visualize_lineage_success(self, mock_webbrowser, mock_figure_class):
        """
        Test (Golden Path): Verifies visualize_lineage correctly processes agents,
        builds a graph, and calls the plotting function with valid data.
        """
        # --- Arrange ---
        self.mock_archive_manager.query_agents = AsyncMock(return_value=self.test_population)

        # Create a mock instance that will be returned when Figure() is called
        mock_fig_instance = MagicMock()
        mock_figure_class.return_value = mock_fig_instance

        # --- Act ---
        await self.visualizer.visualize_lineage()

        # --- Assert ---
        self.mock_archive_manager.query_agents.assert_called_once()
        mock_figure_class.assert_called_once()
        mock_fig_instance.write_html.assert_called_once()
        mock_webbrowser.assert_called_once()

        # Get the arguments passed to the Figure constructor
        fig_args, fig_kwargs = mock_figure_class.call_args
        fig_data = fig_kwargs.get('data', [])

        self.assertEqual(len(fig_data), 2, "Figure should be created with two traces (edges, nodes).")
        node_trace = fig_data[1]  # The second trace is for nodes

        self.assertEqual(len(node_trace.x), 3, "Node trace should have data for 3 agents.")
        # We can't know the order due to the graph layout, so we check for presence
        self.assertTrue(any("agent-001" in text for text in node_trace.text))
        self.assertTrue(any("Score: 9.00" in text for text in node_trace.text))

    @patch('prometheus_agent.Visualizer.webbrowser.open')
    @patch('prometheus_agent.Visualizer.go.Figure')
    async def test_02_visualize_lineage_no_agents(self, mock_figure_class, mock_webbrowser):
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
        mock_figure_class.assert_not_called()
        mock_webbrowser.assert_not_called()

    @patch('prometheus_agent.Visualizer.webbrowser.open')
    @patch('prometheus_agent.Visualizer.go.Figure')
    @patch('prometheus_agent.Visualizer.PCA')
    async def test_03_visualize_cognitive_space_success(self, mock_pca, mock_figure_class, mock_webbrowser):
        """
        Test (Golden Path): Verifies visualize_cognitive_space correctly extracts
        vectors, performs dimensionality reduction, and plots the result.
        """
        # --- Arrange ---
        self.mock_archive_manager.query_agents = AsyncMock(return_value=self.test_population)

        mock_pca_instance = MagicMock()
        mock_pca_instance.fit_transform.return_value = np.array([[0, 1], [1, 0], [0.5, 0.5]])
        mock_pca.return_value = mock_pca_instance

        mock_fig_instance = MagicMock()
        mock_figure_class.return_value = mock_fig_instance

        # --- Act ---
        await self.visualizer.visualize_cognitive_space(method='PCA')

        # --- Assert ---
        self.mock_archive_manager.query_agents.assert_called_once()
        mock_pca.assert_called_once_with(n_components=2)
        mock_pca_instance.fit_transform.assert_called_once()
        mock_figure_class.assert_called_once()
        mock_fig_instance.write_html.assert_called_once()
        mock_webbrowser.assert_called_once()

        fig_args, fig_kwargs = mock_figure_class.call_args
        fig_data = fig_kwargs.get('data', [])

        scatter_trace = fig_data[0]
        self.assertEqual(len(scatter_trace.x), 3)
        self.assertEqual(scatter_trace.x[0], 0)
        self.assertEqual(scatter_trace.y[1], 0)
        self.assertTrue(any("Vector: [9. 8. 9. 9. 9.]" in text for text in scatter_trace.text))

    @patch('prometheus_agent.Visualizer.webbrowser.open')
    @patch('prometheus_agent.Visualizer.go.Figure')
    async def test_04_visualize_cognitive_space_insufficient_data(self, mock_figure_class, mock_webbrowser):
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
        mock_figure_class.assert_not_called()
        mock_webbrowser.assert_not_called()