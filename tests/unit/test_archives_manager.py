# prometheus_agent/tests/unit/test_archives_manager.py

import unittest
import asyncio
from unittest.mock import patch, mock_open, MagicMock
import os
import shutil
import tempfile
import numpy as np

from prometheus_agent.Agent import Agent
from prometheus_agent.ArchivesManager import ArchiveManager, Base


# This structure assumes the test is run from the project root directory



# A-1: We need to patch the asyncio.run call inside the ArchiveManager's __init__
# to prevent it from running during test setup. We will call the load method manually.
@patch('prometheus_agent.ArchivesManager.asyncio.run', MagicMock())
class TestArchiveManager(unittest.IsolatedAsyncioTestCase):
    """
    A comprehensive unit test suite for the ArchiveManager.

    This suite validates the hybrid storage system (SQLite, FAISS, File Store)
    using an in-memory database and mocked file I/O to ensure fast, isolated,
    and reliable testing of the agent's persistence layer.
    """

    def setUp(self):
        """
        Set up a temporary directory for the archive and an ArchiveManager instance
        using an in-memory SQLite database for speed and isolation.
        """
        self.temp_dir = tempfile.mkdtemp()
        # The ArchiveManager will create subdirectories here, but file writes will be mocked.
        self.archive_root = os.path.join(self.temp_dir, "test_archive")

        # Instantiate ArchiveManager. The asyncio.run in its __init__ is patched.
        self.archive_manager = ArchiveManager(archive_root=self.archive_root)

        # Manually re-initialize the engine to use an in-memory database for this test run.
        from sqlalchemy import create_engine

        self.archive_manager.engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(self.archive_manager.engine)
        self.archive_manager.Session.configure(bind=self.archive_manager.engine)

        # We also manually call the load method now that the in-memory DB is set up.
        asyncio.run(self.archive_manager._load_from_persistence())

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.temp_dir)

    def _create_test_agent(self, score: float, vector: list, parent_id: str = None) -> Agent:
        """Helper function to create a validated Agent object for testing."""
        eval_report = {
            "final_score": score,
            "geometric_state_vector": vector,
        }
        agent_data = {
            "metadata": {
                "parent_id": parent_id,
                "origin_prompt": f"Test agent with score {score}",
                "evaluations": [eval_report]
            },
            "code": f"def solve(): return {score}"
        }
        return Agent.model_validate(agent_data)

    async def test_01_save_and_get_agent_by_id(self):
        """
        Test (Core Logic): Verifies that saving an agent correctly persists it
        and that it can be retrieved accurately by its ID.
        """
        # --- Arrange ---
        agent_to_save = self._create_test_agent(score=8.5, vector=[8, 5, 9, 9, 2])
        agent_id = agent_to_save.metadata.version_id

        # Mock file system operations to keep the test isolated
        with patch("builtins.open", mock_open()) as mock_file, \
                patch("os.makedirs", MagicMock()):
            # --- Act ---
            await self.archive_manager.save_agent(agent_to_save)
            retrieved_agent = await self.archive_manager.get_agent_by_id(agent_id)

            # --- Assert ---
            # 1. Verify file system was interacted with
            mock_file.assert_called_once_with(
                os.path.join(self.archive_root, "code_store", f"{agent_id}.py"),
                'w', encoding='utf-8'
            )

            # 2. Verify FAISS index was updated
            self.assertEqual(self.archive_manager.faiss_index.ntotal, 1)

            # 3. Verify database record exists (checked implicitly by get_agent_by_id)
            # 4. Verify retrieved agent is identical to the saved one
            self.assertIsNotNone(retrieved_agent)
            self.assertEqual(agent_to_save, retrieved_agent)

            # 5. Verify caching mechanism
            # The second call should hit the cache and not touch the DB
            with patch.object(self.archive_manager.Session, 'execute') as mock_db_execute:
                retrieved_again = await self.archive_manager.get_agent_by_id(agent_id)
                mock_db_execute.assert_not_called()
                self.assertEqual(agent_to_save, retrieved_again)

    async def test_02_find_similar_agents(self):
        """
        Test (FAISS Logic): Verifies the vector similarity search correctly
        identifies the nearest neighbors.
        """
        # --- Arrange ---
        # Create a population of agents with distinct vectors
        agent1 = self._create_test_agent(score=9.0, vector=[9, 9, 9, 9, 9])  # Target
        agent2 = self._create_test_agent(score=8.9, vector=[8, 9, 9, 9, 9])  # Most similar
        agent3 = self._create_test_agent(score=5.0, vector=[1, 2, 1, 3, 1])  # Least similar
        agent4 = self._create_test_agent(score=7.0, vector=[7, 7, 6, 8, 7])  # Moderately similar

        with patch("builtins.open", mock_open()), patch("os.makedirs", MagicMock()):
            await self.archive_manager.save_agent(agent1)
            await self.archive_manager.save_agent(agent2)
            await self.archive_manager.save_agent(agent3)
            await self.archive_manager.save_agent(agent4)

        # --- Act ---
        # Find the top 2 agents most similar to agent1
        similar_agents_with_dist = await self.archive_manager.find_similar_agents(agent1, k=2)

        # --- Assert ---
        self.assertEqual(len(similar_agents_with_dist), 2)

        # Unpack results
        similar_agents = [agent for agent, dist in similar_agents_with_dist]

        # The most similar agent should be agent2
        self.assertEqual(similar_agents[0].metadata.version_id, agent2.metadata.version_id)
        # The next most similar should be agent4
        self.assertEqual(similar_agents[1].metadata.version_id, agent4.metadata.version_id)
        # The least similar (agent3) should not be in the top 2 results
        self.assertNotIn(agent3, similar_agents)

    async def test_03_query_agents_by_score(self):
        """
        Test (DB Query Logic): Verifies that querying for agents by score
        returns the correct agents in the correct order.
        """
        # --- Arrange ---
        agent_low = self._create_test_agent(score=4.5, vector=[4, 5, 4, 5, 4])
        agent_mid = self._create_test_agent(score=7.2, vector=[7, 2, 7, 2, 7])
        agent_high = self._create_test_agent(score=9.8, vector=[9, 8, 9, 8, 9])

        with patch("builtins.open", mock_open()), patch("os.makedirs", MagicMock()):
            await self.archive_manager.save_agent(agent_low)
            await self.archive_manager.save_agent(agent_mid)
            await self.archive_manager.save_agent(agent_high)

        # --- Act ---
        # Query for agents with a score of 7.0 or higher
        top_agents = await self.archive_manager.query_agents(min_score=7.0)

        # --- Assert ---
        self.assertEqual(len(top_agents), 2)
        # Results should be ordered by score descending
        self.assertEqual(top_agents[0].metadata.version_id, agent_high.metadata.version_id)
        self.assertEqual(top_agents[1].metadata.version_id, agent_mid.metadata.version_id)
        # The low-scoring agent should not be in the results
        self.assertNotIn(agent_low, top_agents)

    async def test_04_get_agent_by_id_not_found(self):
        """
        Test (Edge Case): Verifies that getting a non-existent agent ID
        correctly returns None.
        """
        # --- Act ---
        result = await self.archive_manager.get_agent_by_id("non-existent-id")

        # --- Assert ---
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
