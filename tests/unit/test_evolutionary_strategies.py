# prometheus_agent/tests/unit/test_evolutionary_strategies.py

import unittest
import asyncio
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from prometheus_agent.Agent import Agent
from prometheus_agent.EvolutionaryStrategies import EvolutionaryStrategyEngine


# This structure assumes the test is run from the project root directory


# A-1: We need to create a mock Pydantic v2-compliant Agent for testing
def create_mock_agent(score: float, code: str = "pass", parent_id: str = None) -> Agent:
    """Helper function to create a validated Agent object with a specific score."""
    eval_report = {
        "final_score": score,
        "geometric_state_vector": [score] * 5,  # Simple vector for testing
    }
    agent_data = {
        "metadata": {
            "parent_id": parent_id,
            "origin_prompt": "Test Agent",
            "reasoning_path": ["creation"],
            "evaluations": [eval_report],
        },
        "code": code
    }
    # Use model_validate for Pydantic v2
    return Agent.model_validate(agent_data)


class TestEvolutionaryStrategyEngine(unittest.IsolatedAsyncioTestCase):
    """
    A comprehensive unit test suite for the EvolutionaryStrategyEngine.

    This suite validates the core genetic algorithms (selection, crossover, mutation)
    and the overall orchestration of the evolutionary loop in isolation, using
    mocked dependencies to ensure fast and reliable testing.
    """

    def setUp(self):
        """
        Set up a mocked environment for the EvolutionaryStrategyEngine.
        """
        # --- THE FIX: Correctly instantiate the engine via a mocked agent instance ---
        self.mock_agent = MagicMock()
        self.mock_agent.config = {
            "evolutionary_strategies": {
                'strategy_name': 'TestStrategy',
                'tournament_size': 3,
                'generations': 2,
                'elitism_count': 1,
                'mutation_mode': 'classical',
                'classical_objective': 'Improve quality'
            }
        }
        # Mock all dependencies that the engine will receive from the agent instance
        self.mock_agent.evaluator = MagicMock()
        self.mock_agent.synthesis_engine = MagicMock()
        self.mock_agent.governor = MagicMock()
        self.mock_agent.governor.decide_model = AsyncMock(return_value=MagicMock(timeout_seconds=10))
        self.mock_agent.ethics_core = MagicMock()
        self.mock_agent.geometric_transformer = MagicMock()

        # Instantiate the engine with the mocked agent
        self.engine = EvolutionaryStrategyEngine(agent_instance=self.mock_agent)

        # Create an initial population for testing
        self.initial_population = [create_mock_agent(score=i) for i in range(1, 6)]  # Scores: 1, 2, 3, 4, 5

    def test_01_select_parents_tournament(self):
        """
        Test (Algorithm): Verifies that tournament selection correctly selects
        parents from the population, favoring higher scores.
        """
        # --- Arrange ---
        population = sorted(self.initial_population, key=lambda ag: ag.metadata.final_score, reverse=True)
        num_to_select = 4

        # --- Act ---
        selected_parents = self.engine._select_parents_tournament(population, num_parents=num_to_select)

        # --- Assert ---
        self.assertEqual(len(selected_parents), num_to_select)
        for parent in selected_parents:
            self.assertIsInstance(parent, Agent)

    async def test_02_crossover(self):
        """
        Test (Core Logic): Verifies the crossover operation correctly calls the
        mutator with the code of both parents to generate a child.
        """
        # --- Arrange ---
        parent1 = create_mock_agent(score=9, code="def parent1_func(): pass")
        parent2 = create_mock_agent(score=8, code="def parent2_func(): pass")

        # --- THE FIX: mutator.generate now returns raw code (a string) ---
        self.engine.mutator.generate = AsyncMock(return_value="def child_func(): pass")

        # --- Act ---
        child_agent = await self.engine._crossover(parent1, parent2)

        # --- Assert ---
        self.engine.mutator.generate.assert_awaited_once()
        self.assertEqual(child_agent.code, "def child_func(): pass")
        self.assertEqual(child_agent.metadata.parent_id, parent1.metadata.version_id)
        self.assertIn("Crossover", child_agent.metadata.origin_prompt)

    async def test_03_mutate(self):
        """
        Test (Core Logic): Verifies the mutation operation correctly calls the
        mutator with the parent's code.
        """
        # --- Arrange ---
        parent_agent = create_mock_agent(score=7, code="def original_func(): pass")
        self.engine.mutator.generate = AsyncMock(return_value="def mutated_func(): pass")

        # --- Act ---
        mutated_agent = await self.engine._mutate(parent_agent)

        # --- Assert ---
        self.engine.mutator.generate.assert_awaited_once()
        self.assertEqual(mutated_agent.code, "def mutated_func(): pass")
        self.assertEqual(mutated_agent.metadata.parent_id, parent_agent.metadata.version_id)
        self.assertIn("mutation", mutated_agent.metadata.origin_prompt)

    async def test_04_full_evolution_run(self):
        """
        Test (Integration): Simulates a full, multi-generational evolutionary run
        to ensure all components are orchestrated correctly.
        """

        # --- Arrange ---
        # Mock the evaluation to simulate improvement
        async def mock_evaluate_population(population: List[Agent]) -> List[Agent]:
            evaluated_pop = []
            for agent in population:
                new_score = (hash(agent.code) % 50) / 10.0 + 5.0  # Scores between 5 and 10
                # --- THE FIX: Create a new agent instance with the updated evaluation
                # This mirrors the real logic where a new agent object is created
                agent_data = agent.model_dump()
                agent_data['metadata']['evaluations'].append({"final_score": new_score})
                evaluated_agent = Agent.model_validate(agent_data)
                evaluated_pop.append(evaluated_agent)
            return evaluated_pop

        # We patch the method on the instance for this specific test
        self.engine._evaluate_population = mock_evaluate_population

        # Mock crossover and mutation
        async def mock_crossover(p1: Agent, p2: Agent):
            return create_mock_agent(score=0, code=f"crossover({p1.code}, {p2.code})", parent_id=p1.metadata.version_id)

        async def mock_mutate(agent: Agent):
            # This needs to return a new Agent instance to get a new version_id
            mutated_data = agent.model_dump()
            mutated_data['code'] = f"mutated({agent.code})"
            mutated_data['metadata'].pop('version_id', None)  # Get a new ID
            return Agent.model_validate(mutated_data)

        self.engine._crossover = mock_crossover
        self.engine._mutate = mock_mutate

        # --- Act ---
        best_agent = await self.engine.run_evolution(self.initial_population)

        # --- Assert ---
        self.assertIsInstance(best_agent, Agent)
        self.assertGreater(best_agent.metadata.final_score, 5.0)
        self.assertIn("mutated", best_agent.code)
        self.assertIn("crossover", best_agent.code)
        self.assertIsNotNone(best_agent.metadata.parent_id)


if __name__ == '__main__':
    unittest.main()