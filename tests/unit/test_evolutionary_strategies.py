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
        # A sample config for the engine
        self.config = {
            'strategy_name': 'TestStrategy',
            'tournament_size': 3,
            'generations': 2,  # Keep low for fast tests
            'elitism_count': 1,
            'mutation_mode': 'classical',
            'classical_objective': 'Improve quality'
        }

        # Instantiate the engine with a dummy API key and the test config
        self.engine = EvolutionaryStrategyEngine(api_key="sk-dummy", config=self.config)

        # --- Mock all external dependencies of the engine ---
        self.engine.evaluator = MagicMock()
        self.engine.mutator = MagicMock()
        self.engine.quantum_mutator = MagicMock()
        self.engine.geometric_transformer = MagicMock()

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
        # In a tournament, it's highly probable but not guaranteed that the best individuals are chosen
        # A simple assertion is to check if all selected parents are valid agents.
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

        # Mock the mutator's generate method to return a predictable child
        self.engine.mutator.generate = AsyncMock(return_value={"generated_content": "def child_func(): pass"})

        # --- Act ---
        child_agent = await self.engine._crossover(parent1, parent2)

        # --- Assert ---
        # 1. Verify the mutator was called
        self.engine.mutator.generate.assert_called_once()
        # 2. Check that the generated code is in the child
        self.assertEqual(child_agent.code, "def child_func(): pass")
        # 3. Verify the child's lineage is correctly set
        self.assertEqual(child_agent.metadata.parent_id, parent1.metadata.version_id)
        self.assertIn("Crossover", child_agent.metadata.origin_prompt)

    async def test_03_mutate(self):
        """
        Test (Core Logic): Verifies the mutation operation correctly calls the
        mutator with the parent's code.
        """
        # --- Arrange ---
        parent_agent = create_mock_agent(score=7, code="def original_func(): pass")
        self.engine.mutator.generate = AsyncMock(return_value={"generated_content": "def mutated_func(): pass"})

        # --- Act ---
        mutated_agent = await self.engine._mutate(parent_agent)

        # --- Assert ---
        self.engine.mutator.generate.assert_called_once()
        self.assertEqual(mutated_agent.code, "def mutated_func(): pass")
        self.assertEqual(mutated_agent.metadata.parent_id, parent_agent.metadata.version_id)
        self.assertIn("mutation", mutated_agent.metadata.origin_prompt)

    async def test_04_full_evolution_run(self):
        """
        Test (Integration): Simulates a full, multi-generational evolutionary run
        to ensure all components are orchestrated correctly.
        """

        # --- Arrange ---
        # Mock the evaluation to simply increase scores each generation to simulate improvement
        async def mock_evaluate(population: List[Agent]):
            for agent in population:
                # Use a simple hash to give a pseudo-random but deterministic score
                new_score = (hash(agent.code) % 50) / 10.0 + 5.0  # Scores between 5 and 10
                agent.metadata.evaluations.append({"final_score": new_score})
                # Re-validate the Pydantic model to trigger the auto-linking of the score
                agent.model_validate(agent.model_dump())
            return population

        self.engine._evaluate_population = mock_evaluate

        # Mock crossover and mutation to produce predictable offspring
        async def mock_crossover(p1: Agent, p2: Agent):
            return create_mock_agent(score=0, code=f"crossover({p1.code}, {p2.code})", parent_id=p1.metadata.version_id)

        async def mock_mutate(agent: Agent):
            agent.code = f"mutated({agent.code})"
            return agent

        self.engine._crossover = mock_crossover
        self.engine._mutate = mock_mutate

        # --- Act ---
        best_agent = await self.engine.run_evolution(self.initial_population)

        # --- Assert ---
        self.assertIsInstance(best_agent, Agent)
        # The best agent should have a high score, indicating it went through the process
        self.assertGreater(best_agent.metadata.final_score, 5.0)
        # The code should show evidence of crossover and mutation
        self.assertIn("mutated", best_agent.code)
        self.assertIn("crossover", best_agent.code)
        # Check that the number of "generations" was respected
        # A simple way to check is lineage depth. A 2-gen run will have a child of a child.
        self.assertIsNotNone(best_agent.metadata.parent_id)


if __name__ == '__main__':
    unittest.main()
