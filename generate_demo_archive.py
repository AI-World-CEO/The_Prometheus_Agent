# prometheus_agent/generate_demo_archive.py

import os
import sys
import asyncio
import shutil
import numpy as np
from typing import List

# --- Path Setup ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from prometheus_agent.Agent import Agent
from prometheus_agent.ArchivesManager import ArchiveManager

async def create_demo_archive(archive_root="Archives/main_archive/", num_agents=20):
    """
    Generates a new, richly populated archive for demonstration and testing.
    This script will DELETE any existing archive at the specified location.
    """
    print(f"--- Generating Demo Archive at '{archive_root}' ---")
    archive_path = os.path.join(project_root, archive_root)

    if os.path.exists(archive_path):
        print(f"Warning: Existing archive at '{archive_path}' will be deleted.")
        shutil.rmtree(archive_path)

    # Use the async factory to create the archive manager
    archive = await ArchiveManager.create(archive_root=archive_path)

    # Create a population of diverse agents
    agents_to_save: List[Agent] = []
    parent_id = None

    for i in range(num_agents):
        # --- FIX: Restored the entire agent creation logic inside the loop ---
        # Simulate different quality levels and states
        score = 5.0 + (i / num_agents) * 5.0  # Scores from 5.0 up to just under 10.0

        # Create a plausible geometric vector based on the score
        perf = score + np.random.uniform(-0.5, 0.5)
        clarity = 10.0 - (score / 2) + np.random.uniform(-1, 1)  # Lower score = higher clarity
        brevity = 10.0 - (score / 3)
        safety = 9.5 + np.random.uniform(-0.5, 0.5)
        novelty = (i % 5) * 2.0 + np.random.uniform(-0.5, 0.5)

        vector = np.clip([perf, clarity, brevity, safety, novelty], 0, 10).tolist()

        # Create the evaluation report
        eval_report = {
            "final_score": round(score, 2),
            "geometric_state_vector": vector,
            "dimensions": ["Performance", "Clarity", "Brevity", "Safety", "Novelty"],
            "breakdown": dict(zip(["Performance", "Clarity", "Brevity", "Safety", "Novelty"], vector))
        }

        # Create the Agent object
        agent = Agent(
            metadata={
                "parent_id": parent_id,
                "origin_prompt": f"Evolved solution for task #{i}",
                "reasoning_path": ["code_generation", "final_explanation"],
                "evaluations": [eval_report]
            },
            code=f"# Agent {i + 1}\ndef solve():\n    # Solution with performance score {score:.2f}\n    return {i + 1}"
        )

        agents_to_save.append(agent)

        # The next agent will be a child of this one
        parent_id = agent.metadata.version_id

    # Save all agents concurrently
    print(f"Saving {len(agents_to_save)} generated agents to the archive...")
    await asyncio.gather(*(archive.save_agent(ag) for ag in agents_to_save))

    print("\n--- Demo Archive Generation Complete ---")
    print(f"Database created at: '{archive.db_path}'")
    print(f"Code store created at: '{archive.code_store_path}'")

    # Query for a high-scoring agent to confirm success
    top_agents = await archive.query_agents(limit=1, min_score=9.5)
    if top_agents:
        best_agent = top_agents[0]
        print(f"Best agent in demo archive has score: {best_agent.metadata.final_score:.2f}")
    else:
        # If no agent scores that high, just get the best one overall.
        top_agents = await archive.query_agents(limit=1)
        if top_agents:
            best_agent = top_agents[0]
            print(f"Best agent in demo archive has score: {best_agent.metadata.final_score:.2f}")
        else:
            print("No agents found in the archive.")


if __name__ == "__main__":
    # Run this script to create your populated database file.
    asyncio.run(create_demo_archive())