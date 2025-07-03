# populate_knowledge_base.py

import os
import sys
import json
import yaml
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from tqdm.asyncio import tqdm
import logging

# --- Path Setup & Logging ---
# This ensures the script can find all the necessary agent components.
PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "Prometheus_Agent"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging to see the progress
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
log = logging.getLogger("KnowledgePopulator")

# --- Agent Component Imports ---
from Prometheus_Agent.Src.Mutator import Mutator, SynthesisError
from Prometheus_Agent.Prometheus.CognitiveGovernor import CognitiveGovernor
from Prometheus_Agent.Prometheus.Ethics_Core_Foundation import EthicsCoreFoundation

# --- The Knowledge Structure You Defined ---
# This is your architectural blueprint, converted into a Python dictionary.
KNOWLEDGE_STRUCTURE = {
    "Axioms_Knowledgebase": {
        "Ancient_Civilizations": {
            "Egypt": ["Imhotep", "Hipparchia of Maroneia"],
            "Greece": ["Socrates", "Plato", "Aristotle", "Euclid", "Archimedes", "Hypatia"],
            "China": ["Confucius", "Laozi", "Zhuangzi", "Hua Tuo", "Zu Chongzhi"],
            "India": ["Chanakya", "Panini", "Aryabhata", "Sushruta", "Nagarjuna"],
            "Middle_East": ["Ibn Sina", "Al-Khwarizmi", "Al-Farabi", "Moses Maimonides"]
        },
        "Thinkers_By_Institution": {
            "Harvard": ["John Nash", "W.E.B. Du Bois", "Sally Ride", "Barack Obama", "Steven Pinker",
                        "Henry Kissinger"],
            "Princeton": ["Albert Einstein", "Richard Feynman", "Kurt Gödel", "John von Neumann", "Alan Turing",
                          "Edward Witten"],
            "MIT": ["Noam Chomsky", "Marvin Minsky", "Ray Kurzweil", "Emanuel Derman", "Tim Berners-Lee",
                    "Kendi Young"],
            "Stanford": ["Sergey Brin", "Larry Page", "Condoleezza Rice", "Vinton Cerf", "John McCarthy",
                         "Ellen Ochoa"],
            "Oxford": ["Stephen Hawking", "J.R.R. Tolkien", "Margaret Thatcher", "Oscar Wilde", "Dorothy Hodgkin"],
            "Cambridge": ["Isaac Newton", "Charles Darwin", "James Clerk Maxwell", "Srinivasa Ramanujan",
                          "Francis Crick"],
            "ETH_Zurich": ["Niklaus Wirth", "Richard R. Ernst"],
            "Caltech": ["Linus Pauling", "Gordon Moore", "Charles H. Townes"],
            "Tokyo_University": ["Hideki Yukawa", "Akira Yoshino", "Shinya Yamanaka", "Kenichi Fukui"],
            "Tsinghua_University": ["Xi Jinping", "Yang Zhenning", "Qian Xuesen"],
            "Moscow_State_University": ["Andrey Kolmogorov", "Sergei Kapitsa", "Grigori Perelman", "Sofya Kovalevskaya"]
        },
        "Thinkers_By_Field": {
            "Arts_Literature": ["Leonardo da Vinci", "William Shakespeare", "Frida Kahlo", "Vincent van Gogh",
                                "Maya Angelou"],
            "Medicine_Biology": ["Louis Pasteur", "Rosalind Franklin", "James Watson", "Barbara McClintock",
                                 "Jane Goodall"],
            "Mathematics": ["Carl Friedrich Gauss", "Bernhard Riemann", "Pierre de Fermat", "Sophie Germain",
                            "Paul Erdős", "Emmy Noether"],
            "Physics": ["Max Planck", "Niels Bohr", "Enrico Fermi", "Marie Curie", "Lise Meitner"],
            "Computer_Science": ["Ada Lovelace", "Grace Hopper", "Donald Knuth", "Geoffrey Hinton"],
            "Engineering_Innovation": ["Nikola Tesla", "Thomas Edison", "Hedy Lamarr", "The Wright Brothers",
                                       "Steve Jobs"],
            "Business_Economics": ["Warren Buffett", "John Maynard Keynes", "Amartya Sen", "Oprah Winfrey",
                                   "Sheryl Sandberg"],
            "Social_Sciences": ["Sigmund Freud", "Carl Jung", "Ruth Benedict", "Frantz Fanon", "Jane Addams"],
            "Philosophy": ["Immanuel Kant", "Friedrich Nietzsche", "Simone de Beauvoir", "Mahatma Gandhi",
                           "Jean-Paul Sartre", "Hannah Arendt"],
            "Humanitarian": ["Mother Teresa", "Martin Luther King Jr.", "Malala Yousafzai", "Neil deGrasse Tyson"]
        }
    }
}


class KnowledgePopulator:
    """
    An orchestration engine to autonomously populate the YAML_Brain
    based on a predefined knowledge structure.
    """

    def __init__(self, config_path: Path):
        self.yaml_brain_root = SOURCE_ROOT / "YAML_Brain"
        self.config = self._load_config(config_path)

        # Instantiate the necessary agent components
        model_config = self.config.get("llm_models", {})
        self.mutator = Mutator(model_config, self.config.get("cognitive_toolkit", {}))
        self.governor = CognitiveGovernor(self)  # Pass self as a mock agent
        self.ethics_core = EthicsCoreFoundation(self)

        self.skill_map = {}  # Not needed for this task, but required by governor

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        with config_path.open('r', encoding='utf-8') as f:
            return json.load(f)

    def get_exemplar(self) -> str:
        """Loads a high-quality YAML file to use as an exemplar for the LLM."""
        exemplar_path = self.yaml_brain_root / "4_metacognition" / "Axiom_4.1_combinatorial_creativity.yaml"
        if exemplar_path.exists():
            return exemplar_path.read_text(encoding='utf-8')
        return "id: example-id\ntitle: Example Title\n..."

    async def _generate_axiom_text(self, person: str) -> str:
        """Phase 1: Research the person's core ideas."""
        objective = (
            f"You are a concise historian and philosopher. Research the historical figure or thinker '{person}' and "
            f"distill their most important, impactful, and enduring principles, axioms, or core philosophies into a few paragraphs. "
            f"Focus on the ideas they are most known for that have stood the test of time. Present this as a brief, well-written summary."
        )
        decision = self.governor.decide_model(skill='knowledge_research', domain='history')
        return await self.mutator.generate(
            user_objective=objective,
            model_to_use=decision.model_name,
            system_prompt=self.ethics_core.get_ethical_system_prompt(),
            output_mode='raw'
        )

    async def _refine_text_to_yaml(self, person: str, raw_text: str) -> str:
        """Phase 2: Refine the researched text into a structured YAML document."""
        exemplar = self.get_exemplar()
        objective = (
            f"You are a master ontologist. The following text contains the core principles of '{person}'. "
            f"Your task is to completely rewrite and restructure this text into a high-level YAML protocol. "
            f"Deconstruct the core ideas, principles, and directives into a deeply nested, hierarchical format. "
            f"The output must be ONLY the raw, perfectly formatted YAML text. Adhere to the structure of the provided exemplar.\n\n"
            f"### EXEMPLAR OF TARGET STRUCTURE ###\n{exemplar}\n\n"
            f"### RAW TEXT TO REFINE ###\n{raw_text}"
        )
        decision = self.governor.decide_model(skill='knowledge_refinement', domain='metacognition')
        return await self.mutator.generate(
            user_objective=objective,
            model_to_use=decision.model_name,
            system_prompt=self.ethics_core.get_ethical_system_prompt(),
            output_mode='raw'
        )

    async def process_person(self, path_parts: List[str], person_name: str, index: int):
        """
        The main worker function for a single person. Orchestrates research,
        refinement, and file writing.
        """
        try:
            # 1. Define File Path
            clean_name = person_name.lower().replace(" ", "_").replace(".", "")
            filename = f"{index + 1:02d}_axiom_of_{clean_name}.yaml"
            output_path = self.yaml_brain_root.joinpath(*path_parts, filename)

            if output_path.exists():
                # log.info(f"Skipping '{person_name}', file already exists.")
                return

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 2. Research
            raw_text = await self._generate_axiom_text(person_name)
            if not raw_text or len(raw_text) < 50:
                raise ValueError("Research phase produced insufficient text.")

            # 3. Refine
            refined_yaml = await self._refine_text_to_yaml(person_name, raw_text)
            yaml.safe_load(refined_yaml)  # Validate YAML syntax

            # 4. Write to file
            with output_path.open('w', encoding='utf-8') as f:
                f.write(refined_yaml)

        except (SynthesisError, ValueError, yaml.YAMLError) as e:
            log.error(f"Failed to process '{person_name}'. Reason: {e}")
        except Exception as e:
            log.critical(f"An unexpected error occurred for '{person_name}'.", exc_info=True)


async def main():
    """Main execution block."""
    log.info("--- Starting Autonomous Knowledge Base Population ---")

    config_path = SOURCE_ROOT / "config.json"
    if not config_path.exists():
        log.critical(f"FATAL: config.json not found at '{config_path}'")
        return

    populator = KnowledgePopulator(config_path)

    tasks = []

    # Recursively create tasks from the knowledge structure
    def create_tasks_recursive(data, path):
        for key, value in data.items():
            current_path = path + [key]
            if isinstance(value, dict):
                create_tasks_recursive(value, current_path)
            elif isinstance(value, list):
                for i, person in enumerate(value):
                    task = populator.process_person(current_path, person, i)
                    tasks.append(task)

    create_tasks_recursive(KNOWLEDGE_STRUCTURE, [])

    log.info(f"Found {len(tasks)} knowledge entries to generate.")

    # Run all tasks concurrently with a progress bar
    await tqdm.gather(*tasks, desc="[Populating Knowledgebase]")

    log.info("--- Knowledge Base Population Complete ---")
    log.info("Run 'python bootstrap_brain.py' to compile the new knowledge.")


if __name__ == "__main__":
    # This script is computationally intensive and may take a long time to run.
    # It will make hundreds of LLM calls.
    asyncio.run(main())