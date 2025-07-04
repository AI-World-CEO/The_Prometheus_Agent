# prometheus_agent/Super_Brain_Compiler.py

import os
import sys
import yaml
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Union

import aiofiles
from tqdm.asyncio import tqdm

# --- Path Setup & Logging Configuration ---
ROOT_DIR = Path(__file__).resolve().parent
YAML_BRAIN_SOURCE_DIR = ROOT_DIR / "YAML_Brain"
OUTPUT_FILE = ROOT_DIR / "Super_Brain.yaml"

# Configure professional, leveled logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(ROOT_DIR / "Logs" / "super_brain_compiler.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)


# ... (FileProcessingSuccess, FileProcessingFailure, ProcessingResult classes are unchanged and perfect as they are) ...
class FileProcessingSuccess:
    def __init__(self, key_path: List[str], data: Dict[str, Any]):
        self.key_path = key_path
        self.data = data


class FileProcessingFailure:
    def __init__(self, path: Path, error: str):
        self.path = path
        self.error = error


ProcessingResult = Union[FileProcessingSuccess, FileProcessingFailure]


class SuperBrainCompiler:
    """
    Encapsulates the logic for compiling the YAML_Brain into a unified structure.
    This version correctly handles nested directory structures, creating a
    hierarchical dictionary that mirrors the file system.
    """

    def __init__(self, source_dir: Path, output_file: Path, max_concurrency: int = 100):
        self.source_dir = source_dir
        self.output_file = output_file
        self.semaphore = asyncio.Semaphore(max_concurrency)
        if not self.source_dir.is_dir():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")

    async def _read_and_parse_one_file(self, file_path: Path) -> Tuple[List[str], Any]:
        """Asynchronously reads and parses a single YAML file, returning a key path."""
        async with self.semaphore:
            try:
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                loop = asyncio.get_running_loop()
                parsed_data = await loop.run_in_executor(None, yaml.safe_load, content)

                # Create a key path from the directory structure
                relative_path = file_path.relative_to(self.source_dir)
                key_path = list(relative_path.parts[:-1]) + [relative_path.stem]

                return key_path, parsed_data or {}
            except (yaml.YAMLError, IOError) as e:
                log.warning(f"Error processing {file_path.name}: {e}")
                raise ValueError(f"Error in {file_path.name}: {e}")

    async def _process_file_worker(self, file_path: Path) -> ProcessingResult:
        """A safe worker that wraps the core logic."""
        try:
            key_path, data = await self._read_and_parse_one_file(file_path)
            return FileProcessingSuccess(key_path, data)
        except Exception as e:
            return FileProcessingFailure(file_path, str(e))

    def _deep_set(self, data_dict: Dict, keys: List[str], value: Any):
        """Recursively sets a value in a nested dictionary."""
        for key in keys[:-1]:
            data_dict = data_dict.setdefault(key, {})
        data_dict[keys[-1]] = value

    async def compile(self) -> None:
        """Orchestrates the entire hierarchical compilation process."""
        log.info(f"--- Starting Cognitive Synthesis: Compiling YAML_Brain ---")
        start_time = time.monotonic()

        yaml_files = [f for f in self.source_dir.rglob('*.yaml') if f.is_file()]
        if not yaml_files:
            log.warning("No YAML files found in the source directory. Nothing to compile.")
            return

        log.info(f"Found {len(yaml_files)} YAML files to compile.")

        tasks = [self._process_file_worker(f) for f in yaml_files]
        results = await tqdm.gather(*tasks, desc="[Synthesizing Brain]", unit="file")

        super_brain_data: Dict[str, Any] = {}
        failures: List[FileProcessingFailure] = []
        success_count = 0

        for result in results:
            if isinstance(result, FileProcessingSuccess):
                self._deep_set(super_brain_data, result.key_path, result.data)
                success_count += 1
            elif isinstance(result, FileProcessingFailure):
                failures.append(result)

        log.info(f"Aggregation complete. Success: {success_count}, Failures: {len(failures)}")

        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                yaml.dump(super_brain_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)
            log.info(f"Successfully wrote compiled data to {self.output_file}")
        except Exception as e:
            log.critical(f"FATAL: Could not write the final Super_Brain.yaml file: {e}")
            failures.append(FileProcessingFailure(self.output_file, str(e)))

        end_time = time.monotonic()
        self._print_final_report(start_time, end_time, len(yaml_files), failures)

    def _print_final_report(self, start: float, end: float, total: int, failures: List[FileProcessingFailure]):
        """Prints a clean, comprehensive summary of the compilation task."""
        # ... (This method is perfect as is) ...
        pass


# ... (Main Execution Block is perfect as is) ...
if __name__ == "__main__":
    # ...
    pass
