# bootstrap_brain.py
import os, sys, yaml, asyncio, shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "Prometheus_Agent"
CORPUS_DIR = SOURCE_ROOT / "Corpus"
YAML_BRAIN_DIR = SOURCE_ROOT / "YAML_Brain"
SUPER_BRAIN_FILE = SOURCE_ROOT / "Super_Brain.yaml"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
try:
    from Super_Brain_Compiler import SuperBrainCompiler
except ImportError:
    print(f"FATAL: Could not import SuperBrainCompiler. Make sure 'Super_Brain_Compiler.py' exists inside '{SOURCE_ROOT}'.")
    sys.exit(1)

def bootstrap_yaml_from_corpus():
    print("--- Phase 1: High-Speed YAML Bootstrapping ---")
    if not CORPUS_DIR.is_dir():
        print(f"ERROR: Corpus directory not found at '{CORPUS_DIR}'. Aborting.")
        return 0
    if YAML_BRAIN_DIR.is_dir():
        print(f"Purging old YAML brain fragments from '{YAML_BRAIN_DIR}' for a clean build...")
        shutil.rmtree(YAML_BRAIN_DIR)
    YAML_BRAIN_DIR.mkdir(parents=True, exist_ok=True)
    txt_files = list(CORPUS_DIR.rglob('*.txt'))
    if not txt_files:
        print("No .txt files found in Corpus. Nothing to bootstrap.")
        return 0
    print(f"Found {len(txt_files)} text files in Corpus for transmutation.")
    processed_count = 0
    for txt_file in txt_files:
        try:
            content = txt_file.read_text(encoding='utf-8', errors='ignore').strip()
            if not content: continue
            yaml_data = {'id': txt_file.stem, 'source_file': str(txt_file.relative_to(SOURCE_ROOT)), 'type': 'knowledge_document_bootstrapped', 'tags': [p.name for p in txt_file.relative_to(CORPUS_DIR).parents if p.name != '.'], 'content': content}
            relative_path = txt_file.relative_to(CORPUS_DIR)
            yaml_output_path = (YAML_BRAIN_DIR / relative_path).with_suffix('.yaml')
            yaml_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(yaml_output_path, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, indent=2, sort_keys=False)
            processed_count += 1
        except Exception as e:
            print(f"Could not process '{txt_file.name}': {e}")
    print(f"\n--- Successfully transmuted {processed_count} documents into YAML fragments. ---")
    return processed_count

async def compile_brain():
    print(f"\n--- Phase 2: Compiling Fragments into Unified Super_Brain ---")
    compiler = SuperBrainCompiler(source_dir=YAML_BRAIN_DIR, output_file=SUPER_BRAIN_FILE)
    await compiler.compile()

async def main():
    if SUPER_BRAIN_FILE.exists():
        print(f"An existing '{SUPER_BRAIN_FILE.name}' was found.")
        user_input = input("Do you want to delete it and perform a full brain rebuild? (y/n): ").lower().strip()
        if user_input != 'y':
            print("Operation cancelled by user.")
            return
        try:
            SUPER_BRAIN_FILE.unlink()
            print(f"Removed old '{SUPER_BRAIN_FILE.name}'.")
        except OSError as e:
            print(f"Error removing old brain file: {e}")
            return
    if bootstrap_yaml_from_corpus() > 0:
        await compile_brain()
    else:
        print("No YAML files were created, so compilation is not necessary.")

if __name__ == "__main__":
    asyncio.run(main())