# Save this file as clean_brain_filenames.py in your project root

import os
import re
import logging

# Set up simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# This path points to the YAML_Brain inside your source code package
BRAIN_ROOT = 'prometheus_agent/YAML_Brain'

# This pattern will find 'Axiom_...' and 'ultra_asi_test_suite' numbers
# It looks for a word, then an underscore, then numbers/dots, then another underscore.
# It also handles just numbers at the start of the filename.
PREFIX_PATTERN = re.compile(r'^([A-Za-z_]+_)?\d+(\.\d+)*_')


def clean_filenames_in_brain():
    """
    Walks through the YAML_Brain directory and renames files
    by removing the brittle numeric/axiom prefixes.
    """
    if not os.path.isdir(BRAIN_ROOT):
        logging.error(
            f"Brain directory not found at '{BRAIN_ROOT}'. Make sure you are running this script from the 'The_prometheus_agent' root folder.")
        return

    logging.info(f"Starting filename cleanup in '{BRAIN_ROOT}'...")
    total_renamed = 0

    for dirpath, _, filenames in os.walk(BRAIN_ROOT):
        for filename in filenames:
            # Check if the filename matches our pattern
            if PREFIX_PATTERN.match(filename):
                # Construct the new filename by removing the prefix
                new_filename = PREFIX_PATTERN.sub('', filename)

                old_filepath = os.path.join(dirpath, filename)
                new_filepath = os.path.join(dirpath, new_filename)

                try:
                    os.rename(old_filepath, new_filepath)
                    logging.info(f'Renamed: "{filename}" -> "{new_filename}"')
                    total_renamed += 1
                except OSError as e:
                    logging.error(f"Could not rename {old_filepath}: {e}")

    logging.info(f"Cleanup complete. Total files renamed: {total_renamed}")


if __name__ == "__main__":
    clean_filenames_in_brain()
