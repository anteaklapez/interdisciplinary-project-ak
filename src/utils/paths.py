"""
Centralized system for file or directory creation

Every script imports DATA_DIR without needing to locally build each filepath
or manually create folders. RUN_VERSION can be only changed here instead of 
changing it multiple times within the code.
"""

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Where current run WRITES outputs
RUN_VERSION = 'V0.2.0'
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'runs', RUN_VERSION)

# Where current run READS outputs
SOURCE_RUN_VERSION = 'V0.2.0'
SOURCE_DIR = os.path.join(ROOT_DIR, 'data', 'runs', SOURCE_RUN_VERSION)

def data_path(dirname: str, filename: str) -> str:
    """Build an absolute path inside the current run's data folder,
    creating the folder if it does not exist yet."""
    full_dir = os.path.join(DATA_DIR, dirname)
    os.makedirs(full_dir, exist_ok=True)
    return os.path.join(DATA_DIR, dirname, filename)

def source_path(dirname: str, filename: str) -> str:
    if not os.path.isdir(SOURCE_DIR):
        raise FileNotFoundError(f"Source run directory not found: {SOURCE_DIR}")
    return os.path.join(SOURCE_DIR, dirname, filename)
