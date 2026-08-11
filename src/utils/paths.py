"""
Centralized system for file or directory creation

Every script imports DATA_DIR without needing to locally build each filepath
or manually create folders. RUN_VERSION can be only changed here instead of 
changing it multiple times within the code.
"""

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Current experiment/run version
RUN_VERSION = 'likelihood_v_4'

DATA_DIR = os.path.join(ROOT_DIR, 'data', 'runs', RUN_VERSION)

def data_path(filename: str) -> str:
    """Build an absolute path inside the current run's data folder,
    creating the folder if it does not exist yet."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, filename)