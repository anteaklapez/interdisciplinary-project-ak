"""
Loading seed sweep script

Script for loading seed sweep produced by distribution_test.py for 
reusability across evaluation scripts.
"""

import pandas as pd
from utils.paths import source_path

def load_seed_sweep(filename: str = 'seed_sweep_1000.py')-> pd.DataFrame:
    path = source_path(filename)
    return pd.read_csv(path)