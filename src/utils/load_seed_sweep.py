"""
Loading seed sweep script

Script for loading seed sweep produced by distribution_test.py for 
reusability across evaluation scripts.
"""

import pandas as pd
from paths import source_path

def load_seed_sweep(dirname: str = 'likelihood_v_4',filename: str = 'seed_sweep_1000.py')-> pd.DataFrame:
    path = source_path(dirname, filename)
    return pd.read_csv(path)