"""
Likelihood parameters loading script

Loads parameters written in a json file for likelihoods used in 
Bayesian modelling. Reads json produced by params_computation.py.
"""

import json
from utils.paths import source_path

def load_likelihood_params(filename: str = 'final_likelihood_params.json') -> dict:
    path = source_path(filename)
    with open(path, 'r') as f:
        params = json.load(f)

    return params

