import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gillespie_contact import gillespie_contact, compute_variance_rate
import os
from scipy import stats as scipy_stats
import json


CLASS = ['ag', 'nag', 'bg']
T = 1000

# From Huang et al. 2010, Table 1, 37°C (2D kinetics)
k_off = {'ag': 10.8, 'nag': 1.3, 'bg': 50.0}  # s^(-1)
k_on  = {'ag': 1.2e-2, 'nag': 2.7e-5, 'bg': 1e-6}  # μm^4s^(-1)

L_max = {'ag': 5, 'nag': 2, 'bg': 500} # treated as lambda for poisson sampling
R_max = 50

N_POOLED = 20000 
POOL_SEED = 999  

np.random.seed(POOL_SEED)

final_params = {}

for t_type in CLASS:
    vals = np.array([compute_variance_rate(gillespie_contact(k_off, k_on, R_max, L_max, t_type, T)) for _ in range(N_POOLED)])
    
    vals_pos = vals[vals > 0]
    n_zero = len(vals) - len(vals_pos)
    pi0 = n_zero/len(vals)

    if t_type == 'ag':
        shape_ll, loc_ll, scale_ll = scipy_stats.fisk.fit(vals_pos, floc=0)
        ks, p = scipy_stats.kstest(vals_pos, 'fisk', args=(shape_ll, loc_ll, scale_ll))
        final_params[t_type] = {'family': 'loglogistic', 'shape': shape_ll, 'loc': loc_ll, 'scale': scale_ll,
                                'ks': ks, 'p': p, 'n': len(vals), 'n_zero_dropped': n_zero, 'pi0': pi0}
            
    else:
        shape_g, loc_g, scale_g = scipy_stats.gamma.fit(vals_pos, floc=0)
        ks, p = scipy_stats.kstest(vals_pos, 'gamma', args=(shape_g, loc_g, scale_g))
        final_params[t_type] = {'family': 'gamma', 'shape': shape_g, 'loc': loc_g, 'scale': scale_g,
                                'ks': ks, 'p': p, 'n': len(vals), 'n_zero_dropped': n_zero, 'pi0': pi0}

out_path = os.path.join(os.path.dirname(__file__), '../data/runs/likelihood_v_4/final_likelihood_params.json')
with open(out_path, 'w') as f:
    json.dump(final_params, f, indent=2, default=float)