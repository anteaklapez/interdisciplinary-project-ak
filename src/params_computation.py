import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gillespie_contact import gillespie_contact, compute_variance_rate
import os
from scipy import stats as scipy_stats
import json
from utils.params import CLASS, T_CONTACT, K_OFF, K_ON, R_MAX, L_MAX
from utils.paths import data_path

N_POOLED = 20000 
POOL_SEED = 999
FAMILY_BY_CLASS = {'ag': 'loglogistic',
                   'nag': 'gamma',
                   'bg': 'gamma'}

def fit_class_params(t_type: str, n_pooled: int = N_POOLED) -> dict:
    vals = np.array([compute_variance_rate(gillespie_contact(K_OFF, K_ON, R_MAX[0], L_MAX, t_type, T_CONTACT)) 
                     for _ in range(n_pooled)])
    vals_pos = vals[vals > 0]
    n_zero = len(vals) - len(vals_pos)
    pi0 = n_zero/len(vals)

    family = FAMILY_BY_CLASS[t_type]
    dist = scipy_stats.fisk if family == 'loglogistic' else scipy_stats.gamma
    dist_name = 'fisk' if family == 'loglogistic' else 'gamma'

    shape, loc, scale = dist.fit(vals_pos, floc=0)
    ks, p = scipy_stats.kstest(vals_pos, dist_name, args=(shape, loc, scale))

    return {'family': family, 'shape': shape, 'loc': loc, 'scale': scale,
            'ks': ks, 'p': p, 'n': len(vals), 'n_zero_dropped': n_zero, 'pi0': pi0}


def compute_all_params(classes = CLASS) -> dict:
    np.random.seed(POOL_SEED)
    return {t_type: fit_class_params(t_type) for t_type in classes}

if __name__ == '__main__':
    final_params = compute_all_params()

    out_path = data_path('final_likelihood_params.json')
    with open(out_path, 'w') as f:
        json.dump(final_params, f, indent=2, default=float)

    print(f"Saved likelihood params to {out_path}")
    for cls, p in final_params.items():
        print(f"  {cls}: family={p['family']} shape={p['shape']:.4f} "
              f"scale={p['scale']:.6g} ks={p['ks']:.4f} p={p['p']:.4f} pi0={p['pi0']:.4f}")
