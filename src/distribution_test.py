import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gillespie_contact import gillespie_contact, compute_variance_rate
import os
from scipy import stats as scipy_stats


CLASS = ['ag', 'nag', 'bg']
T = 1000

# From Huang et al. 2010, Table 1, 37°C (2D kinetics)
k_off = {'ag': 10.8, 'nag': 1.3, 'bg': 50.0}  # s^(-1)
k_on  = {'ag': 1.2e-2, 'nag': 2.7e-5, 'bg': 1e-6}  # μm^4s^(-1)

L_max = {'ag': 5, 'nag': 2, 'bg': 500} # treated as lambda for poisson sampling
R_max_vals = [50]


event_counts = [len(gillespie_contact(k_off, k_on, 50, L_max, 'nag')) for _ in range(T)]
print(np.unique(event_counts, return_counts=True))
results = []
n_sims = 1000
for R_max in R_max_vals:
    for t_type in CLASS:
        for _ in range(n_sims):
            B_trajecory = gillespie_contact(k_off, k_on, R_max, L_max, t_type, T)
            results.append((t_type, R_max, compute_variance_rate(B_trajecory)))

df_results = pd.DataFrame(results, columns=['ligand_type', 'r_max', 'var_rate'])

stats = df_results.groupby('ligand_type')['var_rate'].agg(
    mean='mean', std='std',
    ci_low=lambda x: np.percentile(x, 2.5),
    ci_high=lambda x: np.percentile(x, 97.5),
    count='count',
    skew = lambda x: scipy_stats.skew(x),
    kurtosis = lambda x: scipy_stats.kurtosis(x),
    min='min',
    max='max'
).reset_index()

print(stats)

# GOODNESS OF FIT

DISTR_FAMILY = ['gamma', 'lognorm', 'weibull', 'paretto', 'loglogistic']

def fit_and_test(samples, class_name):
    samples = np.array(samples, dtype=float)
    samples_pos = samples[samples > 0]
    n_dropped = len(samples) - len(samples_pos)

    result = {'class': class_name, 'n': len(samples), 'n_zero_dropped': n_dropped, 'n_unique': len(np.unique(samples_pos))}

    if len(samples_pos) < 5:
        for fam in DISTR_FAMILY:
            result[f'{fam}_ks'], result[f'{fam}_p'] = np.nan, np.nan
        return result
    
    fits = {
        'gamma': (scipy_stats.gamma, dict(floc=0)),
        'lognorm': (scipy_stats.lognorm, dict(floc=0)),
        'weibull': (scipy_stats.weibull_min, dict(floc=0)),
        'loglogistic': (scipy_stats.fisk, dict(floc=0)),
    }

    for name, (dist, fit_kwargs) in fits.items():
        try:
            params = dist.fit(samples_pos, **fit_kwargs)
            ks, p = scipy_stats.kstest(samples_pos, dist.name, args=params)
            result[f'{name}_params'] = params
            result[f'{name}_ks'], result[f'{name}_p'] = ks, p
        except Exception as e: 
            result[f'{name}_ks'], result[f'{name}_p'] = np.nan, np.nan
        
    try:
        params_p = scipy_stats.pareto.fit(samples_pos, floc=0)
        ks_p, p_p = scipy_stats.kstest(samples_pos, 'pareto', args=params_p)
        result['pareto_params'] = params
        result['pareto_ks'], result['pareto_p'] = ks_p, p_p
    except Exception as e: 
        result['pareto_ks'], result['pareto_p'] = np.nan, np.nan

    return result


records = []
n_seeds = 30
for seed in range(n_seeds):
    rng_state = np.random.default_rng(seed)
    np.random.seed(seed)  # gillespie_contact uses np.random internally
    for t_type in CLASS:
        vals = [compute_variance_rate(gillespie_contact(k_off, k_on, R_max, L_max, t_type, T))
                for _ in range(n_sims)]
        r = fit_and_test(vals, t_type)
        r.update(seed=seed, class_=t_type)
        records.append(r)

df_seeds = pd.DataFrame(records)
out_path = os.path.join(os.path.dirname(__file__), '../data/runs/likelihood_v_4/seed_sweep_1000.csv')
df_seeds.to_csv(out_path, index=False)

summary = df_seeds.groupby('class_')[['gamma_p','lognorm_p', 'weibull_p', 'pareto_p', 'loglogistic_p','n_zero_dropped']].agg(['min','max','mean'])
print(summary)

# QQ-PLOT: ag vs fitted Gamma (due to heavy tail!)
ag_vals = df_results[df_results['ligand_type'] == 'ag']['var_rate'].values
ag_vals_pos = ag_vals[ag_vals > 0]

shape_ag, loc_ag, scale_ag = scipy_stats.gamma.fit(ag_vals_pos, floc=0)

fig_qq, ax_qq = plt.subplots(figsize=(6, 6))
scipy_stats.probplot(ag_vals_pos, dist=scipy_stats.gamma,
                      sparams=(shape_ag, loc_ag, scale_ag), plot=ax_qq)
ax_qq.set_title(f'QQ-Plot: ag vs Fitted Gamma (shape={shape_ag:.3f}, scale={scale_ag:.5f})')
ax_qq.get_lines()[0].set_markersize(3)
ax_qq.get_lines()[0].set_alpha(0.5)

plt.tight_layout()
qq_path = os.path.join(os.path.dirname(__file__), '../data/runs/likelihood_v_4/ag_gamma_qqplot.png')
plt.savefig(qq_path, dpi=150)
plt.show()