import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simulations.gillespie_contact import gillespie_contact, compute_variance_rate
from scipy import stats as scipy_stats

from utils.params import CLASS, T_CONTACT, K_OFF, K_ON, L_MAX, R_MAX
from utils.paths import data_path, source_path

N_SIMS = 1000
N_SEEDS = 30

DISTR_FAMILY = {'gamma': (scipy_stats.gamma, dict(floc=0)),
                'lognorm': (scipy_stats.lognorm, dict(floc=0)), 
                'weibull': (scipy_stats.weibull_min, dict(floc=0)), 
                'pareto': (scipy_stats.pareto, dict(floc=0)), 
                'loglogistic': (scipy_stats.fisk, dict(floc=0))}

def fit_and_test(samples, class_name) -> dict:
    samples = np.array(samples, dtype=float)
    samples_pos = samples[samples > 0]
    n_dropped = len(samples) - len(samples_pos)

    result = {'class': class_name, 'n': len(samples), 'n_zero_dropped': n_dropped, 'n_unique': len(np.unique(samples_pos))}

    if len(samples_pos) < 5:
        for fam in DISTR_FAMILY:
            result[f'{fam}_ks'], result[f'{fam}_p'] = np.nan, np.nan
        return result

    for name, (dist, fit_kwargs) in DISTR_FAMILY.items():
        try:
            params = dist.fit(samples_pos, **fit_kwargs)
            ks, p = scipy_stats.kstest(samples_pos, dist.name, args=params)
            result[f'{name}_params'] = params
            result[f'{name}_ks'], result[f'{name}_p'] = ks, p
        except Exception as e: 
            result[f'{name}_ks'], result[f'{name}_p'] = np.nan, np.nan

    return result


def run_seed_sweep(n_seeds: int = N_SEEDS, n_sims: int = N_SIMS) -> pd.DataFrame:

    records = []
    for seed in range(n_seeds):
        np.random.seed(seed)  # gillespie_contact uses np.random internally
        for t_type in CLASS:
            vals = [compute_variance_rate(gillespie_contact(K_OFF, K_ON, R_MAX[0], L_MAX, t_type, T_CONTACT))
                    for _ in range(n_sims)]
            r = fit_and_test(vals, t_type)
            r.update(seed=seed, class_=t_type)
            records.append(r)

    return pd.DataFrame(records)

def summarize_seed_sweep(df_seeds: pd.DataFrame) -> pd.DataFrame:
    p_cols = [f'{fam}_p' for fam in DISTR_FAMILY]
    return df_seeds.groupby('class_')[p_cols + ['n_zero_dropped']].agg(['min', 'max', 'mean'])

def plot_ag_gamma_qq(df_results: pd.DataFrame):
    ag_vals = df_results[df_results['ligand_type'] == 'ag']['var_rate'].values
    ag_vals_pos = ag_vals[ag_vals > 0]

    shape_ag, loc_ag, scale_ag = scipy_stats.gamma.fit(ag_vals_pos, floc=0)

    fig, ax = plt.subplots(figsize=(6, 6))
    scipy_stats.probplot(ag_vals_pos, dist=scipy_stats.gamma,
                        sparams=(shape_ag, loc_ag, scale_ag), plot=ax)
    ax.set_title(f'QQ-Plot: ag vs Fitted Gamma (shape={shape_ag:.3f}, scale={scale_ag:.5f})')
    ax.get_lines()[0].set_markersize(3)
    ax.get_lines()[0].set_alpha(0.5)

    plt.tight_layout()
    plt.savefig(data_path('ag_gamma_qqplot.png'), dpi=150)

    return fig


if __name__=='__main__':
    df_seeds = run_seed_sweep()
    df_seeds.to_csv(data_path('seed_sweep_1000.csv'), index=False)

    summary = summarize_seed_sweep(df_seeds)
    summary.to_csv(data_path('summary_seed_sweep.csv'), index=False)

    df_results = pd.read_csv(source_path('var_rate_samples.csv'))
    plot_ag_gamma_qq(df_results)

    plt.show()