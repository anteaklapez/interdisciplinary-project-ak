import numpy as np
import pandas as pd
from simulations.gillespie_contact import gillespie_contact, compute_variance_rate
from scipy import stats as scipy_stats
from utils.paths import data_path
from utils.params import CLASS, T_CONTACT, K_OFF, K_ON, L_MAX, R_MAX
from generators.seq_generator import add_measurement_noise

N_SIMS = 500

def run_simulations(n_sims: int = N_SIMS, r_max_vals = R_MAX) -> pd.DataFrame:
    results = []
    for R_max in r_max_vals:
        for t_type in CLASS:
            for _ in range(n_sims):
                B_trajectory = gillespie_contact(K_OFF, K_ON, R_max, L_MAX, t_type, T_CONTACT)
                results.append((t_type, R_max, compute_variance_rate(B_trajectory)))
    return pd.DataFrame(results, columns=['ligand_type', 'r_max', 'var_rate'])

def summarize(df_results: pd.DataFrame):
    return df_results.groupby('ligand_type')['var_rate'].agg(
                mean='mean', std='std',
                ci_low=lambda x: np.percentile(x, 2.5),
                ci_high=lambda x: np.percentile(x, 97.5),
                count='count',
                skew = lambda x: scipy_stats.skew(x),
                min='min',
                max='max').reset_index()
    
def report_ks_ag_vs_nag(df_results: pd.DataFrame) -> tuple:
    ag_rates = df_results[df_results['ligand_type']=='ag']['var_rate']
    nag_rates = df_results[df_results['ligand_type']=='nag']['var_rate']
    ks_stat, p_value = scipy_stats.ks_2samp(ag_rates, nag_rates)

    return ks_stat, p_value


if __name__ == '__main__':
    np.random.seed(42)
    df_results = run_simulations(N_SIMS)
    df_results['var_rate'] = [add_measurement_noise(v) for v in df_results['var_rate']]
    df_results.to_csv(data_path('var_rate_samples.csv'), index=False)

    stats_df = summarize(df_results)
    stats_df.to_csv(data_path('var_rate_summary_stat.csv'), index=False)

    print(stats_df)

    ks_stat, p_value = report_ks_ag_vs_nag(df_results)
    print(f"KS statistic (ag vs nag): {ks_stat:.3f}, p-value: {p_value:.4f}")
