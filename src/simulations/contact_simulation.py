import numpy as np
import pandas as pd
from simulations.gillespie_contact import gillespie_contact, compute_variance_rate
from scipy import stats as scipy_stats
from utils.paths import data_path, source_path
from utils.params import CLASS, T_CONTACT, K_OFF, K_ON, L_MAX, R_MAX, KINETICS_RATIOS
from generators.seq_generator import add_measurement_noise
from simulations.kinetics_computation import compute_koff, compute_kon

N_SIMS = 500

def run_simulations(n_sims: int = N_SIMS, **kwargs) -> pd.DataFrame:
    results = []
    for R_max in R_MAX:
        for t_type in CLASS:
            for _ in range(n_sims):
                B_trajectory = gillespie_contact(kwargs['K_OFF'], kwargs['K_ON'], R_max, L_MAX, t_type, T_CONTACT)
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

def assign_scenario(ratio):
    if ratio < 3:
        return 'S1'
    elif ratio >=3 and ratio < 10:
        return 'S2'
    else:
        return 'S3'

def aggregate_scenarios():
    scenario_frames = {'S1': [], 'S2': [], 'S3': []}

    for ratio in KINETICS_RATIOS:
        df = pd.read_csv(source_path(f'kinetics/ratios/ratio {ratio}', 'var_rate_samples.csv'))
        df['ratio'] = ratio
        scenario_frames[assign_scenario(ratio)].append(df)

    for scenario, frames in scenario_frames.items():
        pooled = pd.concat(frames, ignore_index=True)
        summary = summarize(pooled)

        pooled.to_csv(data_path(f'kinetics/{scenario}', 'var_rate_samples.csv'), index=False)
        summary.to_csv(data_path(f'kinetics/{scenario}', 'summary_stats.csv'), index=False)


if __name__ == '__main__':
    np.random.seed(42)
    
    k_on = compute_kon()

    for ratio in KINETICS_RATIOS:
        print(f'KINETIC OFF RATE RATIO = {ratio}')
        k_off = compute_koff(1, ratio)
        df_results = run_simulations(N_SIMS, K_ON = k_on, K_OFF = k_off)

        df_results['var_rate'] = [add_measurement_noise(v) for v in df_results['var_rate']]
        df_results.to_csv(data_path(f'kinetics/ratios/ratio {ratio}', 'var_rate_samples.csv'), index=False)

        stats_df = summarize(df_results)
        stats_df.to_csv(data_path(f'kinetics/ratios/ratio {ratio}', 'var_rate_summary_stat.csv'), index=False)

        print(stats_df)
        print()

        ks_stat, p_value = report_ks_ag_vs_nag(df_results)
        print(f"KS statistic (ag vs nag): {ks_stat:.3f}, p-value: {p_value:.4f}")

        print()

    aggregate_scenarios()

    

