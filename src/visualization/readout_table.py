import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from utils.paths import source_path, data_path


SCENARIOS = {
    'S1': 1.5,
    'S2': 4,
    'S3': 10
}

READOUTS = {
    'time_weighted_variance':
        r'Time-weighted variance of $B(t)$',
    'binding_unbinding_events':
        'Binding/unbinding event count',
    'mean_bound_dwell_time':
        'Mean bound dwell time',
    'signal_rate':
        'Signal-production rate'
}


def summarize_values(values):
    values = values.dropna()

    return {
        'median': values.median(),
        'q1': values.quantile(0.25),
        'q3': values.quantile(0.75)
    }


def create_readout_table():
    rows = []

    for scenario, ratio in SCENARIOS.items():
        df = pd.read_csv(
            source_path(
                f'kinetics/ratios/ratio {ratio}',
                'var_samples.csv'
            )
        )

        for column, label in READOUTS.items():
            ag = df.loc[
                df['ligand_type'] == 'ag',
                column
            ].dropna()

            nag = df.loc[
                df['ligand_type'] == 'nag',
                column
            ].dropna()

            ag_stats = summarize_values(ag)
            nag_stats = summarize_values(nag)

            ks_statistic, p_value = scipy_stats.ks_2samp(
                ag,
                nag
            )

            rows.append({
                'scenario': scenario,
                'ratio': ratio,
                'readout': label,
                'ag_median': ag_stats['median'],
                'ag_q1': ag_stats['q1'],
                'ag_q3': ag_stats['q3'],
                'nag_median': nag_stats['median'],
                'nag_q1': nag_stats['q1'],
                'nag_q3': nag_stats['q3'],
                'ks_statistic': ks_statistic,
                'p_value': p_value
            })

    results = pd.DataFrame(rows)

    results.to_csv(
        data_path(
            'kinetics/readout_selection',
            'readout_table.csv'
        ),
        index=False
    )

    return results


def print_latex_rows(results):
    for _, row in results.iterrows():
        p_value = (
            r'$<0.0001$'
            if row['p_value'] < 0.0001
            else f"{row['p_value']:.4f}"
        )

        ag_summary = (
            f"{row['ag_median']:.4g} "
            f"[{row['ag_q1']:.4g}, {row['ag_q3']:.4g}]"
        )

        nag_summary = (
            f"{row['nag_median']:.4g} "
            f"[{row['nag_q1']:.4g}, {row['nag_q3']:.4g}]"
        )

        print(
            f"{row['scenario']} & "
            f"{row['ratio']:g} & "
            f"{row['readout']} & "
            f"{ag_summary} & "
            f"{nag_summary} & "
            f"{row['ks_statistic']:.3f} & "
            f"{p_value} \\\\"
        )


if __name__ == '__main__':
    table_df = create_readout_table()
    print_latex_rows(table_df)