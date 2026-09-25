import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from time import perf_counter

from utils.params import CLASS
from utils.paths import data_path, source_path

RATIOS = [1.5, 4, 10]
N_SPLITS = 10
TRAIN_FRACTION = 0.7

DISTRIBUTIONS = {
    'gamma': (scipy_stats.gamma, {'floc': 0}),
    'lognormal': (scipy_stats.lognorm, {'floc': 0}),
    'weibull': (scipy_stats.weibull_min, {'floc': 0}),
    'loglogistic': (scipy_stats.fisk, {'floc': 0})
}

def fit_one_split(samples, ratio, class_name, split_seed) -> dict:
    samples = np.array(samples, dtype=float)
    samples_pos = samples[samples>0]

    if len(samples_pos) < 10:
        raise ValueError(f'Too few positive values for {class_name}, r={ratio}')

    rng = np.random.default_rng(split_seed)
    shuffled = rng.permutation(samples_pos)

    split_index = int(TRAIN_FRACTION*len(shuffled))
    train = shuffled[:split_index]
    test = shuffled[split_index:]

    records = []

    for family, (distribution, fit_kwargs) in DISTRIBUTIONS.items():
        try:
            start = perf_counter()
            params = distribution.fit(train, **fit_kwargs)
            print(
                f'r={ratio}, class={class_name}, '
                f'split={split_seed}, family={family}, '
                f'time={perf_counter() - start:.2f}s'
            )
            fitted_distribution = distribution(*params)

            ks_result = scipy_stats.ks_1samp(test, fitted_distribution.cdf)
            ks_stat, p_value = ks_result.statistic, ks_result.pvalue
            
            log_density = fitted_distribution.logpdf(test)
            finite_log_density = np.maximum(log_density, np.log(np.finfo(float).tiny))

            records.append({
                'ratio': ratio,
                'ligand_type': class_name,
                'split_seed': split_seed,
                'family': family,
                'n': len(samples),
                'n_zero': np.sum(samples == 0),
                'zero_fraction': np.mean(samples == 0),
                'n_train': len(train),
                'n_test': len(test),
                'test_mean_loglik': finite_log_density.mean(),
                'ks_statistic': ks_stat,
                'ks_p_value': p_value,
                'parameters': repr(tuple(params))
            })

        except Exception as error:
            print(
                f'Fit failed: r={ratio}, class={class_name}, '
                f'family={family}: {error}'
            )

    return records

def run_distribution_sweep() -> pd.DataFrame:
    records = []

    for ratio in RATIOS:
        df = pd.read_csv(source_path(f'kinetics/ratios/ratio {format_ratio(ratio)}', 'var_samples.csv'))

        for ligand_type in CLASS:
            samples = df.loc[df['ligand_type'] == ligand_type, 'time_weighted_variance'].to_numpy()

            for split_seed in range(N_SPLITS):
                records.extend(fit_one_split(samples, ratio, ligand_type, split_seed))

    return pd.DataFrame(records)

def summarize_fits(results: pd.DataFrame):
    summary = (results
                .groupby(['ratio', 'ligand_type', 'family'])
                .agg(
                    mean_test_loglik=('test_mean_loglik', 'mean'),
                    sd_test_loglik=('test_mean_loglik', 'std'),
                    mean_ks=('ks_statistic', 'mean'),
                    max_ks=('ks_statistic', 'max'),
                    mean_ks_p=('ks_p_value', 'mean'),
                    zero_fraction=('zero_fraction', 'first'),
                    n=('n', 'first')
                ).reset_index()
               )
    groups = summary.groupby(['ratio', 'ligand_type'])

    summary['loglik_rank'] = groups['mean_test_loglik'].rank(ascending=False)
    summary['ks_rank'] = groups['mean_ks'].rank(ascending=True)
    summary['combined_rank'] = (summary['loglik_rank'] + summary['ks_rank'])

    return summary.sort_values(['ratio', 'ligand_type', 'combined_rank'])

def select_best_fits(summary: pd.DataFrame):
    return (summary
            .sort_values(['ratio', 'ligand_type', 'combined_rank'])
            .groupby(['ratio', 'ligand_type'], as_index=False)
            .first()
            )

def fit_final_models(selected: pd.DataFrame):
    rows = []
    for _, selection in selected.iterrows():
        ratio = selection['ratio']
        ligand_type = selection['ligand_type']
        family = selection['family']

        df = pd.read_csv(source_path(f'kinetics/ratios/ratio {format_ratio(ratio)}', 'var_samples.csv'))

        samples = df.loc[df['ligand_type']==ligand_type, 'time_weighted_variance'].to_numpy(dtype=float)

        samples_pos = samples[samples > 0]
        distribution, fit_kwargs = DISTRIBUTIONS[family]
        params = distribution.fit(samples_pos, **fit_kwargs)

        rows.append({
            'ratio': ratio,
            'ligand_type': ligand_type,
            'family': family,
            'zero_probability': np.mean(samples == 0),
            'parameters': repr(tuple(params))
        })
    return pd.DataFrame(rows)

def plot_fitted_cdfs(final_models: pd.DataFrame):
    for ratio in RATIOS:
        df = pd.read_csv(
            source_path(
                f'kinetics/ratios/ratio {format_ratio(ratio)}',
                'var_samples.csv'
            )
        )

        fig, axes = plt.subplots(1, len(CLASS), figsize=(12, 3.5))

        ratio_models = final_models[final_models['ratio'] == ratio]

        for ax, ligand_type in zip(axes, CLASS):
            samples = df.loc[df['ligand_type'] == ligand_type, 'time_weighted_variance'].to_numpy(dtype=float)
            samples_pos = np.sort(samples[samples > 0])

            empirical_cdf = np.arange(1, len(samples_pos) + 1) / len(samples_pos)

            model = ratio_models[ratio_models['ligand_type'] == ligand_type].iloc[0]

            family = model['family']
            distribution, fit_kwargs = DISTRIBUTIONS[family]
            params = distribution.fit(samples_pos, **fit_kwargs)

            ax.step(samples_pos, empirical_cdf, where='post', label='Empirical')
            ax.plot(samples_pos, distribution.cdf(samples_pos, *params), label=f'Fitted {family}')

            ax.set_title(ligand_type)
            ax.set_xlabel('Time-weighted variance')
            ax.set_ylabel('Cumulative probability')
            ax.legend(fontsize=8)


        fig.suptitle(
            f'Positive variance likelihood fits, r={ratio}'
        )
        fig.tight_layout()

        fig.savefig(
            data_path(
                f'distribution_test/ratios/ratio {format_ratio(ratio)}',
                'fitted_cdfs.png'
            ),
            dpi=200,
            bbox_inches='tight'
        )
        plt.close(fig)

def format_ratio(ratio):
    return f'{ratio:g}'

if __name__=='__main__':
    results = run_distribution_sweep()
    results.to_csv(data_path('distribution_test', 'variance_split_results.csv'), index=False)

    summary = summarize_fits(results)
    summary.to_csv(data_path('distribution_test', 'variance_fit_summary.csv'), index=False)

    selected = select_best_fits(summary)
    selected.to_csv(data_path('distribution_test', 'selected_variance_families.csv'), index=False)

    final_models = fit_final_models(selected)
    final_models.to_csv(data_path('distribution_test', 'final_variance_models.csv'), index=False)
    plot_fitted_cdfs(final_models)

    print(selected.to_string(index=False))
    print('\nFINAL MODELS')
    print(final_models.to_string(index=False))