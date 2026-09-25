import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.paths import source_path, data_path

SCENARIOS = {
    'S1': 1.5,
    'S2': 4,
    'S3': 10
}

LIGAND_TYPES = ['ag', 'nag', 'bg']

CLASS_LABELS = {
    'ag': 'Agonist',
    'nag': 'Near-agonist',
    'bg': 'Background'
}


def format_ratio(ratio):
    return f'{ratio:g}'


def plot_representative_histograms():
    datasets = {}

    for scenario, ratio in SCENARIOS.items():
        datasets[scenario] = pd.read_csv(
            source_path(
                f'kinetics/ratios/ratio {format_ratio(ratio)}',
                'var_samples.csv'
            )
        )

    # Use the same bin edges down each ligand-class column.
    bin_edges = {}

    for ligand_type in LIGAND_TYPES:
        pooled_positive = pd.concat([
            datasets[scenario].loc[
                datasets[scenario]['ligand_type'] == ligand_type,
                'time_weighted_variance'
            ]
            for scenario in SCENARIOS
        ]).dropna()

        pooled_positive = pooled_positive[pooled_positive > 0]

        bin_edges[ligand_type] = np.histogram_bin_edges(
            pooled_positive,
            bins=18
        )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(11, 8),
        sharex='col'
    )

    for row, (scenario, ratio) in enumerate(SCENARIOS.items()):
        df = datasets[scenario]

        for column, ligand_type in enumerate(LIGAND_TYPES):
            ax = axes[row, column]

            values = df.loc[
                df['ligand_type'] == ligand_type,
                'time_weighted_variance'
            ].dropna()

            positive_values = values[values > 0]
            zero_fraction = (values == 0).mean()

            ax.hist(
                positive_values,
                bins=bin_edges[ligand_type],
                density=True,
                color='#4c78a8',
                alpha=0.75,
                edgecolor='white',
                linewidth=0.6
            )

            if row == 0:
                ax.set_title(CLASS_LABELS[ligand_type])

            if column == 0:
                ax.set_ylabel(
                    rf'{scenario}, $r={ratio:g}$' + '\nDensity'
                )

            if row == 2:
                ax.set_xlabel('Positive time-weighted variance')

            ax.text(
                0.97,
                0.93,
                rf'$\widehat{{P}}(V=0)={zero_fraction:.3f}$',
                transform=ax.transAxes,
                ha='right',
                va='top',
                fontsize=8
            )

    fig.suptitle(
        'Empirical distributions of positive time-weighted variance',
        y=1.01
    )

    fig.tight_layout()

    fig.savefig(
        data_path(
            'kinetics/representative_distributions',
            'variance_histograms.pdf'
        ),
        bbox_inches='tight'
    )

    plt.close(fig)


COLORS = {'ag': '#d62728', 'nag': '#ff7f0e', 'bg': '#1f77b4'}

FEATURE_GROUPS = {
    'bound_readouts': [
        ('time_weighted_mean_b', 'Time-weighted mean B'),
        ('time_weighted_variance', 'Time-weighted variance B'),
        ('total_bound_time', 'Integrated bound time')
    ],
    'event_signal_readouts': [
        ('binding_unbinding_events', 'Binding/unbinding events'),
        ('mean_bound_dwell_time', 'Mean bound dwell time'),
        ('signal_rate', 'Signal rate')
    ]
}


def plot_readout_distributions():
    for filename, features in FEATURE_GROUPS.items():
        fig, axes = plt.subplots(
            3,
            3,
            figsize=(12, 8),
            sharey=True
        )

        for row, (scenario, ratio) in enumerate(SCENARIOS.items()):
            df = pd.read_csv(
                source_path(
                    f'kinetics/ratios/ratio {ratio}',
                    'var_samples.csv'
                )
            )

            for column, (feature, title) in enumerate(features):
                ax = axes[row, column]

                for ligand_type in LIGAND_TYPES:
                    values = (
                        df.loc[df['ligand_type'] == ligand_type, feature]
                        .dropna()
                        .sort_values()
                        .to_numpy()
                    )

                    probabilities = (
                        np.arange(1, len(values) + 1) / len(values)
                    )

                    ax.step(
                        values,
                        probabilities,
                        where='post',
                        label=ligand_type,
                        color=COLORS[ligand_type],
                        linewidth=1.4
                    )

                if row == 0:
                    ax.set_title(title)

                if column == 0:
                    ax.set_ylabel(f'{scenario}, r={ratio}\nECDF')

                if row == 2:
                    ax.set_xlabel(title)

        axes[0, -1].legend()
        fig.tight_layout()
        fig.savefig(
            data_path(
                'kinetics/readout_distributions',
                f'{filename}.pdf'
            ),
            bbox_inches='tight'
        )
        plt.close(fig)


if __name__ == '__main__':
    plot_representative_histograms()
    plot_readout_distributions()