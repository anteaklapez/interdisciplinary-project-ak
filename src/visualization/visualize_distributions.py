import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.paths import source_path, data_path

def plot_scenario_distributions():
    scenarios = ['S1', 'S2', 'S3']
    titles = [
        r'S1: $1 \leq r < 3$',
        r'S2: $3 \leq r < 10$',
        r'S3: $10 \leq r \leq 15$'
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 3.5),
        sharex=True,
        sharey=True
    )

    for ax, scenario, title in zip(axes, scenarios, titles):
        sweep_df = pd.read_csv(source_path(f'kinetics/{scenario}', 'var_samples.csv'))

        for ligand_type in ['ag', 'nag']:
            values = sweep_df.loc[
                sweep_df['ligand_type'] == ligand_type,
                'time_weighted_variance'
            ]

            ax.hist(
                values,
                bins=30,
                density=True,
                histtype='step',
                linewidth=1.5,
                label=ligand_type
            )

        ax.set_title(title)
        ax.set_xlabel('Time-weighted variance')
        ax.legend()

    axes[0].set_ylabel('Density')

    fig.tight_layout()
    fig.savefig(data_path('kinetics/scenario_distributions','scenario_distributions.pdf'),
        bbox_inches='tight'
    )
    plt.close(fig)



SCENARIOS = {'S1': 1.5, 'S2': 4, 'S3': 10}
LIGANDS = ['ag', 'nag', 'bg']
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

                for ligand_type in LIGANDS:
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
    plot_scenario_distributions()
    plot_readout_distributions()