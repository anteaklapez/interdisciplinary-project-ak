import matplotlib.pyplot as plt
import numpy as np

from simulations.gillespie_contact import gillespie_contact
from simulations.kinetics_computation import compute_koff, compute_kon
from utils.params import L_MAX, R_MAX, T_CONTACT
from utils.paths import data_path, source_path

SCENARIOS = {'S1': 1.5, 'S2': 4, 'S3': 10}
LIGANDS = ['ag', 'nag', 'bg']
STATE_VARIABLES = [
    (4, r'$B(t)$'),
    (3, r'$R_A(t)$'),
    (5, r'$S(t)$')
]
N_TRAJECTORIES = 3

def plot_raw_trajectories():
    np.random.seed(42)
    k_on = compute_kon()

    for scenario, ratio in SCENARIOS.items():
        k_off = compute_koff(1, ratio)

        fig, axes = plt.subplots(3, 3, figsize=(11, 7), sharex=True, sharey='row')

        for column, ligand_type in enumerate(LIGANDS):
            _, state_trajectory, _ = gillespie_contact(
                    k_off,
                    k_on,
                    R_MAX[0],
                    L_MAX,
                    ligand_type,
                    T_CONTACT
                )

            states = np.asarray([state[:6] for state in state_trajectory], dtype=float)

            for row, (state_index, ylabel) in enumerate(
                    STATE_VARIABLES
                ):
                    axes[row, column].step(
                        states[:, 0],
                        states[:, state_index],
                        where='post',
                        alpha=0.7,
                        linewidth=1
                    )

                    if column == 0:
                        axes[row, column].set_ylabel(ylabel)

            axes[0, column].set_title(ligand_type)
            axes[-1, column].set_xlabel('Time (s)')

        fig.suptitle(
            f'{scenario}: off-rate ratio r = {ratio}'
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(
            data_path(
                'kinetics/raw_trajectories',
                f'{scenario}_trajectories.pdf'
            ),
            bbox_inches='tight'
        )
        plt.close(fig)

if __name__ == '__main__':
    plot_raw_trajectories()