import matplotlib.pyplot as plt
import numpy as np

from simulations.gillespie_contact import gillespie_contact
from simulations.kinetics_computation import compute_koff, compute_kon
from utils.params import L_MAX, R_MAX, T_CONTACT
from utils.paths import data_path

RATIO = 4
LIGAND_TYPE = 'ag'
SEED = 20261020
WINDOW = 20.0  # contact window (seconds)


def cropped_contact():
    """
    Columns: 0=time, 1=L, 2=R_I, 3=R_A, 4=B, 5=S.
    """
    np.random.seed(SEED)
    k_on = compute_kon()
    k_off = compute_koff(1, RATIO)
    _, state_trajectory, _ = gillespie_contact(
        k_off, k_on, R_MAX[0], L_MAX, LIGAND_TYPE, T_CONTACT
    )
    states = np.asarray([state[:6] for state in state_trajectory], dtype=float)
    times = states[:, 0]

    idx = np.searchsorted(times, WINDOW, side='left')
    t = np.append(times[:idx], WINDOW)
    state = np.vstack([states[:idx], states[idx - 1]])  # last state held to WINDOW
    return t, state


def plot_mean():
    t, state = cropped_contact()
    b = state[:, 4]
    dt = np.diff(t)
    mean = np.sum(b[:-1] * dt) / WINDOW
    mean_unweighted = np.mean(b[:-1])

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.step(t, b, where='post', color='#4c78a8', lw=1.5, label=r'$B(t)$')
    ax.fill_between(t, b, step='post', color='#4c78a8', alpha=0.2,
                label=r'$A_B$')
    ax.axhline(mean, color='#d62728', ls='--', lw=1.5,
           label=r'$\langle B\rangle$' + f'={mean:.2f}')
    ax.axhline(mean_unweighted, color='#9467bd', ls=':', lw=1.5,
           label=f'unweighted mean={mean_unweighted:.2f}')
    ax.legend(loc='upper right')
    
    i = int(np.argmax(dt))
    min_width = 0.04 * WINDOW  # skips too narrow ones

    annotate = [
        j for j in (i - 1, i, i + 1)
        if 0 <= j < len(dt) and dt[j] >= min_width
    ]

    for j in annotate:
        ax.plot([t[j], t[j]], [+0.35, b[j] + 0.4],
                color='gray', ls=':', lw=0.8)
        ax.plot([t[j + 1], t[j + 1]], [+0.35, b[j] + 0.4],
                color='gray', ls=':', lw=0.8)
        ax.annotate('', xy=(t[j + 1], +0.35), xytext=(t[j], +0.35),
                    arrowprops=dict(arrowstyle='<->', color='black'))
        ax.text((t[j] + t[j + 1]) / 2, +0.45,
                rf'$\Delta t_{{{j + 1}}}$', ha='center', fontsize=10)


    ax.set_ylim(-0.01, b.max() + 1.2)
    ax.set_xlim(0, WINDOW)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$B(t)$')
    ax.set_title(f'Time-weighted mean $\\langle B\\rangle$ within a 20-second contact window')
    fig.savefig(data_path('kinetics/readout_extraction', 'readout_mean.png'),
                bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_variance():
    t, state = cropped_contact()
    b = state[:, 4]
    dt = np.diff(t)
    mean = np.sum(b[:-1] * dt) / WINDOW
    sq = (b - mean) ** 2
    var = np.sum(sq[:-1] * dt) / WINDOW

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0, color='gray', lw=0.5)
    ax.step(t, sq, where='post', color='#d62728', lw=1.5,
        label=r'$(B(t)-\langle B\rangle)^2$')
    ax.fill_between(t, sq, step='post', color='#d62728', alpha=0.2,
                label=r'$\sum_i (B_i-\langle B\rangle)^2\,\Delta t_i$')
    ax.legend(loc='upper right')
    ax.set_ylim(0, sq.max() * 1.25)
    ax.set_xlim(0, WINDOW)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$(B(t)-\langle B\rangle)^2$')
    ax.set_title(rf'Time-weighted variance within a 20-second contact window, $\mathrm{{Var}}_T(B) = {var:.2f}$')
    fig.savefig(data_path('kinetics/readout_extraction', 'readout_variance.png'),
                bbox_inches='tight', dpi=200)
    plt.legend()
    plt.close(fig)

def plot_signal():
    t, state = cropped_contact() 
    s = state[:, 5]  
    rate = s[-2] / WINDOW

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(t, s, where='post', color='#2ca02c', lw=1.5, label=r'$S(t)$')
    ax.plot([0, WINDOW], [0, s[-2]], color='black', ls='--', lw=1.2,
            label=f'slope = $S(T)/T$ = {rate:.2f}')
    ax.set_xlim(0, WINDOW)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$S(t)$')
    ax.set_title(f'Signal Rate = {rate:.2f}, 20-second contact window')
    ax.legend(loc='upper left')
    fig.savefig(data_path('kinetics/readout_extraction', 'readout_signal.png'),
                bbox_inches='tight', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    plot_mean()
    plot_variance()
    plot_signal()