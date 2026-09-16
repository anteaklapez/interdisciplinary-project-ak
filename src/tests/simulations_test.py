import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from simulations.gillespie_contact import (
    gillespie_contact,
    compute_timeweighted_variance
)
from simulations.kinetics_computation import compute_koff, compute_kon
from utils.params import T_CONTACT, L_MAX, R_MAX
from utils.paths import data_path


TEST_RATIOS = [1.5, 4, 10]
LIGAND_TYPES = ['ag', 'nag', 'bg']
N_CONTACTS = 10
MAX_WAITS_PER_CONTACT = 1000
R_TEST = R_MAX[0]
SEED = 42

STATE_COLUMNS = ['L', 'R_I', 'R_A', 'B', 'S']

EVENT_CHANGES = {
    'bind_inactive': [-1, -1, 0, 1, 0],
    'unbind_activate': [1, 0, 1, -1, 0],
    'bind_active': [-1, 0, -1, 1, 0],
    'revert': [0, 1, -1, 0, 0],
    'signal': [0, 0, 0, 0, 1],
    'end': [0, 0, 0, 0, 0]
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def trajectory_dataframe(state_trajectory):
    return pd.DataFrame(
        state_trajectory,
        columns=['time', 'L', 'R_I', 'R_A', 'B', 'S', 'event']
    )


def calculate_trajectory_metrics(B_trajectory):
    trajectory = np.asarray(B_trajectory, dtype=float)
    times = trajectory[:, 0]
    values = trajectory[:, 1]

    dt = np.diff(np.concatenate(([0.0], times)))
    total_bound_time = np.sum(values * dt)
    mean_B = total_bound_time / times[-1]

    variance_B = np.sum(
        (values - mean_B) ** 2 * dt
    ) / times[-1]

    return total_bound_time, mean_B, variance_B


def validate_contact(
    ratio,
    ligand_type,
    simulation_id,
    B_trajectory,
    state_trajectory,
    waiting_times,
    k_off
):
    state_df = trajectory_dataframe(state_trajectory)
    state_values = state_df[STATE_COLUMNS].to_numpy()

    label = (
        f'ratio={ratio}, ligand={ligand_type}, '
        f'simulation={simulation_id}'
    )

    require(
        state_df.iloc[0]['event'] == 'initial',
        f'Missing initial state: {label}'
    )
    require(
        state_df.iloc[-1]['event'] == 'end',
        f'Missing terminal state: {label}'
    )
    require(
        np.isclose(state_df.iloc[-1]['time'], T_CONTACT),
        f'Incorrect terminal time: {label}'
    )
    require(
        np.all(np.diff(state_df['time']) >= 0),
        f'Non-monotonic simulation time: {label}'
    )
    require(
        np.all(state_values >= 0),
        f'Negative state found: {label}'
    )

    tcr_total = state_df['R_I'] + state_df['R_A'] + state_df['B']
    require(
        np.all(tcr_total == R_TEST),
        f'TCR conservation failure: {label}'
    )

    initial_ligand_total = (
        state_df.iloc[0]['L'] + state_df.iloc[0]['B']
    )
    ligand_total = state_df['L'] + state_df['B']

    require(
        np.all(ligand_total == initial_ligand_total),
        f'Ligand conservation failure: {label}'
    )
    require(
        np.all(np.diff(state_df['S']) >= 0),
        f'Signal decreased: {label}'
    )

    state_changes = state_df[STATE_COLUMNS].diff()

    for row_index in range(1, len(state_df)):
        event = state_df.iloc[row_index]['event']

        require(
            event in EVENT_CHANGES,
            f'Unknown event "{event}": {label}'
        )

        observed_change = (
            state_changes.iloc[row_index]
            .to_numpy(dtype=float)
        )
        expected_change = np.asarray(
            EVENT_CHANGES[event],
            dtype=float
        )

        require(
            np.array_equal(observed_change, expected_change),
            (
                f'Incorrect update for event "{event}": {label}; '
                f'observed={observed_change}, '
                f'expected={expected_change}'
            )
        )

    event_counts = state_df['event'].value_counts()

    n_bindings = (
        event_counts.get('bind_inactive', 0)
        + event_counts.get('bind_active', 0)
    )
    n_unbindings = event_counts.get('unbind_activate', 0)
    n_signals = event_counts.get('signal', 0)

    require(
        n_bindings - n_unbindings == state_df.iloc[-1]['B'],
        f'Binding-event balance failure: {label}'
    )
    require(
        n_signals == state_df.iloc[-1]['S'],
        f'Signal-event balance failure: {label}'
    )

    B_array = np.asarray(B_trajectory, dtype=float)

    require(
        len(B_array) == len(state_df) - 1,
        f'B-trajectory length mismatch: {label}'
    )
    require(
        np.allclose(B_array[:, 0], state_df['time'].iloc[1:]),
        f'B-trajectory times do not match state times: {label}'
    )
    require(
        np.array_equal(B_array[:, 1], state_df['B'].iloc[:-1]),
        f'B-trajectory values are misaligned: {label}'
    )

    total_bound_time, mean_B, independent_variance = (
        calculate_trajectory_metrics(B_trajectory)
    )

    implemented_variance = compute_timeweighted_variance(
        B_trajectory
    )

    require(
        np.isclose(
            implemented_variance,
            independent_variance,
            rtol=1e-12,
            atol=1e-12
        ),
        f'Time-weighted variance mismatch: {label}'
    )

    row = {
        'ratio': ratio,
        'ligand_type': ligand_type,
        'simulation_id': simulation_id,
        'n_events': len(state_df) - 2,
        'time_weighted_mean_b': mean_B,
        'time_weighted_variance': implemented_variance,
        'total_bound_time': total_bound_time,
        'total_signal': state_df.iloc[-1]['S'],
        'expected_mean_dwell': 1 / k_off[ligand_type]
    }

    for event in EVENT_CHANGES:
        if event != 'end':
            row[event] = event_counts.get(event, 0)

    normalized_waits = [
        normalized_dt
        for _, _, normalized_dt
        in waiting_times[:MAX_WAITS_PER_CONTACT]
    ]

    return row, normalized_waits


def run_validation():
    np.random.seed(SEED)

    k_on = compute_kon()
    contact_rows = []
    normalized_waits = []

    for ratio in TEST_RATIOS:
        k_off = compute_koff(1, ratio=ratio)

        for ligand_type in LIGAND_TYPES:
            for simulation_id in range(N_CONTACTS):
                outputs = gillespie_contact(
                    k_off=k_off,
                    k_on=k_on,
                    R_max=R_TEST,
                    L_max=L_MAX,
                    t_type=ligand_type,
                    T_contact=T_CONTACT
                )

                row, contact_waits = validate_contact(
                    ratio,
                    ligand_type,
                    simulation_id,
                    *outputs,
                    k_off
                )

                contact_rows.append(row)
                normalized_waits.extend(contact_waits)

    contacts_df = pd.DataFrame(contact_rows)

    event_columns = [
        'bind_inactive',
        'unbind_activate',
        'bind_active',
        'revert',
        'signal'
    ]

    missing_events = [
        event
        for event in event_columns
        if contacts_df[event].sum() == 0
    ]

    require(
        not missing_events,
        f'Reactions never observed: {missing_events}'
    )

    contacts_df.to_csv(
        data_path(
            'simulation_tests',
            'validation_contacts.csv'
        ),
        index=False
    )

    return contacts_df, np.asarray(normalized_waits)


def summarize_events(contacts_df):
    event_columns = [
        'bind_inactive',
        'unbind_activate',
        'bind_active',
        'revert',
        'signal'
    ]

    summary_df = (
        contacts_df
        .groupby(['ratio', 'ligand_type'])[event_columns]
        .mean()
        .reset_index()
    )

    summary_df.to_csv(
        data_path(
            'simulation_tests',
            'event_count_summary.csv'
        ),
        index=False
    )

    return summary_df


def validate_dwell_times(contacts_df):
    rows = []

    for (ratio, ligand_type), group in contacts_df.groupby(
        ['ratio', 'ligand_type']
    ):
        n_unbindings = group['unbind_activate'].sum()
        total_bound_time = group['total_bound_time'].sum()
        expected_dwell = group['expected_mean_dwell'].iloc[0]

        empirical_dwell = (
            total_bound_time / n_unbindings
            if n_unbindings > 0
            else np.nan
        )

        relative_error = (
            abs(empirical_dwell - expected_dwell)
            / expected_dwell
        )

        rows.append({
            'ratio': ratio,
            'ligand_type': ligand_type,
            'n_unbindings': n_unbindings,
            'expected_mean_dwell': expected_dwell,
            'empirical_mean_dwell': empirical_dwell,
            'relative_error': relative_error,
            'within_25_percent': relative_error <= 0.25
        })

    dwell_df = pd.DataFrame(rows)

    dwell_df.to_csv(
        data_path(
            'simulation_tests',
            'dwell_time_validation.csv'
        ),
        index=False
    )

    return dwell_df


def validate_waiting_times(normalized_waits):
    require(
        len(normalized_waits) > 0,
        'No waiting times were recorded.'
    )

    ks_statistic, p_value = scipy_stats.kstest(
        normalized_waits,
        'expon'
    )

    mean_wait = normalized_waits.mean()
    mean_tolerance = max(
        0.05,
        4 / np.sqrt(len(normalized_waits))
    )

    summary_df = pd.DataFrame([{
        'count': len(normalized_waits),
        'mean': mean_wait,
        'std': normalized_waits.std(ddof=1),
        'median': np.median(normalized_waits),
        'ks_statistic': ks_statistic,
        'p_value': p_value,
        'mean_tolerance': mean_tolerance,
        'mean_check_passed':
            abs(mean_wait - 1.0) <= mean_tolerance
    }])

    summary_df.to_csv(
        data_path(
            'simulation_tests',
            'waiting_time_summary.csv'
        ),
        index=False
    )

    upper_limit = np.percentile(normalized_waits, 99.5)
    x_values = np.linspace(0, upper_limit, 400)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.hist(
        normalized_waits,
        bins=50,
        range=(0, upper_limit),
        density=True,
        alpha=0.6,
        label='Simulated'
    )
    ax.plot(
        x_values,
        scipy_stats.expon.pdf(x_values),
        linewidth=2,
        label='Expected Exp(1)'
    )

    ax.set_xlabel(r'Normalized waiting time $a_0\Delta t$')
    ax.set_ylabel('Density')
    ax.set_title('Gillespie waiting-time validation')
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        data_path(
            'simulation_tests',
            'waiting_time_validation.png'
        ),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    return summary_df


if __name__ == '__main__':
    contacts_df, normalized_waits = run_validation()

    event_summary = summarize_events(contacts_df)
    dwell_summary = validate_dwell_times(contacts_df)
    waiting_summary = validate_waiting_times(normalized_waits)

    print('\nAll structural trajectory checks passed.')

    print('\nEVENT COUNTS')
    print(event_summary.to_string(index=False))

    print('\nDWELL-TIME CHECK')
    print(dwell_summary.to_string(index=False))

    print('\nWAITING-TIME CHECK')
    print(waiting_summary.to_string(index=False))