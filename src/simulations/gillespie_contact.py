import numpy as np

def gillespie_contact(k_off: dict, k_on: dict, R_max, L_max, t_type, T_contact=500):
    B_trajectory = []
    state_trajectory = []

    L = np.random.poisson(L_max[t_type])     # free pMHC ligands
    L_initial = L           # initial ligand count
    R_I = R_max             # TCR inactive
    R_A = 0                 # TCR active
    B = 0                   # TCR-pMHC complex
    t = 0                   # time
    S = 0                   # signal
    K_OFF = k_off[t_type]
    K_ON = k_on[t_type]
    waiting_times = []

    state_trajectory.append((t, L, R_I, R_A, B, S, 'initial'))
    
    while t < T_contact:
        a_bind_inactive = L*R_I*K_ON
        a_unbind = B*K_OFF
        a_bind_active = L*R_A*K_ON
        a_revert = R_A*K_OFF
        a_signal = R_A*B*K_OFF

        propensities = np.array([
            a_bind_inactive, a_unbind, a_bind_active, a_revert, a_signal
        ], dtype=float)

        a_total = propensities.sum()

        if a_total <= 0:
            B_trajectory.append((T_contact, B))
            state_trajectory.append((T_contact, L, R_I, R_A, B, S, 'end'))
            break

        dt = np.random.exponential(1 / a_total)

        if t+dt > T_contact:
            B_trajectory.append((T_contact, B))
            state_trajectory.append((T_contact, L, R_I, R_A, B, S, 'end'))
            break

        waiting_times.append((dt, a_total, a_total * dt))

        t += dt
        B_trajectory.append((t, B))

        threshold = np.random.uniform(0.0, a_total)
        cumulative = np.cumsum(propensities)

        if threshold < cumulative[0]:
            L -= 1
            R_I -= 1
            B += 1
            event = 'bind_inactive'
        elif threshold < cumulative[1]: 
            B -= 1
            L += 1
            R_A += 1
            event = 'unbind_activate'
        elif threshold < cumulative[2]: 
            L -= 1
            R_A -= 1
            B += 1
            event = 'bind_active'        
        elif threshold < cumulative[3]: 
            R_A -= 1
            R_I += 1
            event = 'revert'
        else:
            S += 1
            event = 'signal'

        if min(L, R_I, R_A, B, S) < 0:
            raise RuntimeError(f'Negative state after {event} at t={t:.6f}')

        if R_I + R_A + B != R_max:
            raise RuntimeError(f'TCR conservation violated after {event}')
        
        if L + B != L_initial:
            raise RuntimeError(f'Ligand conservation violated after {event} at t={t:.6f}')

        state_trajectory.append(
            (t, L, R_I, R_A, B, S, event)
        )

    return B_trajectory, state_trajectory, waiting_times


def compute_timeweighted_stats(trajectory):
    if len(trajectory) == 0:
        return 0.0, 0.0, 0.0    
    times, B_values = zip(*trajectory)
    times = np.array(times)
    B_values = np.array(B_values, dtype=float)

    t_prev = np.concatenate(([0.0], times[:-1]))
    dt = times - t_prev
    T_actual = times[-1]

    if T_actual <= 0:
        return 0.0, 0.0, 0.0
    total_bound_time = np.sum(B_values*dt)
    mean = total_bound_time / T_actual
    variance =np.sum(((B_values - mean) ** 2) * dt) / T_actual
    
    return mean, variance, total_bound_time

def compute_timeweighted_variance(trajectory):
    return compute_timeweighted_stats(trajectory)[1]

def summarize_contact(B_trajectory, state_trajectory):
    mean_B, variance_B, total_bound_time = compute_timeweighted_stats(B_trajectory)

    events = [state[6] for state in state_trajectory]
    n_binding = sum(event in {'bind_inactive', 'bind_active'} for event in events)
    n_unbinding = events.count('unbind_activate')

    mean_dwell_time = (total_bound_time/n_unbinding if n_unbinding>0 else np.nan)

    T_actual = B_trajectory[-1][0] if B_trajectory else 0.0
    total_signal = state_trajectory[-1][5]
    signal_rate = total_signal/T_actual if T_actual > 0 else 0.0

    return {
        'time_weighted_mean_b': mean_B,
        'time_weighted_variance': variance_B,
        'total_bound_time': total_bound_time,
        'n_binding_events': n_binding,
        'n_unbinding_events': n_unbinding,
        'binding_unbinding_events': n_binding + n_unbinding,
        'mean_bound_dwell_time': mean_dwell_time,
        'total_signal': total_signal,
        'signal_rate': signal_rate
    }