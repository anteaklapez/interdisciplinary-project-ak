import numpy as np

def gillespie_contact(k_off: dict, k_on: dict, R_max, L_max, t_type, T_contact=500):
    B_trajectory = []
    L = np.random.poisson(L_max[t_type])     # free pMHC ligands
    R_I = R_max             # TCR inactive
    R_A = 0                 # TCR active
    B = 0                   # TCR-pMHC complex
    t = 0                   # time
    K_OFF = k_off[t_type]
    K_ON = k_on[t_type]
    
    while t < T_contact:
        a_bind = L*R_I*K_ON
        a_unbind = B*K_OFF
        a_total = a_bind + a_unbind

        if a_total == 0:
            break

        dt = np.random.exponential(1 / a_total)
        t += dt
        B_trajectory.append((t, B))

        if np.random.uniform() < a_bind / a_total: # binding
            R_I -= 1
            B += 1
        else: # unbinding and activation
            B -= 1
            R_A += 1
            # TODO: R_A must become R_I again at some point

    return B_trajectory


def compute_variance_rate(trajectory):
    if len(trajectory) == 0:
        return 0.0
    times, B_values = zip(*trajectory)
    times = np.array(times)
    B_values = np.array(B_values, dtype=float)

    t_prev = np.concatenate(([0.0], times[:-1]))
    dt = times - t_prev
    T_actual = times[-1]

    B_mean = np.sum(B_values*dt) / T_actual
    B_var =np.sum(((B_values - B_mean) ** 2) * dt) / T_actual
    
    return B_var / T_actual