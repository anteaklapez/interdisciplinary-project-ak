import numpy as np

def gillespie_contact(k_off: dict, k_on: dict, R_max, L_max, t_type, T_contact=500):
    B_trajectory = []
    L = L_max[t_type]       # free pMHC ligands (TODO: Use poisson instead of one value)
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