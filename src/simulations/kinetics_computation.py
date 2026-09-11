from utils.params import CLASS, R_MAX, L_MAX, T_CONTACT

def compute_koff(tau_ag: float = 1, ratio: float = 1):
    k_off_ag = 1/tau_ag
    k_off_nag = k_off_ag*ratio
    k_off_bg = 1/0.033

    k_off = {'ag': k_off_ag, 'nag': k_off_nag, 'bg': k_off_bg}
    return k_off

def compute_kon():
    return {'ag': 0.01, 'nag': 0.009, 'bg': 1e-6}
    

