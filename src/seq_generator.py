import numpy as np
from scipy import stats
from gillespie_contact import gillespie_contact, compute_variance_rate

def generate_contaminated_sequence(t_type, contamination_rate, n_contacts, 
                                   k_off: dict, k_on: dict, R_max, L_max, 
                                   T, other_classes):
    seq = []
    labels = []
    for _ in range(n_contacts):
        if np.random.uniform() < contamination_rate:
            actual = np.random.choice(other_classes)
        else:
            actual = t_type

        v = compute_variance_rate(gillespie_contact(k_off, k_on, R_max, L_max, actual, T))
        seq.append(v)
        labels.append(actual)

    return seq, labels

def add_measurement_noise(v, noise_frac=0.15):
    if v == 0:
        return 0.0
    
    noise = np.random.normal(1.0, noise_frac)
    return max(v * noise, 0.0)

def generate_apc_sequence(n_contacts, n_agonist, k_off, k_on, R_max, L_max, T,
                            self_classes=['bg', 'nag'], self_weights=[0.3, 0.7]):
    
    labels = ['ag'] * n_agonist + list(np.random.choice(self_classes, n_contacts-n_agonist, p = self_weights))
    np.random.shuffle(labels)

    seq = [compute_variance_rate(gillespie_contact(k_off, k_on, R_max, L_max, lbl, T)) for lbl in labels]

    return seq, labels
