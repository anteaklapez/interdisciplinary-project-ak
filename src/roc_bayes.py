import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from utils.paths import data_path
from gillespie_contact import gillespie_contact, compute_variance_rate
from seq_generator import generate_apc_sequence, add_measurement_noise
from tcell_model_base import danger_prior, bayes_update_log, K_OFF, K_ON, R_MAX, L_MAX, T_CONTACT

N_SEQ_PER_CLASS = 500     # number of ag-sequences and non-ag-sequences each
N_CONTACTS = 200          # length of each sequence
N_AGONIST_IF_POSITIVE = 24
DANGER_PRIOR_D = 0.1


def run_one_sequence(is_ag_sequence: bool):
    n_agonist = N_AGONIST_IF_POSITIVE if is_ag_sequence else 0
    seq_v, seq_label = generate_apc_sequence(N_CONTACTS, n_agonist, K_OFF, K_ON, R_MAX[0], L_MAX, T_CONTACT)

    seq_v_noisy = [add_measurement_noise(v) for v in seq_v]
    log_belief = {cls: np.log(p) for cls, p in danger_prior(DANGER_PRIOR_D).items()}

    trajectory = np.empty(N_CONTACTS)
    for i,v in enumerate(seq_v_noisy):
        log_belief = bayes_update_log(log_belief, v)
        trajectory[i] = np.exp(log_belief['ag'])
    
    return trajectory

def collect_trajectories():
    records_score = np.empty((2 * N_SEQ_PER_CLASS, N_CONTACTS))
    records_label = np.empty(2 * N_SEQ_PER_CLASS, dtype=bool)

    idx = 0
    for is_ag in [True, False]:
        for _ in range(N_SEQ_PER_CLASS): 
            records_score[idx] = run_one_sequence(is_ag)
            records_label[idx] = is_ag
            idx += 1

    return records_score, records_label

def auc_per_step(scores: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    n_steps = scores.shape[1]
    results = []
    for step in range(n_steps):
        step_scores = scores[:, step]
        try:
            auc = roc_auc_score(labels, step_scores)
        except ValueError:
            auc = np.nan

        results.append({'step': step + 1, 'auc': auc})
    
    return pd.DataFrame(results)

def roc_at_step(scores: np.ndarray, labels: np.ndarray, step: int) -> pd.DataFrame:
    fpr, tpr, thresholds = roc_curve(labels, scores[:, step - 1])
    return pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})

if __name__ == '__main__':
    np.random.seed(42)
    scores, labels = collect_trajectories()

    auc_df = auc_per_step(scores, labels)
    out_path_auc = data_path('bayes_auc_per_step.csv')
    auc_df.to_csv(out_path_auc, index=False)

    for step in [1, 5, 20, 50, 100, 200]:
        roc_df = roc_at_step(scores, labels, step)
        out_path_roc = data_path(f'bayes_roc_step_{step}.csv')

        roc_df.to_csv(out_path_roc, index=False)

    print(auc_df.tail(10))







