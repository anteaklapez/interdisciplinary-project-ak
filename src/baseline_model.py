import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from seq_generator import generate_apc_sequence, add_measurement_noise
from utils.params import CLASS, T_CONTACT, K_OFF, K_ON, R_MAX, L_MAX
from utils.paths import data_path, source_path

N_SEQ_PER_CLASS = 500
N_CONTACTS = 200
N_AGONIST_IF_POSITIVE = 24
TAU_CANDIDATES = np.logspace(-7, -3, 25)
CALIB_WINDOW = 5
ROLLING_WINDOW = 5

def run_one_sequence(is_ag_sequence: bool):
    n_agonist = N_AGONIST_IF_POSITIVE if is_ag_sequence else 0
    seq_v, seq_labels = generate_apc_sequence(N_CONTACTS, n_agonist, K_OFF, K_ON, R_MAX[0], L_MAX, T_CONTACT)

    seq_v_noisy = np.array([add_measurement_noise(v) for v in seq_v])

    return seq_v_noisy

def collect_sequences(n_seq_per_class: int = N_SEQ_PER_CLASS, seed = None):
    if seed is not None:
        np.random.seed(seed)
    
    all_seq = np.empty((2 * n_seq_per_class, N_CONTACTS))
    labels = np.empty(2 * n_seq_per_class, dtype=bool)
    idx = 0
    for is_ag in [True, False]:
        for _ in range(n_seq_per_class):
            all_seq[idx] = run_one_sequence(is_ag)
            labels[idx] = is_ag
            idx += 1

    return all_seq, labels

def local_rolling_volatility(sequences: np.ndarray, window: int = ROLLING_WINDOW) -> np.ndarray:
    n_seqs, n_contacts = sequences.shape
    vol = np.empty_like(sequences)

    for i in range(n_contacts):
        start = max(0, i - window + 1)
        vol[:, i] = sequences[:, start:i +1].std(axis=1)

    return vol

def calibrate_tau(calib_sequences: np.ndarray, calib_labels: np.ndarray) -> float:
    step1_vals = calib_sequences[:, 0]
    best_auc, best_tau = -1, None

    for tau in TAU_CANDIDATES:
        flagged = (step1_vals > tau).astype(float)
        try:
            auc = roc_auc_score(calib_labels, flagged)
        except ValueError:
            continue

        if auc > best_auc:
            best_auc, best_tau = auc, tau

    return best_tau

def _youden_j(labels: np.ndarray, flagged: np.ndarray) -> float:
    tpr = np.sum(labels & flagged) / max(np.sum(labels), 1)
    fpr = np.sum((~labels) & flagged) / max(np.sum(~labels), 1)

    return tpr - fpr

def calibrate_two_phase_tau(calib_sequences: np.ndarray, calib_labels: np.ndarray, window: int = CALIB_WINDOW):
    vals = calib_sequences[:, :window].mean(axis=1)

    best_tpr, tau1 = -1, None
    for tau in TAU_CANDIDATES:
        flagged = vals > tau
        tpr = np.sum(calib_labels & flagged) / max(np.sum(calib_labels), 1)
        fpr = np.sum((~calib_labels) & flagged) / max(np.sum(~calib_labels), 1)

        if fpr <= 0.3 and tpr > best_tpr:
            best_tpr, tau1 = tpr, tau

    passed_phase1 = vals > tau1
    best_j, tau2 = -1, None
    for tau in TAU_CANDIDATES:
        if tau <= tau1:
            continue

        flagged = passed_phase1 & (vals > tau)
        j = _youden_j(calib_labels, flagged)
        if j > best_j:
            best_j, tau2 = j, tau
    
    return tau1, tau2


def running_flag_fraction(sequences: np.ndarray, tau1: float, tau2: float) -> pd.DataFrame:
    passed_phase1 = sequences > tau1
    passed_both = passed_phase1 & (sequences > tau2)

    flags = passed_both.astype(float)
    cum_flags = np.cumsum(flags, axis=1)
    n_so_far = np.arange(1, sequences.shape[1] + 1)    
    return cum_flags / n_so_far

def auc_per_step(sequences: np.ndarray, labels: np.ndarray, tau1: float, tau2: float) -> pd.DataFrame:
    running_score = running_flag_fraction(sequences, tau1, tau2)
    n_steps = sequences.shape[1]
    aucs = np.empty(n_steps)

    for step in range(n_steps):
        try:
            aucs[step] = roc_auc_score(labels, running_score[:, step])
        except ValueError:
            aucs[step] = np.nan

    return pd.DataFrame({'step': np.arange(1, n_steps + 1), 'auc': aucs})

if __name__ == '__main__':
    calib_sequences, calib_labels = collect_sequences(n_seq_per_class=100, seed=1)
    tau1, tau2 = calibrate_two_phase_tau(calib_sequences, calib_labels)
    print(f"Calibrated tau1 (screen, Phase 1) = {tau1:.6g}")
    print(f"Calibrated tau2 (confirm, Phase 2) = {tau2:.6g}")

    eval_sequences, eval_labels = collect_sequences(n_seq_per_class=N_SEQ_PER_CLASS, seed=42)
    result_df = auc_per_step(eval_sequences, eval_labels, tau1, tau2)

    result_df.to_csv(data_path('baseline_sequential_auc_per_step.csv'), index=False)

    print(result_df.head(15))
    print(result_df.tail(10))

