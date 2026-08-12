import numpy as np
from scipy import stats
from gillespie_contact import gillespie_contact, compute_variance_rate
from seq_generator import generate_contaminated_sequence, add_measurement_noise, generate_apc_sequence
from utils.params import CLASS, T_CONTACT, K_OFF, K_ON, R_MAX, L_MAX
from utils.load_likelihood_params import load_likelihood_params

N_SIMS = 1000

params = load_likelihood_params()

def likelihood(v, cls, floor=1e-8):
    p = params[cls]

    if cls == 'nag':
        if v==0:
            return p['pi0']
        else:
            return (1-p['pi0']) * stats.gamma.pdf(v, p['shape'], scale = p['scale'])
        
    if cls == 'ag':
        return stats.fisk.pdf(v, p['shape'], scale = p['scale']) + floor
    
    return stats.gamma.pdf(v, p['shape'], scale = p['scale']) + floor

def bayes_update_log(log_prior, v):
    log_unnorm = {cls: np.log(likelihood(v, cls) + 1e-300) + log_prior[cls] for cls in log_prior}
    log_total = np.logaddexp.reduce(list(log_unnorm.values()))
    return {cls: log_unnorm[cls] - log_total for cls in log_unnorm}

def danger_prior(d, nag_share=0.7):
    p_ag = d
    remaining = 1 - p_ag
    p_nag = nag_share * remaining
    p_bg = remaining * (1-nag_share)
    return {'ag': p_ag, 'nag': p_nag, 'bg': p_bg}

def run_belief_trajectory(values, initial_belief: dict) -> tuple:
    log_belief = {cls: np.log(p) for cls, p in initial_belief.items()}
    belief_trace = []

    for v in values:
        log_belief = bayes_update_log(log_belief, v)
        belief_trace.append({cls: np.exp(lp) for cls, lp in log_belief.items()})
    
    return log_belief, belief_trace


def test_ideal(n_sims:int, danger_levels = (0.1, 0.05, 0.2)):
    for t_type in CLASS:
        class_v = [compute_variance_rate(gillespie_contact(K_OFF, K_ON, R_MAX[0], L_MAX, t_type, T_CONTACT))
                for _ in range(n_sims)]
        for d in danger_levels:
            initial_belief = danger_prior(d)
            log_belief, belief_trace = run_belief_trajectory(class_v, initial_belief)

            print(f"\nDANGER: {d} \t true_class={t_type} \t initial belief: {initial_belief}")

            convergence_step = next((i for i, b in enumerate(belief_trace, start=1) if b[t_type] > 0.95), None)

            print(f"true_class={t_type}\ndanger={d}\nconverged (>95% confidence) at contact {convergence_step}")

def _run_contamination_report(seq_v, seq_labels, true_class: str, danger_levels, 
                              drop_threshold: float, recovery_threshold: float, 
                              label: str):
    for d in danger_levels:
        initial_belief = danger_prior(d)
        log_belief, belief_trace = run_belief_trajectory(seq_v, initial_belief)

        contamination_steps = [i for i,lbl in enumerate(seq_labels, start=1) if lbl != true_class]

        first_drop_step = next((i for i, b in enumerate(belief_trace, start=1) if b[true_class] < drop_threshold), None)
        
        recovery_step = None

        if first_drop_step is not None:
            recovery_step = next((i for i, b in enumerate(belief_trace, start=1) 
                                  if i > first_drop_step and b[true_class] > recovery_threshold), None)
        
        final_belief_val = np.exp(log_belief[true_class])

        print(f"\ndanger={d} | true_class={true_class}{label}")
        print(f"  contaminating contacts occurred at steps: {contamination_steps[:10]}"
              f"{'...' if len(contamination_steps) > 10 else ''}")
        print(f"  belief first dropped below {drop_threshold} at step: {first_drop_step}")
        print(f"  belief recovered above {recovery_threshold} at step: {recovery_step}")
        print(f"  final belief[{true_class}] = {final_belief_val:.6g}  (raw log = {log_belief[true_class]:.2f})")


def test_noisy(true_class, contamination_rate, n_contacts, other_classes, danger_levels=(0.1, 0.05, 0.2),
                drop_threshold=0.5, recovery_threshold=0.95):
    np.random.seed(42)

    seq_v, seq_labels = generate_contaminated_sequence(true_class, contamination_rate, n_contacts, 
                                                       K_OFF, K_ON, R_MAX[0], L_MAX, T_CONTACT, 
                                                       other_classes)
    
    seq_v_noisy = [add_measurement_noise(v) for v in seq_v]

    _run_contamination_report(seq_v_noisy, seq_labels, true_class, danger_levels, drop_threshold, 
                              recovery_threshold, label=f" | contamination_rate={contamination_rate}")

def test_noisy_apc(n_contacts, n_agonist, danger_levels=(0.1, 0.05, 0.2), 
                   drop_threshold=0.5, recovery_threshold=0.95):
    np.random.seed(42)

    seq_v, seq_labels = generate_apc_sequence(n_contacts, n_agonist, K_OFF, K_ON, R_MAX, L_MAX, T_CONTACT)
    
    seq_v_noisy = [add_measurement_noise(v) for v in seq_v]

    _run_contamination_report(seq_v_noisy, seq_labels, 'ag', danger_levels, drop_threshold, 
                              recovery_threshold, label='')
        
if __name__ == '__main__':
    # test_noisy('bg', 0.05, 200, ['nag'])
    # test_noisy('ag', 0.5, 300, ['bg', 'nag'])
    test_noisy_apc(200, 24)