import numpy as np
from scipy import stats
from gillespie_contact import gillespie_contact, compute_variance_rate
from seq_generator import generate_contaminated_sequence, add_measurement_noise, generate_apc_sequence

CLASS = ['ag', 'nag', 'bg']
T = 1000
n_sims = 1000

# From Huang et al. 2010, Table 1, 37°C (2D kinetics)
k_off = {'ag': 10.8, 'nag': 1.3, 'bg': 50.0}  # s^(-1)
k_on  = {'ag': 1.2e-2, 'nag': 2.7e-5, 'bg': 1e-6}  # μm^4s^(-1)

L_max = {'ag': 5, 'nag': 2, 'bg': 500} # treated as lambda for poisson sampling
R_max_vals = [50]

params = {
    'ag': {'family': 'loglogistic', 'shape': 1.53, 'scale': 0.000801},
    'bg': {'family': 'gamma', 'shape': 10.54, 'scale': 3.32*1e-8},
    'nag': {'family': 'gamma', 'shape': 0.897, 'scale': 1.92*1e-6, 'pi0': 0.226}
}

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

def test_ideal(n_sims):
    for t_type in CLASS:
        class_v = [compute_variance_rate(gillespie_contact(k_off, k_on, R_max_vals[0], L_max, t_type, T))
                for _ in range(n_sims)]
        for d in [0.1, 0.05, 0.2]:
            initial_belief = danger_prior(d)
            log_belief = {cls: np.log(p) for cls, p in initial_belief.items()}

            print(f"\nDANGER: {d} \t true_class={t_type} \t initial belief: {initial_belief}")

            convergence_step = None
            for i, v in enumerate(class_v, start = 1):
                log_belief = bayes_update_log(log_belief, v)
                belief = {cls: np.exp(lp) for cls, lp in log_belief.items()}
                
                if convergence_step is None and belief[t_type] > 0.95:
                    convergence_step = i

            print(f"true_class={t_type}\ndanger={d}\nconverged (>95% confidence) at contact {convergence_step}")

def test_noisy(true_class, contamination_rate, n_contacts, k_off, k_on, R_max, L_max, T, other_classes,
                drop_threshold=0.5, recovery_threshold=0.95):
    np.random.seed(42)

    seq_v, seq_labels = generate_contaminated_sequence(true_class, contamination_rate, n_contacts, 
                                                       k_off, k_on, R_max, L_max, T, other_classes)
    
    seq_v_noisy = [add_measurement_noise(v) for v in seq_v]

    for d in [0.1, 0.05, 0.2]:
        log_belief = {cls: np.log(p) for cls, p in danger_prior(d).items()}

        first_drop_step = None
        recovery_step = None
        contamination_steps = [i for i, lbl in enumerate(seq_labels, start=1) if lbl != true_class]

        for i, v in enumerate(seq_v_noisy, start=1):
            log_belief = bayes_update_log(log_belief, v)
            belief = {cls: np.exp(lp) for cls, lp in log_belief.items()}

            if first_drop_step is None and belief[true_class] < drop_threshold:
                first_drop_step = i
            if first_drop_step is not None and recovery_step is None and belief[true_class] > recovery_threshold:
                recovery_step = i

        final_belief_val = np.exp(log_belief[true_class])
        
        print(f"\ndanger={d} | true_class={true_class} | contamination_rate={contamination_rate}")
        print(f"  contaminating contacts occurred at steps: {contamination_steps[:10]}"
              f"{'...' if len(contamination_steps) > 10 else ''}")
        print(f"  belief first dropped below {drop_threshold} at step: {first_drop_step}")
        print(f"  belief recovered above {recovery_threshold} at step: {recovery_step}")
        print(f"  final belief[{true_class}] = {final_belief_val:.6g}  (raw log = {log_belief[true_class]:.2f})")

def test_noisy_apc(n_contacts, n_agonist, k_off, k_on, R_max, L_max, 
                    T, drop_threshold=0.5, recovery_threshold=0.95):
    np.random.seed(42)

    seq_v, seq_labels = generate_apc_sequence(n_contacts, n_agonist, k_off, k_on, R_max, L_max, T)
    
    seq_v_noisy = [add_measurement_noise(v) for v in seq_v]

    for d in [0.1, 0.05, 0.2]:
        log_belief = {cls: np.log(p) for cls, p in danger_prior(d).items()}

        first_drop_step = None
        recovery_step = None
        contamination_steps = [i for i, lbl in enumerate(seq_labels, start=1) if lbl != 'ag']

        for i, v in enumerate(seq_v_noisy, start=1):
            log_belief = bayes_update_log(log_belief, v)
            belief = {cls: np.exp(lp) for cls, lp in log_belief.items()}

            if first_drop_step is None and belief['ag'] < drop_threshold:
                first_drop_step = i
            if first_drop_step is not None and recovery_step is None and belief['ag'] > recovery_threshold:
                recovery_step = i

        final_belief_val = np.exp(log_belief['ag'])
        
        print(f"\ndanger={d} | true_class={'ag'}")
        print(f"  contaminating contacts occurred at steps: {contamination_steps[:10]}"
              f"{'...' if len(contamination_steps) > 10 else ''}")
        print(f"  belief first dropped below {drop_threshold} at step: {first_drop_step}")
        print(f"  belief recovered above {recovery_threshold} at step: {recovery_step}")
        print(f"  final belief[{'ag'}] = {final_belief_val:.6g}  (raw log = {log_belief['ag']:.2f})")
        
#test_noisy('bg', 0.05, 200, k_off, k_on, R_max_vals[0], L_max, T, ['nag'])
#test_noisy('ag', 0.5, 300, k_off, k_on, R_max_vals[0], L_max, T, ['bg', 'nag'])
test_noisy_apc(200, 24, k_off, k_on, R_max_vals[0], L_max, T)