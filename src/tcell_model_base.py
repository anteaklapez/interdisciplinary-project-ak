import numpy as np
from scipy import stats
from gillespie_contact import gillespie_contact, compute_variance_rate

CLASS = ['ag', 'nag', 'bg']
T = 1000

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

def likelihood(v, cls):
    p = params[cls]

    if cls == 'nag':
        if v==0:
            return p['pi0']
        else:
            return (1-p['pi0']) * stats.gamma.pdf(v, p['shape'], scale = p['scale'])
        
    if cls == 'ag':
        return stats.fisk.pdf(v, p['shape'], scale = p['scale'])
    
    return stats.gamma.pdf(v, p['shape'], scale = p['scale'])

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

n_sims = 1000

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

