import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gillespie_contact import gillespie_contact
import os
from scipy import stats as scipy_stats


CLASS = ['ag', 'nag', 'bg']
T = 1000

# From Huang et al. 2010, Table 1, 37°C (2D kinetics)
k_off = {'ag': 10.8, 'nag': 1.3, 'bg': 50.0}  # s^(-1)
k_on  = {'ag': 1.2e-2, 'nag': 2.7e-5, 'bg': 1e-6}  # μm^4s^(-1)

L_max = {'ag': 5, 'nag': 2, 'bg': 500} # treated as lambda for poisson sampling
R_max_vals = [50]


def compute_variance_rate(trajectory, T_contact=T):
    if len(trajectory) == 0:
        return 0.0
    _, B_values = zip(*trajectory)
    return np.var(np.array(B_values, dtype=float)) / T_contact

event_counts = [len(gillespie_contact(k_off, k_on, 50, L_max, 'nag')) for _ in range(T)]
print(np.unique(event_counts, return_counts=True))
results = []
n_sims = 500
for R_max in R_max_vals:
    for t_type in CLASS:
        for _ in range(n_sims):
            B_trajecory = gillespie_contact(k_off, k_on, R_max, L_max, t_type, T)
            results.append((t_type, R_max, compute_variance_rate(B_trajecory)))

df_results = pd.DataFrame(results, columns=['ligand_type', 'r_max', 'var_rate'])

stats = df_results.groupby('ligand_type')['var_rate'].agg(
    mean='mean', std='std',
    ci_low=lambda x: np.percentile(x, 2.5),
    ci_high=lambda x: np.percentile(x, 97.5),
    count='count',
    skew = lambda x: scipy_stats.skew(x),
    min='min',
    max='max'
).reset_index()

# STATS PLOT
fig_stats, ax_stats = plt.subplots(figsize=(8, 5))

sns.boxplot(data=df_results, x='ligand_type', y='var_rate', ax=ax_stats, showfliers=False, boxprops=dict(alpha=0.3))
sns.stripplot(data=df_results, x='ligand_type', y='var_rate', ax=ax_stats, size=2, alpha=0.4, jitter=True)

ax_stats.set_title('Variance Rate: Boxplot + Individual Simulations')

plt.tight_layout()

data_path_stats = os.path.join(os.path.dirname(__file__), '../data/runs/likelihood_v_3/var_rate_boxplot.png')
plt.savefig(data_path_stats, dpi=150)

plt.show()


# VARIANCE RATE
fig1, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
for ax, t_type in zip(axes, CLASS):
    subset = df_results[df_results['ligand_type'] == t_type]
    sns.kdeplot(data=subset, x='var_rate', ax=ax, fill=True)
    ax.set_title(t_type)
    ax.set_xlabel('Variance Rate')

plt.tight_layout()

data_path1 = os.path.join(os.path.dirname(__file__), '../data/runs/likelihood_v_3/var_rate_distributions.png')
plt.savefig(data_path1, dpi=150)
plt.show()

# HISTOGRAM
fig2, ax2 = plt.subplots(figsize=(8, 5))
for t_type, group in df_results.groupby('ligand_type'):
    sns.histplot(group['var_rate'], alpha=0.5, label=t_type, bins=12, stat='density', ax=ax2)

ax2.set_title('Variance Rate Histogram by Ligand Class')
ax2.set_xlabel('Variance Rate')
ax2.legend()

plt.tight_layout()

data_path2 = os.path.join(os.path.dirname(__file__), '../data/runs/likelihood_v_3/var_rate_histogram.png')
plt.savefig(data_path2, dpi=150)
plt.show()

# OVERLAP AGONIST VS NEAR-AGONIST
subset_ag_nag = df_results[df_results['ligand_type'].isin(['ag', 'nag'])]
fig3, ax3 = plt.subplots(figsize=(8,5))
for t_type, group in subset_ag_nag.groupby('ligand_type'):
    sns.kdeplot(group['var_rate'], ax=ax3, label=t_type, fill=True)

ax3.set_title('Variance Rate Overlap: Agonist vs Near-agonist')
ax3.set_xlabel('Variance Rate')
ax3.legend()

plt.tight_layout()
data_path3 = os.path.join(os.path.dirname(__file__), '../data/runs/likelihood_v_3/var_rate_ag_nag_overlap.png')
plt.savefig(data_path3, dpi=150)
plt.show()

# KS - measures how well distributions ag and nag are completely separable
from scipy import stats as scipy_stats

ag_rates  = df_results[df_results['ligand_type'] == 'ag']['var_rate']
nag_rates = df_results[df_results['ligand_type'] == 'nag']['var_rate']

ks_stat, p_value = scipy_stats.ks_2samp(ag_rates, nag_rates)
print(f"KS statistic: {ks_stat:.3f}, p-value: {p_value:.4f}")