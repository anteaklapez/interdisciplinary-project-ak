import pandas as pd
import matplotlib.pyplot as plt
from utils.paths import data_path, source_path
from sklearn.metrics import roc_auc_score
import os

BASELINE_PATH = os.path.join(os.path.dirname(__file__), '..','data', 'runs', 'baseline_noisy', 'baseline_sequential_auc_per_step.csv')
BAYES_PATH = os.path.join(os.path.dirname(__file__), '..','data', 'runs', 'bayes', 'bayes_auc_per_step.csv')

def plot_comparison(bayes_path=BAYES_PATH, baseline_path=BASELINE_PATH, out_path='bayes_vs_baseline_auc.png'):
    bayes_df = pd.read_csv(bayes_path)
    baseline_df = pd.read_csv(baseline_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(bayes_df['step'], bayes_df['auc'], color='#5b6ee1', linewidth=2, label='Bayesian model')
    ax.fill_between(bayes_df['step'], bayes_df['auc'], 0.4, color='#5b6ee1', alpha=0.12)

    ax.plot(baseline_df['step'], baseline_df['auc'], color='#e19b5b', linewidth=2,
             linestyle='--', label='Sequential baseline (fixed threshold)')

    ax.set_title(
        "Bayesian Model vs. Sequential Baseline: AUC by Contact Step\n"
        "Simulated ag vs. non-ag cell sequences",
        fontsize=13, loc='left'
    )
    ax.set_xlabel("Contact step")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(data_path(out_path), dpi=150)
    return out_path


if __name__ == '__main__':
    plot_comparison()