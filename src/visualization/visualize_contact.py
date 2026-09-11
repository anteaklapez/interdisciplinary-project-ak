import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from utils.paths import data_path
from utils.params import CLASS, KINETICS_RATIOS

def load_var_rate_samples(filename: str = 'var_rate_samples.csv') -> pd.DataFrame:
    return pd.read_csv(filename)

def plot_boxplot(df_results: pd.DataFrame, dirname: str):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.boxplot(data=df_results, x='ligand_type', y='var_rate', ax=ax, showfliers=False, boxprops=dict(alpha=0.3))
    sns.stripplot(data=df_results, x='ligand_type', y='var_rate', ax=ax, size=2, alpha=0.4, jitter=True)

    ax.set_title('Variance Rate: Boxplot + Individual Simulations')
    fig.tight_layout()
    fig.savefig(data_path(dirname,'var_rate_boxplot.png'), dpi=150)

    return fig

def plot_histograms(df_results: pd.DataFrame, dirname: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

    for ax, t_type in zip(axes, CLASS):
        subset = df_results[df_results['ligand_type'] == t_type]
        sns.histplot(data=subset, x='var_rate', ax=ax, bins=40, stat='density') 

        ax.set_title(t_type)
        ax.set_xlabel('Variance Rate')

    fig.tight_layout()
    fig.savefig(data_path(dirname,'var_rate_distributions.png'), dpi=150)

    return fig

def plot_log_histogram(df_results: pd.DataFrame, dirname: str):
    fig, ax = plt.subplots(figsize=(8,5))
    
    for t_type, group in df_results.groupby('ligand_type'):
        sns.histplot(group['var_rate'], alpha=0.5, label=t_type, bins=40, stat='density', ax=ax)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('Variance Rate Histogram by Ligand Class')
    ax.set_xlabel('Variance Rate')
    ax.legend()

    fig.tight_layout()
    fig.savefig(data_path(dirname,'var_rate_histogram.png'), dpi=150)

    return fig

def plot_ag_nag_ecdf(df_results: pd.DataFrame, dirname: str):
    subset_ag_nag = df_results[df_results['ligand_type'].isin(['ag', 'nag'])]
    fig, ax = plt.subplots(figsize=(8,5))

    for t_type, group in subset_ag_nag.groupby('ligand_type'):
        sns.ecdfplot(group['var_rate'], ax=ax, label=t_type)

    ax.set_title('Variance Rate ECDF: Agonist vs Near-agonist')
    ax.set_xlabel('Variance Rate')
    ax.legend()

    fig.tight_layout()
    fig.savefig(data_path(dirname,'var_rate_ag_nag_overlap.png'), dpi=150)


if __name__ == '__main__':
    df_results = load_var_rate_samples()
    for ratio in KINETICS_RATIOS:
        plot_boxplot(df_results, f'kinetics/ratio {ratio}')
        plot_histograms(df_results, f'kinetics/ratio {ratio}')
        plot_log_histogram(df_results, f'kinetics/ratio {ratio}')
        plot_ag_nag_ecdf(df_results, f'kinetics/ratio {ratio}')

    
