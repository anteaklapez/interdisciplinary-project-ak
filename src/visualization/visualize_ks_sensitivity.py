import matplotlib.pyplot as plt
import pandas as pd
from utils.paths import source_path, data_path
from utils.params import KINETICS_RATIOS
from simulations.contact_simulation import report_ks_ag_vs_nag

def compute_ks_curve():
    records = []
    for ratio in KINETICS_RATIOS:
        df = pd.read_csv(source_path(f'kinetics/ratios/ratio {ratio}', 'var_rate_samples.csv'))
        ks_stat, p_value = report_ks_ag_vs_nag(df)
        records.append({'ratio': ratio, 'ks_stat': ks_stat, 'p_val': p_value})

    return pd.DataFrame(records).sort_values('ratio')


def plot_ks_sensitivity(curve_df):
    fig, ax = plt.subplots(figsize = (8, 5))

    ax.plot(curve_df['ratio'], curve_df['ks_stat'], marker = 'o', color = '#1f77b4')

    ax.set_xlabel(r'Off-rate ratio $k_{off,nag}/k_{off,ag}$')
    ax.set_ylabel('KS statistic (ag vs nag)')
    ax.set_title('Sensitivity of class separability to off-rate ratio')

    ax.axvspan(0, 3, color='grey', alpha=0.08, label='S1: near-equal')
    ax.axvspan(3, 10, color='grey', alpha=0.15, label='S2: moderate')
    ax.axvspan(10, curve_df['ratio'].max(), color='grey', alpha=0.22, label='S3: strong')

    ax.legend(loc='lower right', fontsize=8)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(data_path(f'kinetics/ks_curve', 'ks_curve_visualization.png'), dpi=300)
    plt.close(fig)

if __name__ == '__main__':
    curve_df = compute_ks_curve()
    curve_df.to_csv(data_path('kinetics/ks_curve', 'ks_curve.csv'), index=False)

    plot_ks_sensitivity(curve_df)