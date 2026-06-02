#!/usr/bin/env python3
"""Generate all figures for the paper."""

import pickle, sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import friedmanchisquare, norm
import itertools

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
DATA = Path(__file__).resolve().parent.parent / "data" / "results"
FIGS = Path(__file__).resolve().parent.parent / "data" / "figures"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#6A994E']
METHOD_COLORS = {
    'STL-PELT_0.7': '#2E86AB',
    'STL+IQR': '#A23B72',
    'Autoencoder': '#F18F01',
    'TimesNet': '#6A994E',
    'IsolationForest': '#C73E1D',
    'LOF': '#3B1F2B',
    'ProphetResidual': '#888888',
}


def fig1_cd_diagram():
    """Critical Difference diagram: STL-PELT vs baselines on M5 synthetic."""
    orig = pd.read_parquet(DATA / "all_results.parquet")
    df = pd.read_parquet(DATA / "phase3_results.parquet")
    combined = pd.concat([orig, df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['dataset','series_id','method','anomaly_rate'])

    for ds_name, label in [('m5', 'M5 Competition'), ('online_retail', 'Online Retail II')]:
        sub = combined[combined['dataset'] == ds_name]
        methods = ['STL-PELT_0.7','STL+IQR','Autoencoder','IsolationForest','LOF','ProphetResidual']
        pivot = sub.pivot_table(index=['series_id','anomaly_rate'], columns='method', values='f1')[methods].dropna()
        ranks = pivot.rank(axis=1, ascending=False)
        avg_ranks = ranks.mean().sort_values()

        fig, ax = plt.subplots(figsize=(10, 4))
        methods_sorted = avg_ranks.index.tolist()
        ranks_sorted = avg_ranks.values

        # Horizontal bar chart of average ranks
        colors = [METHOD_COLORS.get(m, '#888888') for m in methods_sorted]
        bars = ax.barh(range(len(methods_sorted)), ranks_sorted, color=colors, height=0.6)
        ax.set_yticks(range(len(methods_sorted)))
        ax.set_yticklabels(methods_sorted, fontsize=10)
        ax.set_xlabel('Average Rank (lower is better)')
        ax.set_title(f'{label} — Method Rankings')
        ax.invert_yaxis()
        ax.set_xlim(0, max(ranks_sorted) + 0.5)

        # Add rank values
        for i, (bar, v) in enumerate(zip(bars, ranks_sorted)):
            ax.text(0.1, bar.get_y() + bar.get_height()/2, f'{v:.2f}',
                    va='center', fontsize=9, fontweight='bold', color='white')

        plt.tight_layout()
        path = FIGS / f'cd_diagram_{ds_name}.png'
        plt.savefig(path)
        plt.close()
        print(f"  Saved {path}")


def fig2_calendar_comparison():
    """Bar chart comparing all methods on calendar data."""
    df = pd.read_parquet(DATA / "phase3_results.parquet")
    tn = pd.read_parquet(DATA / "timesnet_calendar_results.parquet")
    cal = pd.concat([df[df['dataset'] == 'food_prices_calendar'], tn], ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for idx, rate in enumerate([0.01, 0.05, 0.10]):
        ax = axes[idx]
        sub = cal[cal['anomaly_rate'] == rate]
        grouped = sub.groupby('method')[['precision', 'recall', 'f1']].mean()

        methods = [m for m in ['STL-PELT_0.7','STL+IQR','Autoencoder','TimesNet'] if m in grouped.index]
        x = np.arange(len(methods))
        width = 0.25

        ax.bar(x - width, grouped.loc[methods, 'precision'], width, label='Precision',
               color='#2E86AB', alpha=0.8)
        ax.bar(x, grouped.loc[methods, 'recall'], width, label='Recall',
               color='#F18F01', alpha=0.8)
        ax.bar(x + width, grouped.loc[methods, 'f1'], width, label='F1',
               color='#6A994E', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha='right', fontsize=8)
        ax.set_title(f'Anomaly Rate: {rate*100:.0f}%', fontsize=11)
        ax.set_ylim(0, 0.75)
        ax.legend(fontsize=8, loc='upper right')

    fig.suptitle('Food Prices Calendar — Precision, Recall, F1 by Method', fontsize=13, y=1.02)
    plt.tight_layout()
    path = FIGS / 'calendar_comparison.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def fig3_alpha_ablation():
    """Alpha ablation line plot."""
    df = pd.read_parquet(DATA / "alpha_ablation_results.parquet")

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {0.01: 'o', 0.05: 's', 0.10: '^'}
    colors = {0.01: '#2E86AB', 0.05: '#F18F01', 0.10: '#C73E1D'}

    for rate in [0.01, 0.05, 0.10]:
        sub = df[df['anomaly_rate'] == rate]
        alphas = sorted(sub['alpha'].unique())
        f1s = [sub[sub['alpha'] == a]['f1'].mean() for a in alphas]
        ax.plot(alphas, f1s, marker=markers[rate], color=colors[rate],
                label=f'{rate*100:.0f}% anomalies', linewidth=2, markersize=8)

    ax.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5, label='α=0.7 (default)')
    ax.set_xlabel('Alpha (balance weight)', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('STL-PELT: Effect of α on Detection Performance', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.grid(alpha=0.2)
    plt.tight_layout()
    path = FIGS / 'alpha_ablation.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def fig4_clean_validation():
    """Detection rate + lift from clean PIHPS validation."""
    # Data from clean_validation_output.log
    methods = ['STL+IQR', 'STL-PELT_0.7', 'Autoencoder']
    detection_rates = [84, 48, 24]
    lifts = [4.45, 5.64, 4.23]
    bg_rates = [0.121, 0.060, 0.021]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Detection rate
    ax = axes[0]
    colors = [METHOD_COLORS.get(m, '#888888') for m in methods]
    ax.bar(methods, detection_rates, color=colors, alpha=0.8)
    ax.set_ylabel('Detection Rate (%)', fontsize=11)
    ax.set_title('Real Event Detection Rate', fontsize=12)
    for i, v in enumerate(detection_rates):
        ax.text(i, v + 2, f'{v}%', ha='center', fontsize=10, fontweight='bold')

    # Lift
    ax = axes[1]
    ax.bar(methods, lifts, color=colors, alpha=0.8)
    ax.set_ylabel('Lift (event / background)', fontsize=11)
    ax.set_title('Event vs Background Density', fontsize=12)
    for i, v in enumerate(lifts):
        ax.text(i, v + 0.1, f'{v:.1f}x', ha='center', fontsize=10, fontweight='bold')

    # Background FPR
    ax = axes[2]
    ax.bar(methods, bg_rates, color=colors, alpha=0.8)
    ax.set_ylabel('Background Anomaly Rate', fontsize=11)
    ax.set_title('False Positive Baseline', fontsize=12)
    for i, v in enumerate(bg_rates):
        ax.text(i, v + 0.003, f'{v:.1%}', ha='center', fontsize=10, fontweight='bold')

    for ax in axes:
        ax.tick_params(axis='x', rotation=20)

    plt.tight_layout()
    path = FIGS / 'clean_validation.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def fig5_computational_cost():
    """Bar chart of fit times."""
    df = pd.read_parquet(DATA / "all_results.parquet")
    df2 = pd.read_parquet(DATA / "phase3_results.parquet")
    combined = pd.concat([df, df2], ignore_index=True)
    combined = combined.drop_duplicates(subset=['dataset','series_id','method','anomaly_rate'])
    m5 = combined[combined['dataset'] == 'm5']
    times = m5.groupby('method')['fit_time_s'].mean().sort_values()

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [METHOD_COLORS.get(m, '#888888') if m in METHOD_COLORS else '#888888' for m in times.index]
    bars = ax.barh(range(len(times)), times.values, color=colors, height=0.6)
    ax.set_yticks(range(len(times)))
    ax.set_yticklabels(times.index, fontsize=9)
    ax.set_xlabel('Mean Fit Time (seconds)', fontsize=11)
    ax.set_title('Computational Cost per Series (M5)', fontsize=12)

    for i, (bar, v) in enumerate(zip(bars, times.values)):
        ax.text(0.1, bar.get_y() + bar.get_height()/2, f'{v:.2f}s',
                va='center', fontsize=8, color='white', fontweight='bold')

    ax.invert_yaxis()
    plt.tight_layout()
    path = FIGS / 'computational_cost.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def fig6_summary_table():
    """Text-based summary table as a figure."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')

    # Calendar data F1 comparison
    df = pd.read_parquet(DATA / "phase3_results.parquet")
    tn = pd.read_parquet(DATA / "timesnet_calendar_results.parquet")
    cal = pd.concat([df[df['dataset'] == 'food_prices_calendar'], tn], ignore_index=True)

    table_data = []
    for method in ['STL-PELT_0.7', 'STL+IQR', 'Autoencoder', 'TimesNet']:
        row = [method]
        for rate in [0.01, 0.05, 0.10]:
            sub = cal[(cal['method'] == method) & (cal['anomaly_rate'] == rate)]
            row.append(f"{sub['f1'].mean():.3f}")
            row.append(f"{sub['precision'].mean():.3f}")
            row.append(f"{sub['recall'].mean():.3f}")
        table_data.append(row)

    col_labels = ['Method',
                  '1%\nF1', '1%\nP', '1%\nR',
                  '5%\nF1', '5%\nP', '5%\nR',
                  '10%\nF1', '10%\nP', '10%\nR']
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center',
                     cellLoc='center', colWidths=[0.15]*10)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # Bold best values
    for col in [1, 4, 7]:  # F1 columns
        vals = [row[col] for row in table_data]
        best_idx = np.argmax([float(v) for v in vals])
        table[(best_idx + 1, col)].set_facecolor('#2E86AB')
        table[(best_idx + 1, col)].set_text_props(weight='bold')

    ax.set_title('Food Prices Calendar: F1 / Precision / Recall by Method', fontsize=12, pad=20)
    plt.tight_layout()
    path = FIGS / 'summary_table.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


if __name__ == '__main__':
    print("Generating figures...")
    fig1_cd_diagram()
    fig2_calendar_comparison()
    fig3_alpha_ablation()
    fig4_clean_validation()
    fig5_computational_cost()
    fig6_summary_table()
    print(f"\nAll figures saved to {FIGS}")
