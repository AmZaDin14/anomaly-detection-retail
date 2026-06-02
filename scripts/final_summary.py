#!/usr/bin/env python3
"""Final consolidated results for the paper."""
import pandas as pd
import numpy as np

# Load all results
orig = pd.read_parquet('data/results/all_results.parquet')
phase3 = pd.read_parquet('data/results/phase3_results.parquet')
timesnet = pd.read_parquet('data/results/timesnet_calendar_results.parquet')

# Combine
all_df = pd.concat([orig, phase3, timesnet], ignore_index=True)
all_df = all_df.drop_duplicates(subset=['dataset','series_id','method','anomaly_rate'])

# =========================================================
# TABLE 1: Calendar data — ALL methods comparison
# =========================================================
print('=' * 80)
print('TABLE 1: Food Prices Calendar — All Methods')
print('=' * 80)
cal = all_df[all_df['dataset'] == 'food_prices_calendar']
for rate in [0.01, 0.05, 0.10]:
    sub = cal[cal['anomaly_rate'] == rate]
    ranked = sub.groupby('method')[['precision','recall','f1']].mean().round(4).sort_values('f1', ascending=False)
    print(f'\nAnomaly rate: {rate*100:.0f}%')
    print(f'{"Method":25s} {"Precision":>10s} {"Recall":>10s} {"F1":>10s}')
    print('-' * 55)
    for method, row in ranked.iterrows():
        print(f'{method:25s} {row["precision"]:>10.4f} {row["recall"]:>10.4f} {row["f1"]:>10.4f}')

# =========================================================
# TABLE 2: Cross-dataset STL-PELT performance
# =========================================================
print('\n' + '=' * 80)
print('TABLE 2: STL-PELT_0.7 Cross-Dataset Performance')
print('=' * 80)
pelt = all_df[all_df['method'] == 'STL-PELT_0.7']
for ds in sorted(pelt['dataset'].unique()):
    sub = pelt[pelt['dataset'] == ds]
    print(f'\n{ds}:')
    for rate in [0.01, 0.05, 0.10]:
        r = sub[sub['anomaly_rate'] == rate]
        p = r['precision'].mean()
        rec = r['recall'].mean()
        f = r['f1'].mean()
        print(f'  @{rate*100:.0f}%: P={p:.4f} R={rec:.4f} F1={f:.4f}')

# =========================================================
# TABLE 3: Computational cost
# =========================================================
print('\n' + '=' * 80)
print('TABLE 3: Computational Cost (fit time in seconds)')
print('=' * 80)
# Use M5 synthetic as standard benchmark
m5 = all_df[all_df['dataset'] == 'm5']
times = m5.groupby('method')['fit_time_s'].agg(['mean','std']).round(4).sort_values('mean')
print(times.to_string())

print('\nDone.')
