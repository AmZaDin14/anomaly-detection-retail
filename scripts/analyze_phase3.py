#!/usr/bin/env python
"""Analyze Phase 3 results."""
import pandas as pd, numpy as np

df = pd.read_parquet('data/results/phase3_results.parquet')

print("=" * 70)
print("PHASE 3: FULL RESULTS")
print("=" * 70)

for ds in sorted(df['dataset'].unique()):
    sub = df[df['dataset'] == ds]
    print(f"\n--- {ds.upper()} ---")
    pivot = sub.groupby(['method', 'anomaly_rate'])[['precision', 'recall', 'f1', 'auc_roc', 'auc_pr']].mean().round(4)
    print(pivot.to_string())

print("\n" + "=" * 70)
print("STL-PELT_0.7 IMPROVEMENT vs STL+IQR (calendar data)")
print("=" * 70)
cal = df[df['dataset'] == 'food_prices_calendar']
bl = cal[cal['method'] == 'STL+IQR'].groupby('anomaly_rate')['f1'].mean()
pelt = cal[cal['method'] == 'STL-PELT_0.7'].groupby('anomaly_rate')['f1'].mean()
ae = cal[cal['method'] == 'Autoencoder'].groupby('anomaly_rate')['f1'].mean()

for rate in [0.01, 0.05, 0.10]:
    print(f"\n  Rate {rate*100:.0f}%:")
    print(f"    STL-PELT_0.7:  F1={pelt[rate]:.4f} (+{(pelt[rate]-bl[rate])/bl[rate]*100:.1f}% vs STL+IQR)")
    print(f"    STL+IQR:       F1={bl[rate]:.4f}")
    print(f"    Autoencoder:   F1={ae[rate]:.4f} (+{(ae[rate]-bl[rate])/bl[rate]*100:.1f}% vs STL+IQR)")

print("\n combined with original baselines")
orig = pd.read_parquet('data/results/all_results.parquet')
combined = pd.concat([orig, df], ignore_index=True)
combined = combined.drop_duplicates(subset=['dataset','series_id','method','anomaly_rate'])
by_group = combined.groupby(['dataset','anomaly_rate','method'])['f1'].mean().reset_index()

for ds in ['m5', 'food_prices', 'online_retail', 'food_prices_calendar']:
    sub = by_group[by_group['dataset'] == ds]
    print(f"\n  {ds}:")
    for rate in [0.01, 0.05, 0.10]:
        rate_sub = sub[sub['anomaly_rate'] == rate].sort_values('f1', ascending=False)
        print(f"  @{rate*100:.0f}%:")
        for _, row in rate_sub.head(6).iterrows():
            print(f"    {row['method']:20s} F1={row['f1']:.4f}")

print("\nDone.")
