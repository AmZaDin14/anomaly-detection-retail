#!/usr/bin/env python
"""Friedman + Nemenyi tests on Phase 3 results."""
import pandas as pd, numpy as np
from scipy.stats import friedmanchisquare, norm
import itertools

# Load Phase 3 + original results
df = pd.read_parquet('data/results/phase3_results.parquet')
orig = pd.read_parquet('data/results/all_results.parquet')
combined = pd.concat([orig, df], ignore_index=True)
combined = combined.drop_duplicates(subset=['dataset','series_id','method','anomaly_rate'])

print("=" * 70)
print("FRIEDMAN + NEMENYI TESTS")
print("=" * 70)

# Test 1: All methods on M5 synthetic
# Test 2: All methods on food_prices calendar
# Test 3: STL-PELT_0.7 vs STL+IQR vs Autoencoder on calendar

for ds_name, methods_list, label in [
    ('m5', ['STL+IQR','STL-PELT_0.7','Autoencoder','IsolationForest','LOF','ProphetResidual'], 'M5 Synthetic'),
    ('food_prices_calendar', ['STL+IQR','STL-PELT_0.7','Autoencoder'], 'Food Prices Calendar'),
    ('online_retail', ['STL-PELT_0.7','STL+IQR','Autoencoder','IsolationForest','LOF','ProphetResidual'], 'Online Retail'),
]:
    sub = combined[combined['dataset'] == ds_name]
    # Pivot: each block = (series_id, anomaly_rate), columns = methods
    pivot = sub.pivot_table(index=['series_id', 'anomaly_rate'], columns='method', values='f1')[methods_list].dropna()
    
    if len(pivot) < 5:
        print(f"\n{label}: Too few blocks ({len(pivot)}), skipping")
        continue
    
    print(f"\n--- {label} ---")
    print(f"Blocks: {len(pivot)}, Methods: {len(methods_list)}")
    
    # Friedman
    stat, p_val = friedmanchisquare(*[pivot[m] for m in methods_list])
    print(f"Friedman: chi2={stat:.2f}, p={p_val:.6e}")
    
    # Average ranks
    ranks = pivot.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean().sort_values()
    print("\nAverage ranks (1=best):")
    for m in avg_ranks.index:
        print(f"  {m:25s} {avg_ranks[m]:.3f}")
    
    if p_val < 0.05 and len(methods_list) > 2:
        # Nemenyi pairwise
        k = len(methods_list)
        n = len(pivot)
        se = np.sqrt(k * (k + 1) / (6 * n))
        n_comp = k * (k - 1) // 2
        
        print(f"\nNemenyi pairwise (CD critical region):")
        for a, b in itertools.combinations(methods_list, 2):
            diff = avg_ranks[a] - avg_ranks[b]
            z = abs(diff) / se
            p_raw = 2 * (1 - norm.cdf(z))
            p_adj = min(p_raw * n_comp, 1.0)
            sig = '***' if p_adj < 0.001 else '**' if p_adj < 0.01 else '*' if p_adj < 0.05 else 'ns'
            print(f"  {a:25s} vs {b:25s}: diff={diff:.3f}, p_adj={p_adj:.4f} {sig}")

print("\n\nDone.")
