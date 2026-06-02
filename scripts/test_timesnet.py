#!/usr/bin/env python3
"""Smoke test for TimesNet detector."""
import pickle, sys
sys.path.insert(0, '.')
with open('data/results/m5_prepared.pkl', 'rb') as f:
    data = pickle.load(f)
s = data[0]['clean'][:512]
print(f'Series: {len(s)} points')

from src.methods.timesnet_detector import TimesNetDetector
d = TimesNetDetector(window=64, epochs=10, device='cpu')
d.fit(s)
sc = d.score(s)
print(f'Score range: [{sc.min():.4f}, {sc.max():.4f}]')
pred = d.predict(s)
print(f'Anomalies: {pred.sum()} ({pred.mean()*100:.1f}%)')
print('TimesNet smoke test OK')
