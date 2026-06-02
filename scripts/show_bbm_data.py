#!/usr/bin/env python3
"""Show price data around BBM hike."""
import pickle, sys
from datetime import date, timedelta
sys.path.insert(0, '.')
with open('data/results/food_prices_prepared.pkl', 'rb') as f:
    data = pickle.load(f)

event_date = date(2022, 9, 3)
start = event_date - timedelta(days=30)
end = event_date + timedelta(days=30)

for name in ['beras', 'minyak_goreng', 'cabai_rawit']:
    matches = [d for d in data if d['name'] == name]
    s = matches[0]['clean']
    mask = (s.index.date >= start) & (s.index.date <= end)
    sub = s[mask]
    print(f'\n--- {name} ---')
    for dt, val in sub.items():
        if dt.day % 3 == 0 or dt.date() == event_date:
            marker = ' <-- BBM' if dt.date() == event_date else ''
            print(f'  {dt.date()}: {val:>8.2f}{marker}')
