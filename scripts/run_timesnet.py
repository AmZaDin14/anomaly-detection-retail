#!/usr/bin/env python3
"""Run TimesNet on calendar data and combine with existing results."""
import pickle, sys, time, signal, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DATA = Path(__file__).resolve().parent.parent / 'data' / 'results'

from src.evaluation.metrics import evaluate_detection
from src.methods.timesnet_detector import TimesNetDetector

with open(DATA / 'food_prices_calendar_prepared.pkl', 'rb') as f:
    datasets = pickle.load(f)

results = []
total = sum(len(d['configs']) for d in datasets)
log.info('Running TimesNet on %d calendar configs...', total)

i = 0
for ds in datasets:
    sid = ds['name']
    for cfg in ds['configs']:
        i += 1
        if i % 5 == 0 or i == 1:
            log.info('  [TimesNet] %d/%d', i, total)
        contaminated = cfg['contaminated']
        labels = cfg['labels']
        rate = cfg['cfg']['rate']
        try:
            detector = TimesNetDetector(window=64, epochs=20, device='cpu')
            t0 = time.time()
            detector.fit(contaminated)
            scores = detector.score(contaminated)
            predictions = detector.predict(contaminated)
            fit_time = time.time() - t0
            metrics = evaluate_detection(labels, predictions, scores)
            results.append({
                'dataset': 'food_prices_calendar', 'series_id': sid,
                'method': 'TimesNet', 'anomaly_rate': rate,
                'fit_time_s': round(fit_time, 4), **metrics,
            })
        except Exception as e:
            log.warning('  [TimesNet] FAILED %s rate=%.2f: %s', sid, rate, e)

import pandas as pd
df = pd.DataFrame(results)
out = DATA / 'timesnet_calendar_results.parquet'
df.to_parquet(out)
log.info('Saved %d rows to %s', len(df), out)

# Summary
summary = df.groupby('anomaly_rate')[['precision','recall','f1']].mean().round(4)
log.info('\n%s', summary.to_string())
log.info('Done.')
