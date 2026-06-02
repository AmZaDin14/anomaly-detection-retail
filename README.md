# Anomaly Detection for Micro-Retail Food Prices

**Unsupervised Anomaly Detection for Retail Food Price Time Series: A Comparative Study with Real-Event Validation**

This repository contains the code and experimental results for a comparative study of unsupervised anomaly detection methods applied to retail food price time series, with a focus on Indonesian PIHPS data.

## Key Contributions

1. **Real-event validation framework** — A curated calendar of 25 real economic events (BBM hike, El Nino, Idul Fitri/Adha, cooking oil crisis) used as ground truth for anomaly detection, enabling evaluation of methods in realistic conditions.
2. **STL-PELT hybrid anomaly score** — A novel parameterized score combining STL decomposition residuals with PELT change point proximity, controlled by an alpha weighting parameter.
3. **Empirical finding** — Synthetic evaluation systematically misranks anomaly detection methods. Autoencoder achieves the highest F1 on synthetic benchmarks but misses 76% of real events.

## Methods Implemented

| Method | Category | Reference |
|--------|----------|-----------|
| Isolation Forest | Density-based | Liu et al. (2008) |
| LOF | Density-based | Breunig et al. (2000) |
| Autoencoder | Deep learning | Aggarwal (2017) |
| Prophet + Residual | Time series | Taylor & Letham (2018) |
| STL + IQR | Decomposition | Cleveland et al. (1990) |
| **STL-PELT (proposed)** | **Hybrid** | **This work** |
| TimesNet | Deep learning | Wu et al. (2023) |

## Project Structure

```
data/
  raw/              -- Original datasets (download via script)
  processed/        -- Cleaned daily series (parquet)
  results/          -- Experiment outputs + figures
src/
  data/             -- Data loading + preprocessing
  methods/          -- 7 AD method implementations
  evaluation/       -- Metrics + statistical tests (Friedman, Nemenyi)
  visualization/    -- Publication-quality plots
scripts/
  run_pipeline.py         -- End-to-end experiment runner
  generate_report.py      -- Paper-ready tables
  alpha_ablation.py       -- STL-PELT alpha parameter sweep
  clean_validation.py     -- Real-event calendar validation
```

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
cd anomaly-detection-retail
uv sync
uv run python -m src.data.download_datasets
uv run python -m src.data.preprocessing
```

## Reproducing Experiments

```bash
# Full pipeline (all methods on all datasets)
uv run python scripts/run_pipeline.py

# STL-PELT alpha ablation
uv run python scripts/alpha_ablation.py

# Real-event calendar validation
uv run python scripts/clean_validation.py

# Generate paper tables
uv run python scripts/generate_report.py
```

## Results

Pre-computed experiment outputs are in `data/results/`:
- `all_results.parquet` — 1,350 result rows (5 methods x 3 datasets x 3 anomaly rates x 30 series)
- `alpha_ablation_results.parquet` — 630 rows (STL-PELT alpha sweep)
- `clean_validation_output.log` — Real-event detection rates by method

## Datasets

- **PIHPS** (Indonesian Food Price Data) — Daily prices for 30+ commodities, 2015-2026
- **M5 Forecasting** — Unit sales of 30 retail items, daily
- **Online Retail II** — UK e-commerce transaction data
- **Synthetic** — Controlled synthetic time series with injected anomalies

## Citation

If you use this code, please cite the associated paper (forthcoming).
