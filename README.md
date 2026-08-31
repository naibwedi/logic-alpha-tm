# LogicAlpha — Tsetlin Machine Research Platform

An interpretable, leakage-aware research platform for selecting among market
strategies with Boolean features and an optional Tsetlin Machine.

> Research software only. It is not investment advice and does not place trades.

## Why this design

The first version deliberately avoids a learned “regime classifier” that merely
reconstructs a hand-written trailing-return label. Regimes are descriptive
outputs. The falsifiable question is whether point-in-time Boolean information
can select strategies out of sample better than static and blended baselines.

```text
point-in-time prices -> features -> train-only Boolean thresholds
                                      |
                                      v
                         purged walk-forward selector
                                      |
                                      v
                    weekly choice -> costs -> portfolio

regime description -----------------> explanation/report
```

Implemented safeguards include next-session execution, overlapping-label purge,
train-only quantiles, a no-edge label, weekly decisions, transaction costs, and
a never-touched-by-training walk-forward test stream.

## Quick start

The demo uses a deterministic synthetic market so the repository is runnable
without network access or unverifiable vendor data.

```powershell
$env:PYTHONPATH = "src"
python -m logic_alpha_tm.cli demo --output results
python -m unittest discover -s tests -v
```

For real data, supply a wide CSV with `date` and close columns named
`SPY`, `QQQ`, `IWM`, and `TLT`:

```powershell
python -m logic_alpha_tm.cli run --csv data/raw/prices.csv --output results
```

### Free long-history path: Tiingo Starter

Tiingo Starter is sufficient for the four-symbol preliminary benchmark. Each
user supplies their own token; downloaded data remain local and must not be
redistributed. The downloader saves adjusted closes for research, raw closes and
corporate actions for audit, conservative availability metadata, and checksums.

```powershell
$env:TIINGO_API_TOKEN = "your-local-token"
python -m logic_alpha_tm.cli download-tiingo --start 2005-01-01 --end 2025-12-31
python -m logic_alpha_tm.cli benchmark --csv data/raw/tiingo-prices.csv --spec experiments/tiingo-v0.2.json --phase development --output results/tiingo-development-v0.2
```

Tiingo history is current-vintage corrected data, not a historical revision
archive. Results therefore form a low-cost preliminary benchmark rather than the
strictest possible point-in-time evidence. The final holdout remains locked.

### Optional private Google Colab GPU run

The repository includes a
[Colab GPU notebook](notebooks/tiingo_gpu_colab.ipynb) for the same frozen
development benchmark. Open it in Google Colab, select a T4 GPU, and upload only
`tiingo-prices.csv` plus `tiingo-prices.available-at.csv` when prompted. The
notebook never asks for the Tiingo token, records `CUDA` in the run manifest, and
downloads the results as a ZIP. Never make a copy public while licensed files
remain attached to its runtime.

Local CPU remains the default. The equivalent explicit local command adds
`--tmu-platform CPU`; use `CUDA` only on a configured NVIDIA/PyCUDA environment.

The repository also includes a [licensed Massive data adapter](https://massive.com/docs/rest/stocks/aggregates/custom-bars). Set the API key
locally, download unadjusted daily bars with point-in-time availability metadata,
then run the experiment. Never commit the key or downloaded vendor data.

```powershell
$env:MASSIVE_API_KEY = "your-local-key"
python -m logic_alpha_tm.cli download-massive --start 2005-01-01 --end 2025-12-31 --vendor-plan "your-plan-label"
python -m logic_alpha_tm.cli run --csv data/raw/massive-prices.csv --output results/real-bernoulli
```

The default interpretable model is Bernoulli Naive Bayes, which provides a
dependency-light verified baseline. Select `--model tmu` after installing
Tsetlin Machine Unified (TMU) with `pip install -e ".[tm]"`.
The adapter follows the official import path
`tmu.models.classification.vanilla_classifier.TMClassifier`. Tsetlin Machine vote
margins remain unavailable through the version-stable adapter and are never
presented as probabilities.

## Frozen real-market benchmark

The pre-registered experiment specification is
[`experiments/real-market-v0.2.json`](experiments/real-market-v0.2.json). It fixes
the development and final-holdout dates, comparison models, sensitivity runs,
and decision gates before the licensed benchmark is executed. The benchmark
requires the matching `*.available-at.csv` file and records SHA-256 fingerprints
for the prices, availability metadata, and experiment specification. It writes
`comparison.csv`, `decision-gates.csv`, `BENCHMARK.md`, and complete per-run
artifacts for independent review.

```powershell
pip install -e ".[tm]"
logic-alpha benchmark --csv data/raw/massive-prices.csv --phase development --output results/development-v0.2
```

After reviewing development results and committing any final methodology choices,
run the holdout exactly once. The explicit flag makes accidental holdout access
harder and records a separate result folder.

```powershell
logic-alpha benchmark --csv data/raw/massive-prices.csv --phase holdout --unlock-holdout --output results/holdout-v0.2
```

## Outputs

- `summary.json`: performance and experimental settings
- `equity.csv`: selector and baseline return streams
- `predictions.csv`: out-of-sample strategy decisions and vote margins
- `rules.csv`: most influential Boolean literals by strategy
- `rule_stability.csv`: fold recurrence of influential literals
- `margin_calibration.csv`: empirical accuracy by score-margin bucket
- `report.svg`: equity, drawdown, and selection visual
- `REPORT.md`: concise generated research report

Read [docs/METHODOLOGY.md](docs/METHODOLOGY.md) before interpreting results and
[docs/ROADMAP.md](docs/ROADMAP.md) before expanding the system. The first
[model comparison](docs/EXPERIMENTS.md) and [data-source protocol](docs/DATA_SOURCES.md)
record what is validated and what still requires licensed credentials.

## Current scope

This repository proves the full research plumbing with synthetic data. A real
market conclusion requires licensed, point-in-time data; repeated walk-forward
runs; parameter sensitivity tests; and a locked final holdout. Synthetic demo
performance is a software check, not evidence of tradable alpha.

## Research and financial disclaimer

This project is open-source research software. It does not provide investment
advice, recommendations, brokerage services, or assurances of future returns.
Models can fail, backtests can be biased, and users are responsible for validating
data, assumptions, execution constraints, and applicable regulations.

Contributions are welcome—see [CONTRIBUTING.md](CONTRIBUTING.md). For academic
reuse, cite the repository using [CITATION.cff](CITATION.cff).
