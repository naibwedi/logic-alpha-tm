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

For real data, supply a wide CSV with `date` and adjusted close columns named
`SPY`, `QQQ`, `IWM`, and `TLT`:

```powershell
python -m logic_alpha_tm.cli run --csv data/raw/prices.csv --output results
```

The default interpretable model is Bernoulli Naive Bayes, which provides a
dependency-light verified baseline. Select `--model tmu` after installing
Tsetlin Machine Unified (TMU).
The adapter follows the official import path
`tmu.models.classification.vanilla_classifier.TMClassifier`. Tsetlin Machine
vote margins
remain unavailable through the version-stable adapter and are never presented as
probabilities.

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
[docs/ROADMAP.md](docs/ROADMAP.md) before expanding the system.

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
