# Changelog

## Unreleased

- Added LogicVol Experiment A (`logic-alpha logicvol`): a Regression Tsetlin
  Machine forecasting next-week log volatility against historical, EWMA,
  HAR-style and Boolean-ridge baselines, with a frozen pass/fail rule, purged
  walk-forward folds, per-fold checkpoints and a log-normal correction for QLIKE.
- Added `logic-alpha logicvol-cut` to write a price copy that ends at the
  development end, and a Colab notebook for the private run.
- Added a free Tiingo EOD adapter with raw/adjusted audit files, corporate
  actions, availability metadata, coverage checks, and internal-use warnings.
- Added a separate preliminary Tiingo experiment specification so current-vintage
  evidence is not confused with strict revision-point-in-time validation.
- Added a frozen real-market development and holdout experiment specification.
- Added point-in-time availability validation and SHA-256 dataset manifests.
- Added dependency-light logistic-regression and boosted-stump benchmarks.
- Added multi-model sensitivity runs, decision-gate summaries, and holdout locking.
- Expanded tests and documentation for the real-data evidence workflow.

## 0.1.0 — 2026-08-28

- Added the complete leakage-aware research MVP.
- Added deterministic synthetic validation, CSV ingestion, strategies, labels,
  Boolean encoding, walk-forward selection, costs, reports, visuals, and tests.
- Added an optional Tsetlin Machine Unified (TMU) adapter and a verified
  dependency-light baseline.
