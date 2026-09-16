# Tsetlin Machine risk-filter pilot (v0.3)

The research question is whether a model can improve a strategy blend by
reducing exposure during unfavorable periods. The two actions are **blend**
and **half blend**. The model never selects full cash in this experiment.
Ambiguous training outcomes default to the blend. Votes are not interpreted
as probabilities, and the model does not yet have calibrated abstention.

## Run and resume

Use a Python 3.12 project environment with the `tm` extra installed:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[tm]"
.\.venv\Scripts\python.exe -m logic_alpha_tm.cli risk-audit --csv data/raw/tiingo-prices.csv --output results/tiingo-risk-v0.3
```

After an interruption, repeat the command with `--resume`. Completed folds are
restored; the interrupted fold is retrained. Progress includes model, seed,
fold, training observations, test dates and TMU epochs. Checkpoints are not
shared across changed data, source code, settings or runtime versions. Use a
new output folder for a different experiment. Do not run two writers in the
same output folder.

## Frozen pilot choices

`experiments/tiingo-risk-v0.3.json` fixes 2008–2020 development dates, three TMU
seeds, a 20-session label horizon, five-session allocation decisions, and
200 clauses / 10 epochs. This smaller model is a predeclared capacity choice
for a binary pilot, not a parameter selected after observing results.
Bernoulli and logistic alternatives are evaluated once because their current
implementations are deterministic. No extra features were added in this pilot,
so the effect of the new decision problem can be assessed first.

The seed passed to TMU is the experiment seed plus the fold index. CPU
repeatability is checked separately; GPU and CPU results need not be identical.
Requested CUDA runs fail if TMU silently selects a CPU clause bank.

## Audit corrections

- Prices are truncated at the development end **before** features, labels or
  returns are constructed. Unresolved labels at the end are not scored; the
  model can still make predictions on those dates.
- Labels compare compounded future return minus horizon-scale volatility and
  drawdown penalties. A reduced-exposure utility advantage must exceed 0.3%.
  Ties use the blend. Future windows start at the first executable return.
- Training rows are purged until the entire label window ends strictly before
  the first test decision date. Boolean thresholds use only that training set.
- A signal formed after close t is executed at close t+1 and first earns the
  close-to-close return ending t+2. All underlying strategies obey that timing.
- Costs use the absolute change in risky-asset weights after accounting for
  price-driven drift, at a per-side rate. Switching strategy names has no
  separate fee. Initial entry is charged. Costs are approximated relative to
  pre-cost wealth, and portfolios rebalance to target weights daily.
- Sharpe uses arithmetic daily mean divided by sample standard deviation,
  annualized by sqrt(252). Sortino uses downside root-mean-square. Drawdown
  includes starting wealth, so an initial loss is counted.
- Cash and risk-free returns remain zero. Real historical cash yields, closing
  auction slippage and next-open execution are not implemented here.

## Comparators and success rule

Each model is compared with the equal-weight strategy blend, constant half
blend, the always-defensive strategy, SPY, and a simple risk filter that halves
the blend when SPY's 60-day return is negative and 20-day volatility exceeds
60-day volatility. The simple filter updates daily; learned actions update
every five sessions. This cadence difference is recorded, not optimized.

Cost tests replay the **same learned decisions** at 2, 5 and 10 bps without
retraining. They measure execution-cost sensitivity, not robustness to
cost-dependent label changes. The training-label cost remains 2 bps.

Advance the TMU pilot only if every seed beats both the blend and simple risk
filter in Sharpe at every cost, while its maximum drawdown is no worse than
the blend. Report all seeds, including failures. This is a descriptive
development screen, not a statistical-significance test or final evidence.

## Artifacts and provenance

The manifest fingerprints source files, both input files, settings, Python,
packages and operating system. Each fold has an atomic checkpoint. Outputs
include per-asset-portfolio accounting summaries, net return streams, model
decisions, label diagnostics, a full comparison, decision gates and a report.
`completed.json` is written only after the full experiment finishes.

The original v0.2 code and outputs are retained for provenance; their Sharpe,
execution and label definitions differ from this engine. Compare new models
against the new engine's baselines, not directly against the old headline
numbers. v0.3 currently has no holdout execution command.

Downloaded data and derived run artifacts remain ignored by Git. Store shared
research outputs only when the provider's license permits their disclosure.
Clause-level explanations, cash-yield sensitivity, uncertainty intervals and
additional feature groups remain future work rather than completed features.
