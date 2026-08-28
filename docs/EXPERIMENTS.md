# Experiment log

## Synthetic walk-forward benchmark

This benchmark validates software behavior only. Both models used the same
deterministic synthetic series, features, purged expanding folds, next-session
execution, costs, and untouched test observations.

| Model / strategy | OOS accuracy | Total return | CAGR | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|---:|
| Tsetlin Machine selector | 61.73% | 1.18% | 0.25% | 0.05 | -13.56% |
| Bernoulli selector | 56.29% | -4.56% | -1.00% | -0.10 | -19.72% |
| Equal weight | — | -15.33% | -3.50% | -0.32 | -28.36% |
| SPY | — | -9.59% | -2.14% | -0.11 | -37.07% |
| Always trend | — | 7.48% | 1.56% | 0.16 | -18.61% |

The Tsetlin Machine beat the learned baseline and two passive comparators on
this synthetic path, but did not beat the static trend strategy. These results
are not evidence of market predictability or tradable alpha.

## Licensed-market benchmark

Status: **adapter implemented; execution awaiting a locally configured licensed
API key**. The repository cannot redistribute vendor data or embed credentials.
When credentials are available, run both commands in `DATA_SOURCES.md`, preserve
the complete output folders, and add a dated table here without tuning against
the final holdout.
