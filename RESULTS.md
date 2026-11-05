# Results

Every experiment in this repository was specified before it ran, and every
result is reported here, including the failures. All numbers come from the
development period (2008 to 2020). The 2021 to 2025 holdout has not been used.
These are research results, not trading advice.

## LogicVol v0.1: Regression Tsetlin Machine volatility forecast — FAIL

Spec: [`experiments/logicvol-v0.1.json`](experiments/logicvol-v0.1.json).
Run on 2026-11-05 from commit `550b6f2`, Tiingo current-vintage adjusted
closes from 2002-07-26 to 2020-12-31 (input SHA-256 `785caf8c…d1fe28`),
26 walk-forward folds, 3,269 daily test forecasts per asset.

**Question.** Can a Regression Tsetlin Machine on train-only Boolean features
forecast next-week SPY volatility better than simple estimates and close to a
HAR-style model?

**Decision (SPY, frozen rule): FAIL on all three checks.**

| SPY, 2008–2020 | QLIKE (lower is better) | MSE of log volatility |
|---|--:|--:|
| HAR-style | **0.524** | **0.259** |
| EWMA (λ = 0.94) | 0.555 | 0.291 |
| Ridge on the Boolean features | 0.622 | 0.279 |
| Historical, last 22 sessions | 0.638 | 0.301 |
| Tsetlin Machine, seed 7 / 19 / 41 | 0.798 / 0.729 / 0.796 | 0.304 / 0.308 / 0.315 |

The rule needed the median seed to beat historical and EWMA and to come within
5% of HAR-style (about 0.550). The median was 0.796.

Robustness assets (reported only, they do not decide): the Tsetlin Machine was
also worse than HAR-style on QQQ (median 0.584 vs 0.467), IWM (0.659 vs 0.408)
and TLT (0.417 vs 0.322).

### Where it lost

QLIKE per calendar year, Tsetlin Machine (median of seeds) against HAR-style:

| Asset | Years the TM beat HAR-style | Worst TM year |
|---|---|---|
| SPY | 7 of 13 (2010, 2011, 2013–2016, 2019) | 2008: 3.32 vs 0.55 |
| QQQ | 7 of 13 (2012–2017, 2019) | 2008: 1.41 vs 0.44 |
| IWM | 6 of 13 (2011, 2012, 2014–2017) | 2008: 2.72 vs 0.62 |
| TLT | 4 of 13 (2010, 2012, 2016, 2019) | 2020: 0.91 vs 0.66 |

The Tsetlin Machine was often better in calm years and failed badly in shock
years (2008, and for SPY also 2018 and 2020). The likely reason is structural:
the thermometer encoding stops at the training window's 90th percentile, so the
model cannot describe, or forecast, volatility above levels it has seen. Ridge
regression on the same Boolean features shows the same weakness (0.622), while
HAR-style, a linear model on log volatility, extrapolates.

This per-year pattern was found after the run. It is a lead, not a finding:
yearly QLIKE on overlapping weekly targets is noisy.

### Changes to the spec before the run

Both are recorded in the spec's `implementation` section:

1. Weighted clauses, because unweighted clauses cap the output at the clause
   count, below `T` for two of the three settings.
2. A log-normal correction for QLIKE on models of mean log volatility. It was
   added after one baselines-only run showed HAR-style losing to EWMA on QLIKE
   while winning on MSE, and before any Tsetlin Machine result existed. The
   uncorrected QLIKE is reported alongside; the decision is FAIL under either.

### What follows

As the spec requires, v0.1 is closed and nothing was tuned on its test folds.
A gated hybrid that uses HAR-style outside the Tsetlin Machine's input range is
being designed as v0.2. Because it was designed from these results, only the
locked holdout can judge it.

## Earlier strategy-selection experiments

All three failed their pre-registered gates on 2008–2020:

| Experiment | Question | Result |
|---|---|---|
| v0.2 (`tiingo-v0.2.json`) | Can a model pick among trend, momentum, defensive and cash better than an equal blend? | 0 of 28 model/setting runs passed |
| v0.3 (`tiingo-risk-v0.3.json`) | Can a model decide when to halve the blend? | 0 of 9 Tsetlin Machine seed/cost runs passed; a hand-written stress rule did better |
| v0.4 (`tiingo-v0.4-features.json`) | Do three predeclared feature groups help? | 0 of 18 runs passed; no consistent gain across seeds |

Conclusion so far: on these four ETFs, neither strategy selection nor a
Tsetlin Machine volatility forecast beat simple baselines over 2008–2020.
