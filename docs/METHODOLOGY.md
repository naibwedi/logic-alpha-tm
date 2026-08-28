# Methodology

## Research question

Can an interpretable Boolean classifier select a small strategy set better out
of sample than static and equal-weight alternatives after estimated costs?

## Information timing

Features at date `t` use closes available at `t`. A decision formed at `t` first
earns a strategy return at `t+1`. Forward labels may use future outcomes, but a
training row is admitted only when its complete label horizon ends before the
walk-forward test boundary. Boolean thresholds are fitted on that training fold.

This handles the most common forms of look-ahead leakage. A production dataset
must additionally record each observation's `available_at` timestamp and account
for macro release delays, revisions, historical constituents, and delistings.

## Strategies

- `trend`: SPY when its 20-day mean exceeds its 100-day mean, otherwise cash.
- `momentum`: strongest trailing 60-day return among SPY, QQQ, and IWM.
- `defensive`: SPY in a positive 100-day trend, otherwise TLT.
- `cash`: zero gross return.

Signals are lagged one session. Strategy changes incur configurable one-way
costs. The meta-selector makes decisions weekly and incurs an additional switch
cost when it changes strategy.

## Labels

For strategy `s` and horizon `h`:

`utility = compounded_return - lambda_vol * realized_volatility - lambda_dd * abs(max_drawdown)`

The winning strategy is labeled only when it beats cash and the runner-up by the
configured dead zone. Otherwise the label is cash. This avoids forcing a winner
when estimated differences are economically negligible.

## Validation

The runner uses expanding walk-forward folds. Each fold trains only on labels
fully resolved before its test block. Model selection must happen within the
historical portion; the generated test stream must not be repeatedly optimized.

Primary metrics are CAGR, annualized volatility, Sharpe, Sortino, Calmar, maximum
drawdown, turnover, and total return. Classification accuracy is diagnostic only.

## Interpretability

The baseline exports per-class log-likelihood contributions for each Boolean
literal. Tsetlin Machine Unified (TMU) exposes clause-level logic, but clause
comparisons require
canonicalization: redundant nested thresholds should be simplified before exact
or Jaccard comparison. Stability and profitability are separate hypotheses.

## Limitations

- Synthetic data cannot establish financial value.
- Adjusted-close CSVs do not model intraday execution, taxes, borrow, or capacity.
- The volatility/drawdown utility embeds researcher preferences.
- Multiple comparisons can create false discoveries.
- Tsetlin Machine vote margins are scores, not calibrated probabilities.

