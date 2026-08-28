# Roadmap and research log

## Completed in v0.1

- Reproducible synthetic multi-regime market generator.
- Wide-CSV loader and schema validation.
- Point-in-time trend, momentum, volatility, relative-strength, and regime fields.
- Train-fold quantile Boolean encoder.
- Four executable strategy streams with costs and lagged signals.
- Risk-adjusted forward utility and no-edge/cash labels.
- Purged expanding walk-forward evaluation.
- Interpretable Bernoulli baseline and optional Tsetlin Machine Unified (TMU)
  model adapter.
- Selector, static, equal-weight, and buy-and-hold comparisons.
- Metrics, rule export, vote-margin calibration, SVG visual, and Markdown report.
- Unit tests for timing, encoding, labels, folds, and end-to-end execution.

## Next evidence milestones

1. Add licensed point-in-time ETF, VIX, yields, and breadth data with `available_at`.
2. Freeze a final holdout before tuning.
3. Run nested walk-forward model and utility-parameter selection.
4. Compare the Tsetlin Machine against logistic regression, boosted trees, and
   strategy blending.
5. Repeat seeds; canonicalize clauses and measure activation-set similarity.
6. Test whether clause stability predicts future utility after correction for
   multiple testing.
7. Paper trade with recorded decisions before considering broker integration.

## Decision gates

- Stop if the selector cannot beat equal-weight strategies after costs.
- Stop if results depend on one crisis or one utility setting.
- Continue only if conclusions persist across seeds, windows, and cost assumptions.

