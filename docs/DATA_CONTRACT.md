# Point-in-time data contract

The minimal runnable CSV has one row per trading session:

```csv
date,SPY,QQQ,IWM,TLT
2024-01-02,472.65,401.56,198.90,98.01
```

Values must be positive, complete adjusted closes with increasing unique dates.
For serious research, replace the simple CSV loader with a long-form feature
store containing at least:

| Field | Meaning |
|---|---|
| `observation_at` | Economic period represented by the value |
| `available_at` | Earliest timestamp the strategy could have known it |
| `source` | Vendor and series identifier |
| `revision` | Vintage/version when the source revises history |
| `value` | Numeric observation |

Features must join on `available_at <= decision_at`, not merely on observation
date. Historical breadth requires the constituent universe known on each date,
including later-delisted securities.

