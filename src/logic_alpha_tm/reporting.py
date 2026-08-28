from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _points(values: pd.Series, x: float, y: float, width: float, height: float) -> str:
    arr = values.to_numpy(dtype=float)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    span = hi - lo or 1.0
    xs = np.linspace(x, x + width, len(arr))
    ys = y + height - (arr - lo) / span * height
    return " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))


def write_svg(equity: pd.DataFrame, predictions: pd.DataFrame, path: Path) -> None:
    colors = {"selector": "#ff5a36", "equal_weight": "#17b890", "SPY": "#5b7cfa"}
    lines = []
    for name in equity.columns:
        color = colors.get(name, "#9ca3af")
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{_points(equity[name], 70, 85, 970, 300)}"/>')
    dd = equity.selector / equity.selector.cummax() - 1
    counts = predictions.prediction.value_counts()
    total = max(int(counts.sum()), 1)
    bars, cursor = [], 70
    palette = ["#ff5a36", "#17b890", "#5b7cfa", "#d1d5db"]
    for (name, count), color in zip(counts.items(), palette):
        width = 970 * count / total
        bars.append(f'<rect x="{cursor:.1f}" y="610" width="{width:.1f}" height="38" fill="{color}"/><text x="{cursor + 6:.1f}" y="635" font-size="13">{html.escape(str(name))} {count/total:.0%}</text>')
        cursor += width
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="700" viewBox="0 0 1120 700">
<rect width="1120" height="700" fill="#0f172a"/><style>text{{fill:#e5e7eb;font-family:Arial,sans-serif}} .grid{{stroke:#334155;stroke-width:1}}</style>
<text x="70" y="42" font-size="26" font-weight="bold">LogicAlpha-TM walk-forward report</text>
<text x="70" y="68" font-size="14">Synthetic validation — not evidence of tradable alpha</text>
<line class="grid" x1="70" y1="385" x2="1040" y2="385"/><text x="70" y="105" font-size="15">Growth of $1</text>{''.join(lines)}
<text x="70" y="430" font-size="15">Selector drawdown</text><polyline fill="none" stroke="#fbbf24" stroke-width="2" points="{_points(dd, 70, 450, 970, 115)}"/>
<text x="70" y="595" font-size="15">Out-of-sample strategy selections</text>{''.join(bars)}
<text x="70" y="680" font-size="12">Orange: selector · Green: equal-weight · Blue: SPY</text></svg>'''
    path.write_text(svg, encoding="utf-8")


def write_report(output: Path, summary: dict, predictions: pd.DataFrame, rules: pd.DataFrame) -> None:
    metric_rows = "\n".join(
        f"| {name} | {stats['cagr']:.2%} | {stats['sharpe']:.2f} | {stats['max_drawdown']:.2%} |"
        for name, stats in summary["metrics"].items()
    )
    report = f"""# LogicAlpha-TM generated report

This run used **{summary['data_kind']}** data and **{summary['model']}** as the selector.
Results are out-of-sample across expanding walk-forward folds. They are a software
validation result, not investment evidence.

| Portfolio | CAGR | Sharpe | Max drawdown |
|---|---:|---:|---:|
{metric_rows}

Observations: {summary['observations']}

Walk-forward folds: {summary['folds']}

Decision accuracy (diagnostic): {summary['accuracy']:.2%}

## Interpretation

The appropriate success test is whether the selector consistently beats simple
static/blended alternatives after costs on point-in-time real data. Accuracy alone
does not establish value. Vote margin is not a probability and remains uncalibrated.

See `rules.csv` for fold-specific influential literals and `report.svg` for the visual.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
