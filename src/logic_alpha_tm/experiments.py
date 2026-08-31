from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import ResearchConfig
from .data import load_prices_csv
from .pipeline import run_research


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def load_spec(path: str | Path) -> dict:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"name", "models", "development", "holdout", "robustness"}
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"Experiment specification is missing: {missing}")
    return spec


def validate_availability(prices: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    availability = pd.read_csv(path, parse_dates=["observation_at", "available_at"])
    required = {"observation_at", "available_at", "source", "revision"}
    missing = sorted(required - set(availability.columns))
    if missing:
        raise ValueError(f"Availability metadata is missing: {missing}")
    observations = pd.DatetimeIndex(availability["observation_at"]).tz_localize(None).normalize()
    if observations.has_duplicates or not observations.is_monotonic_increasing:
        raise ValueError("Availability observations must be unique and increasing")
    if not observations.equals(prices.index.normalize()):
        raise ValueError("Availability metadata must contain exactly one row per price observation")
    available_at = pd.to_datetime(availability["available_at"], utc=True)
    next_sessions = prices.index[1:].tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
    if (available_at.iloc[:-1].array >= next_sessions.tz_convert("UTC").array).any():
        raise ValueError("At least one observation was not available before its next-session decision time")
    return availability


def run_benchmark(
    csv_path: str | Path,
    spec_path: str | Path,
    output: str | Path,
    phase: str,
    unlock_holdout: bool = False,
    tmu_platform: str = "CPU",
) -> pd.DataFrame:
    spec = load_spec(spec_path)
    if phase == "holdout" and not unlock_holdout:
        raise ValueError("The final holdout is locked. Re-run once with --unlock-holdout after the specification is frozen.")
    if phase not in ("development", "holdout"):
        raise ValueError("phase must be 'development' or 'holdout'")

    period = spec[phase]
    prices = load_prices_csv(csv_path)
    csv_path = Path(csv_path)
    availability_path = csv_path.with_name(f"{csv_path.stem}.available-at.csv")
    if not availability_path.exists():
        raise ValueError(f"Point-in-time availability file is required: {availability_path}")
    validate_availability(prices, availability_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec_copy = output / "experiment-spec.json"
    spec_copy.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "experiment": spec["name"],
        "evidence_tier": spec.get("evidence_tier", "not-specified"),
        "experiment_commit": git_commit(),
        "phase": phase,
        "prices_file": Path(csv_path).name,
        "prices_sha256": sha256_file(csv_path),
        "availability_file": availability_path.name,
        "availability_sha256": sha256_file(availability_path),
        "spec_sha256": sha256_file(spec_copy),
        "rows": len(prices),
        "first_observation": str(prices.index.min().date()),
        "last_observation": str(prices.index.max().date()),
        "tmu_platform": tmu_platform,
    }
    (output / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    rows = []
    base = replace(ResearchConfig(), tmu_platform=tmu_platform)
    sensitivity = spec["robustness"] if phase == "development" else [{}]
    for model in spec["models"]:
        for run_id, overrides in enumerate(sensitivity):
            config = replace(base, **overrides)
            run_output = output / f"{model}-run-{run_id:02d}"
            summary = run_research(
                prices,
                run_output,
                config,
                model=model,
                data_kind=spec.get(
                    "data_kind", f"licensed point-in-time market data ({phase})"
                ),
                evaluation_start=period["start"],
                evaluation_end=period["end"],
            )
            for portfolio, stats in summary["metrics"].items():
                rows.append({
                    "model": model,
                    "run": run_id,
                    "portfolio": portfolio,
                    "accuracy": summary["accuracy"] if portfolio == "selector" else None,
                    **stats,
                    **{f"config_{key}": value for key, value in overrides.items()},
                })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "comparison.csv", index=False)
    keyed = comparison.set_index(["model", "run", "portfolio"])
    gates = []
    for model, run in comparison[["model", "run"]].drop_duplicates().itertuples(index=False):
        selector = keyed.loc[(model, run, "selector")]
        equal_weight = keyed.loc[(model, run, "equal_weight")]
        gates.append({
            "model": model,
            "run": run,
            "selector_sharpe": selector["sharpe"],
            "equal_weight_sharpe": equal_weight["sharpe"],
            "sharpe_difference": selector["sharpe"] - equal_weight["sharpe"],
            "primary_gate_passed": bool(selector["sharpe"] > equal_weight["sharpe"]),
        })
    gate_frame = pd.DataFrame(gates)
    gate_frame.to_csv(output / "decision-gates.csv", index=False)
    passes = int(gate_frame.primary_gate_passed.sum())
    report = f"""# Frozen benchmark summary

Experiment: **{spec['name']}**  
Phase: **{phase}**  
Input fingerprint: `{manifest['prices_sha256']}`

The pre-registered primary gate passed in **{passes} of {len(gate_frame)}**
model/sensitivity runs. This count is descriptive; inspect `comparison.csv` and
the individual run folders before drawing a conclusion. A backtest is not
investment advice or evidence of future performance.
"""
    (output / "BENCHMARK.md").write_text(report, encoding="utf-8")
    return comparison
