"""Versioned development audit: asset accounting and a conservative risk filter.

The v0.2 runner is retained for provenance. This engine never opens the holdout.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_prices_csv
from .experiments import git_commit, sha256_file, validate_availability
from .features import QuantileBooleanEncoder, build_features
from .models import BernoulliSelector, LogisticSelector, TMUSelector


@dataclass(frozen=True)
class AuditConfig:
    horizon: int = 20
    execution_lag: int = 2
    min_train: int = 504
    test_size: int = 126
    rebalance_every: int = 5
    reduction: float = 0.5
    dead_zone: float = 0.003
    lambda_vol: float = 0.15
    lambda_drawdown: float = 0.20
    cost_bps: float = 2.0
    seed: int = 7
    clauses: int = 200
    epochs: int = 10
    tmu_platform: str = "CPU"

    def __post_init__(self):
        for name in ("horizon", "min_train", "test_size", "rebalance_every", "clauses", "epochs"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.execution_lag < 2:
            raise ValueError("After-close information requires at least two sessions to its first close-to-close return")
        if not 0 <= self.reduction < 1:
            raise ValueError("reduction must be in [0, 1)")
        if min(self.cost_bps, self.dead_zone, self.lambda_vol, self.lambda_drawdown) < 0:
            raise ValueError("Costs, dead zone and risk penalties cannot be negative")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def target_weights(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Targets known after today's close, before any execution delay."""
    zero = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    ma20, ma100 = prices.SPY.rolling(20).mean(), prices.SPY.rolling(100).mean()
    trend = zero.copy()
    trend["SPY"] = (ma20 > ma100).astype(float)
    scores = prices[["SPY", "QQQ", "IWM"]].pct_change(60)
    leader = scores.fillna(-np.inf).idxmax(axis=1)
    momentum = zero.copy()
    for asset in scores.columns:
        momentum[asset] = ((leader == asset) & scores.notna().all(axis=1)).astype(float)
    defensive = zero.copy()
    defensive["SPY"] = (prices.SPY > ma100).astype(float)
    defensive["TLT"] = ((prices.SPY <= ma100) & ma100.notna()).astype(float)
    spy = zero.copy()
    spy["SPY"] = 1.0
    return {"trend": trend, "momentum": momentum, "defensive": defensive,
            "blend": (trend + momentum + defensive) / 3, "SPY": spy}


def account(prices: pd.DataFrame, targets: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """Daily target-weight rebalancing, cash earns zero, costs per risky dollar traded.

    A target formed after close t is executed at close t+1 and first earns
    the return ending at t+2. Portfolio weights drift before the next trade.
    Turnover is an approximation relative to pre-cost wealth; costs are applied
    before the following return. No extra strategy-name switching fee is charged.
    """
    returns = prices.pct_change().fillna(0.0)
    weights = targets.reindex_like(prices).shift(config.execution_lag).fillna(0.0)
    if (weights < 0).any().any() or (weights.sum(axis=1) > 1 + 1e-10).any():
        raise ValueError("Only unlevered, long-only weights are supported")
    gross = (weights * returns).sum(axis=1)
    drifted = weights.shift(1).fillna(0) * (1 + returns.shift(1).fillna(0))
    drifted = drifted.div(1 + gross.shift(1).fillna(0), axis=0)
    turnover = (weights - drifted).abs().sum(axis=1)
    cost = turnover * config.cost_bps / 10_000
    net = (1 - cost) * (1 + gross) - 1
    return pd.DataFrame({"return": net, "gross_return": gross,
                         "turnover": turnover, "cost": cost,
                         "exposure": weights.sum(axis=1)})


def drawdown(returns: np.ndarray) -> float:
    wealth = np.r_[1.0, np.cumprod(1 + returns)]
    return float(np.min(wealth / np.maximum.accumulate(wealth) - 1))


def audit_metrics(returns: pd.Series) -> dict:
    values = returns.to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Metrics require finite, nonempty returns")
    wealth = float(np.prod(1 + values))
    vol = float(np.std(values, ddof=1) * np.sqrt(252)) if len(values) > 1 else 0.0
    mean = float(np.mean(values) * 252)
    downside = float(np.sqrt(np.mean(np.minimum(values, 0) ** 2)) * np.sqrt(252))
    return {"cagr": wealth ** (252 / len(values)) - 1,
            "annual_volatility": vol, "sharpe": mean / vol if vol else 0.0,
            "sortino": mean / downside if downside else 0.0,
            "max_drawdown": drawdown(values), "total_return": wealth - 1}


def risk_labels(blend: pd.Series, reduced: pd.Series, config: AuditConfig) -> pd.Series:
    """Reduce only when reduced exposure's horizon utility clearly beats the blend."""
    labels = pd.Series(None, index=blend.index, dtype=object, name="actual_label")
    for i in range(len(blend) - config.execution_lag - config.horizon + 1):
        start = i + config.execution_lag
        stop = start + config.horizon
        utilities = []
        for stream in (blend, reduced):
            values = stream.iloc[start:stop].to_numpy()
            ret = np.prod(1 + values) - 1
            horizon_vol = np.std(values, ddof=0) * np.sqrt(config.horizon)
            utilities.append(ret - config.lambda_vol * horizon_vol
                             - config.lambda_drawdown * abs(drawdown(values)))
        labels.iloc[i] = "reduced" if utilities[1] - utilities[0] > config.dead_zone else "blend"
    return labels


def predictions_for(
    features: pd.DataFrame, labels: pd.Series, start: str, model: str,
    config: AuditConfig, directory: Path, identity: str, resume: bool,
) -> pd.DataFrame:
    x = features.drop(columns="regime").dropna()
    first = max(config.min_train, int(x.index.searchsorted(pd.Timestamp(start))))
    folds = list(range(first, len(x), config.test_size))
    if not folds:
        raise ValueError("No evaluation folds")
    blocks = []
    for fold, left in enumerate(folds):
        started = time.monotonic()
        test = x.iloc[left:left + config.test_size]
        # Last label outcome must be strictly before this test decision date.
        train_stop = left - (config.horizon + config.execution_lag - 1)
        train = x.iloc[:max(0, train_stop)]
        y = labels.reindex(train.index).dropna()
        train = train.loc[y.index]
        checkpoint = directory / f"fold-{fold:03d}.json"
        dates = [d.isoformat() for d in test.index]
        if resume and checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if saved["identity"] != identity or saved["dates"] != dates:
                raise ValueError("Checkpoint does not match this experiment")
            predicted = saved["predictions"]
            if len(predicted) != len(test) or not set(predicted) <= {"blend", "reduced"}:
                raise ValueError("Invalid checkpoint predictions")
            log(f"{model} seed={config.seed} fold {fold+1}/{len(folds)} restored")
        else:
            log(f"{model} seed={config.seed} fold {fold+1}/{len(folds)} "
                f"training={len(train)} test={test.index[0].date()}..{test.index[-1].date()}")
            if y.empty:
                raise ValueError("No matured training labels")
            if y.nunique() == 1:
                predicted = [y.iloc[0]] * len(test)
            else:
                encoder = QuantileBooleanEncoder().fit(train)
                selectors = {"bernoulli": lambda: BernoulliSelector(),
                             "logistic": lambda: LogisticSelector(config.seed),
                             "tmu": lambda: TMUSelector(clauses=config.clauses, epochs=config.epochs,
                                                        platform=config.tmu_platform, seed=config.seed + fold,
                                                        progress=log)}
                selector = selectors[model]()
                selector.fit(encoder.transform(train), y)
                predicted = selector.predict_with_margin(encoder.transform(test))[0].tolist()
            atomic_json(checkpoint, {"identity": identity, "dates": dates,
                                    "predictions": predicted})
            log(f"  fold completed in {time.monotonic()-started:.1f}s")
        blocks.append(pd.DataFrame({"prediction": predicted, "fold": fold}, index=test.index))
    return pd.concat(blocks)


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_audit(csv: str, spec_path: str, output: str, resume: bool = False) -> pd.DataFrame:
    started = time.monotonic()
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if spec.get("engine") != "risk-audit-v0.3":
        raise ValueError("Expected risk-audit-v0.3 specification")
    if not spec["models"] or not set(spec["models"]) <= {"bernoulli", "logistic", "tmu"}:
        raise ValueError("Unknown or empty model list")
    if not spec["seeds"] or len(set(spec["seeds"])) != len(spec["seeds"]):
        raise ValueError("Seeds must be nonempty and unique")
    base_config = AuditConfig(**spec["config"])
    if (not spec["evaluation_cost_bps"] or min(spec["evaluation_cost_bps"]) < 0
            or base_config.cost_bps not in spec["evaluation_cost_bps"]):
        raise ValueError("Evaluation costs must include the training cost and be nonnegative")
    if "tmu" in spec["models"]:
        from tmu.models.classification.vanilla_classifier import TMClassifier  # noqa: F401
    period = spec["development"]
    if pd.Timestamp(period["end"]) >= pd.Timestamp(spec["holdout"]["start"]):
        raise ValueError("Development overlaps the holdout")
    prices_all = load_prices_csv(csv)
    availability_path = Path(csv).with_name(Path(csv).stem + ".available-at.csv")
    validate_availability(prices_all, availability_path)
    # Cut BEFORE any features, returns, labels, diagnostics, or model selection.
    prices = prices_all.loc[:period["end"]].copy()
    packages = {}
    for name in ("numpy", "pandas", "tmu", "scipy", "scikit-learn"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not-installed"
    manifest = {"engine": spec["engine"], "spec": spec,
                "prices_sha256": sha256_file(csv), "availability_sha256": sha256_file(availability_path),
                "source_sha256": source_fingerprint(), "python": platform.python_version(),
                "packages": packages, "platform": platform.platform()}
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "run-manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume or previous["identity"] != identity:
            raise ValueError("Existing output: use --resume with identical inputs/code or choose a new folder")
    elif any(directory.iterdir()):
        raise ValueError("Output directory must be empty for a new audit")
    atomic_json(manifest_path, {**manifest, "identity": identity, "commit": git_commit(),
                               "last_used_observation": str(prices.index[-1].date()),
                               "cash_and_risk_free_return": "zero; cash-yield sensitivity remains outstanding"})
    features = build_features(prices)
    targets = target_weights(prices)
    rows = []
    for model in spec["models"]:
        for seed in (spec["seeds"] if model == "tmu" else [spec["seeds"][0]]):
            config = AuditConfig(**{**spec["config"], "seed": seed})
            run_dir = directory / f"{model}-seed-{seed}"
            run_dir.mkdir(exist_ok=True)
            base = account(prices, targets["blend"], config)
            half = account(prices, targets["blend"] * config.reduction, config)
            labels = risk_labels(base["return"], half["return"], config)
            prediction = predictions_for(features, labels, period["start"], model,
                                         config, run_dir, identity, resume)
            held = prediction.prediction.where(np.arange(len(prediction)) % config.rebalance_every == 0).ffill()
            scale = held.map({"blend": 1.0, "reduced": config.reduction}).reindex(prices.index).fillna(1.0)
            portfolios = {"selector": targets["blend"].mul(scale, axis=0),
                          "equal_weight": targets["blend"], "constant_half": targets["blend"] * config.reduction,
                          "always_defensive": targets["defensive"], "SPY": targets["SPY"]}
            daily = prices.pct_change()
            stress = (prices.SPY.pct_change(60) < 0) & (daily.SPY.rolling(20).std() > daily.SPY.rolling(60).std())
            portfolios["simple_risk_filter"] = targets["blend"].mul(
                pd.Series(np.where(stress, config.reduction, 1.0), index=prices.index), axis=0)
            first_return = prices.index[prices.index.get_loc(prediction.index[0]) + config.execution_lag]
            streams = {}
            for cost in spec["evaluation_cost_bps"]:
                cost_config = AuditConfig(**{**asdict(config), "cost_bps": cost})
                for name, weights in portfolios.items():
                    ledger = account(prices, weights, cost_config).loc[first_return:]
                    values = ledger["return"]
                    rows.append({"model": model, "seed": seed, "cost_bps": cost, "portfolio": name,
                                 "start": str(values.index[0].date()), "end": str(values.index[-1].date()),
                                 "observations": len(values), **audit_metrics(values),
                                 "mean_exposure": float(ledger.exposure.mean()),
                                 "annual_turnover": float(ledger.turnover.mean() * 252)})
                    if cost == config.cost_bps:
                        streams[name] = values
                        ledger.to_csv(run_dir / f"{name}-ledger.csv", index_label="date")
            prediction["held_action"] = held
            prediction["actual_label"] = labels.reindex(prediction.index)
            prediction.to_csv(run_dir / "predictions.csv", index_label="date")
            scored = prediction.dropna(subset=["actual_label"])
            atomic_json(run_dir / "diagnostics.json", {
                "config": asdict(config), "scored_observations": len(scored),
                "accuracy": float((scored.prediction == scored.actual_label).mean()) if len(scored) else None,
                "reduce_fraction": float((held == "reduced").mean()),
                "label_fractions": scored.actual_label.value_counts(normalize=True).to_dict(),
                "execution": "observe after close t, fill close t+1, first return ends t+2",
                "decision_rule": "default blend; reduce to 50% when predicted reduced; no probability claim"})
            pd.DataFrame(streams).to_csv(run_dir / "returns.csv", index_label="date")
            pd.DataFrame(rows).to_csv(directory / "comparison.partial.csv", index=False)
            log(f"Completed {model} seed={seed}; elapsed {time.monotonic()-started:.1f}s")
    comparison = pd.DataFrame(rows)
    comparison.to_csv(directory / "comparison.csv", index=False)
    gates = []
    for (model, seed, cost), group in comparison.groupby(["model", "seed", "cost_bps"]):
        keyed = group.set_index("portfolio")
        selected = keyed.loc["selector"]
        blend = keyed.loc["equal_weight"]
        simple = keyed.loc["simple_risk_filter"]
        gates.append({"model": model, "seed": int(seed), "cost_bps": float(cost),
                      "sharpe_difference": float(selected.sharpe - blend.sharpe),
                      "passes": bool(selected.sharpe > max(blend.sharpe, simple.sharpe)
                                     and selected.max_drawdown >= blend.max_drawdown)})
    atomic_json(directory / "decision-gates.json", {"rule": spec.get("decision_gate", ""), "runs": gates})
    tmu_gates = [g for g in gates if g["model"] == "tmu"]
    advance = bool(tmu_gates) and all(g["passes"] for g in tmu_gates)
    display = comparison[comparison.cost_bps == base_config.cost_bps]
    lines = ["# Development risk-filter audit", "", f"Advance TMU pilot: **{advance}**.", "",
             "Sharpe uses arithmetic daily mean / sample standard deviation, annualized; risk-free and cash returns are zero.",
             "Original v0.2 outputs use different timing, labels and metrics and are not directly comparable.", "",
             "| Model | Seed | Portfolio | CAGR | Sharpe | Max drawdown | Mean exposure |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for row in display.itertuples():
        lines.append(f"| {row.model} | {row.seed} | {row.portfolio} | {row.cagr:.2%} | "
                     f"{row.sharpe:.3f} | {row.max_drawdown:.2%} | {row.mean_exposure:.1%} |")
    lines.extend(["", "All seed/cost results are in comparison.csv. Cash-yield sensitivity, "
                  "realistic next-open fills, and clause-level explanations remain future work.",
                  "Development only; final holdout remains locked."])
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(directory / "completed.json", {"identity": identity, "rows": len(comparison),
                                               "elapsed_seconds": time.monotonic()-started})
    log(f"Development audit completed: {len(comparison)} portfolio/cost rows. Holdout remains locked.")
    return comparison
