"""LogicVol Experiment A: Regression Tsetlin Machine vs volatility baselines.

Implements experiments/logicvol-v0.1.json. Every number the decision rule
uses comes from walk-forward test folds inside the development period; prices
after the development end are dropped before any feature or target exists.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform as _platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_prices_csv
from .experiments import git_commit, sha256_file
from .features import QuantileBooleanEncoder

VAR_FLOOR = 1e-12
DECILES = tuple(round(0.1 * k, 1) for k in range(1, 10))
BASELINES = ("historical_22", "ewma", "har_style", "ridge_boolean")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def load_spec(path: str | Path) -> dict:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("engine") != "logicvol-v0.1":
        raise ValueError("Not a logicvol-v0.1 specification")
    return spec


# ---------------------------------------------------------------- data, target, features
def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))


def log_vol(squared: pd.Series, window: int) -> pd.Series:
    """0.5 * log of annualised variance from the mean of the last `window` squared returns."""
    var = squared.rolling(window, min_periods=window).mean() * 252
    return 0.5 * np.log(var.clip(lower=VAR_FLOOR))


def target(returns: pd.Series, horizon: int) -> pd.Series:
    """Log annualised volatility over the next `horizon` sessions (t+1 .. t+horizon)."""
    future = (returns ** 2).rolling(horizon, min_periods=horizon).mean().shift(-horizon) * 252
    return 0.5 * np.log(future.clip(lower=VAR_FLOOR))


def build_features(prices: pd.DataFrame, asset: str) -> pd.DataFrame:
    r = log_returns(prices)
    sq = r ** 2
    p = prices[asset]
    lv = {k: log_vol(sq[asset], k) for k in (1, 5, 22, 66)}
    out = {
        "lv1": lv[1], "lv5": lv[5], "lv22": lv[22], "lv66": lv[66],
        "lv5_minus_lv66": lv[5] - lv[66],
        "lv22_minus_lv66": lv[22] - lv[66],
        "ret5": p / p.shift(5) - 1,
        "ret22": p / p.shift(22) - 1,
        "drawdown100": p / p.rolling(100, min_periods=100).max() - 1,
        "absret1": r[asset].abs(),
        "tlt_lv22": log_vol(sq["TLT"], 22),
        "spy_tlt_corr60": r["SPY"].rolling(60, min_periods=60).corr(r["TLT"]),
        "iwm_minus_spy_lv22": log_vol(sq["IWM"], 22) - log_vol(sq["SPY"], 22),
    }
    return pd.DataFrame(out, index=prices.index)


def ewma_log_vol(returns: pd.Series, lam: float = 0.94, seed_window: int = 22) -> pd.Series:
    """RiskMetrics variance, flat over the horizon, as log annualised volatility."""
    sq = (returns ** 2).to_numpy()
    out = np.full(len(sq), np.nan)
    if len(sq) <= seed_window:
        return pd.Series(out, index=returns.index)
    var = float(np.nanmean(sq[1:seed_window + 1]))
    for t in range(seed_window + 1, len(sq)):
        var = lam * var + (1 - lam) * sq[t]
        out[t] = 0.5 * math.log(max(var * 252, VAR_FLOOR))
    return pd.Series(out, index=returns.index)


def frame_for(prices: pd.DataFrame, asset: str, horizon: int) -> pd.DataFrame:
    r = log_returns(prices)
    feats = build_features(prices, asset)
    frame = feats.assign(
        y=target(r[asset], horizon),
        historical_22=feats["lv22"],
        ewma=ewma_log_vol(r[asset]),
    )
    return frame.dropna()


# ---------------------------------------------------------------- folds
def folds(dates: pd.DatetimeIndex, dev_start: str, min_train: int, test_size: int, purge: int):
    first = max(int(dates.searchsorted(pd.Timestamp(dev_start))), min_train + purge)
    out = []
    for start in range(first, len(dates), test_size):
        train_stop = start - purge
        out.append((np.arange(train_stop), np.arange(start, min(start + test_size, len(dates)))))
    if not out:
        raise ValueError("No walk-forward folds: too little history before the development start")
    return out


# ---------------------------------------------------------------- models
def ols(x: np.ndarray, y: np.ndarray, alpha: float = 0.0) -> tuple[float, np.ndarray]:
    """Least squares with an unpenalised intercept; ridge when alpha > 0."""
    xm, ym = x.mean(axis=0), y.mean()
    xc, yc = x - xm, y - ym
    coef = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ yc)
    return float(ym - xm @ coef), coef


def inner_split(train: pd.DataFrame, purge: int, fraction: float = 0.2):
    """Last `fraction` of the training window held back, with a purge gap before it."""
    n_val = max(1, int(math.ceil(fraction * len(train))))
    return train.iloc[: len(train) - n_val - purge], train.iloc[len(train) - n_val:]


def lognormal_shift(y_val: np.ndarray, yhat_val: np.ndarray) -> float:
    """For a model of mean log-volatility: E[vol^2] = exp(2*mu + 2*sigma^2), so the variance
    forecast uses yhat + sigma^2, with sigma^2 the out-of-sample residual variance."""
    return float(np.var(y_val - yhat_val))


HAR_COLS = ["lv1", "lv5", "lv22"]


def har_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    intercept, coef = ols(train[HAR_COLS].to_numpy(), train["y"].to_numpy())
    return intercept + test[HAR_COLS].to_numpy() @ coef


def boolean_matrices(train_x: pd.DataFrame, other_x: list[pd.DataFrame]):
    encoder = QuantileBooleanEncoder(DECILES).fit(train_x)
    return encoder.transform(train_x), [encoder.transform(o) for o in other_x]


def ridge_forecast(btrain: pd.DataFrame, ytrain: np.ndarray, btest: pd.DataFrame, alpha: float) -> np.ndarray:
    intercept, coef = ols(btrain.to_numpy(dtype=float), ytrain, alpha)
    return intercept + btest.to_numpy(dtype=float) @ coef


def fit_tm(bx: pd.DataFrame, y: np.ndarray, cand: dict, epochs: int, seed: int, platform: str, weighted: bool):
    from tmu.models.regression.vanilla_regressor import TMRegressor

    model = TMRegressor(cand["clauses"], cand["T"], cand["s"], platform=platform,
                        weighted_clauses=weighted, seed=seed)
    x = bx.to_numpy(dtype=np.uint32)
    for _ in range(epochs):
        model.fit(x, y.astype(np.float64))
    return model


def tm_predict(model, bx: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(bx.to_numpy(dtype=np.uint32)), dtype=float)


# ---------------------------------------------------------------- metrics
def qlike(y: np.ndarray, yhat: np.ndarray) -> float:
    ratio = np.exp(2 * y) / np.exp(2 * yhat)
    return float(np.mean(ratio - np.log(ratio) - 1))


def metrics(y: np.ndarray, yhat: np.ndarray, yhat_var: np.ndarray) -> dict:
    """QLIKE scores the variance forecast (yhat_var); the other metrics score the log forecast."""
    a, b = ols(yhat.reshape(-1, 1), y)
    top_y, top_f = y >= np.quantile(y, 0.9), yhat >= np.quantile(yhat, 0.9)
    return {
        "qlike": qlike(y, yhat_var),
        "qlike_uncorrected": qlike(y, yhat),
        "mse_log_vol": float(np.mean((y - yhat) ** 2)),
        "mae_annual_vol": float(np.mean(np.abs(np.exp(y) - np.exp(yhat)))),
        "mz_intercept": a,
        "mz_slope": float(b[0]),
        "top_decile_recall": float((top_f & top_y).sum() / max(top_y.sum(), 1)),
        "observations": int(len(y)),
    }


def decision(asset_metrics: dict, seeds: list[int]) -> dict:
    tm = [asset_metrics[f"tm_seed{s}"]["qlike"] for s in seeds]
    med = float(np.median(tm))
    hist, ewma, har = (asset_metrics[k]["qlike"] for k in ("historical_22", "ewma", "har_style"))
    checks = {
        "beats_historical_and_ewma": med < hist and med < ewma,
        "median_within_5pct_of_har": med <= 1.05 * har,
        "every_seed_within_10pct_of_har": all(q <= 1.10 * har for q in tm),
    }
    return {"tm_median_qlike": med, "har_qlike": har, "historical_qlike": hist, "ewma_qlike": ewma,
            "checks": checks, "status": "pass" if all(checks.values()) else "fail"}


# ---------------------------------------------------------------- identity
def packages() -> dict:
    out = {}
    for name in ("numpy", "pandas", "tmu"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not-installed"
    return out


def source_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def cut_to_development(csv: str | Path, spec_path: str | Path, output: str | Path) -> dict:
    """Write a copy of the prices that ends at the development end, so holdout prices never travel."""
    spec = load_spec(spec_path)
    prices = load_prices_csv(csv).loc[: spec["evaluation"]["development"]["end"]]
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    prices.rename_axis("date").to_csv(out)
    return {"rows": len(prices), "first": str(prices.index[0].date()), "last": str(prices.index[-1].date()),
            "output": str(out), "sha256": sha256_file(out)}


# ---------------------------------------------------------------- runner
def run(
    csv: str | Path,
    spec_path: str | Path,
    output: str | Path,
    assets: list[str] | None = None,
    resume: bool = False,
    platform: str = "CPU",
    quick: bool = False,
    skip_tm: bool = False,
) -> dict:
    spec = load_spec(spec_path)
    ev, tms = spec["evaluation"], spec["tm_settings"]
    dev = ev["development"]
    if pd.Timestamp(dev["end"]) >= pd.Timestamp(ev["holdout"]["start"]):
        raise ValueError("Development overlaps the holdout")
    horizon = spec["target"]["horizon_sessions"]
    primary = spec["data"]["primary_asset"]
    assets = assets or [primary, *spec["data"]["robustness_assets"]]
    seeds = list(tms["seeds"])
    candidates = list(tms["candidates"])
    epochs = int(tms["epochs"])
    weighted = bool(tms["weighted_clauses"])
    ridge_alpha = float(spec["implementation"]["ridge_alpha"])
    purge = int(ev["purge_sessions"])
    max_folds = None
    if quick:
        seeds, candidates, epochs, max_folds = seeds[:1], candidates[:1], 2, 3

    prices_all = load_prices_csv(csv)
    prices = prices_all.loc[: dev["end"]].copy()  # cut BEFORE any feature, target or model

    manifest = {
        "engine": spec["engine"], "spec_sha256": sha256_file(spec_path), "prices_sha256": sha256_file(csv),
        "source_sha256": source_fingerprint(), "packages": packages(), "python": _platform.python_version(),
        "platform": platform, "assets": assets, "quick": quick, "skip_tm": skip_tm,
    }
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "run-manifest.json"
    if mpath.exists():
        previous = json.loads(mpath.read_text(encoding="utf-8"))
        if not resume or previous.get("identity") != identity:
            raise ValueError("Existing output: use --resume with identical inputs and code, or choose a new folder")
    elif any(out.iterdir()):
        raise ValueError("Output directory must be empty for a new run")
    atomic_json(mpath, {**manifest, "identity": identity, "commit": git_commit(),
                        "first_price": str(prices.index[0].date()), "last_price_used": str(prices.index[-1].date()),
                        "rows_dropped_after_development": int(len(prices_all) - len(prices))})

    all_metrics, selections = {}, {}
    for asset in assets:
        frame = frame_for(prices, asset, horizon)
        frame = frame.loc[frame.index <= pd.Timestamp(dev["end"])]
        feature_cols = [c for c in frame.columns if c not in ("y", "historical_22", "ewma")]
        fl = folds(frame.index, dev["start"], ev["walk_forward_min_train"], ev["walk_forward_test_size"], purge)
        if max_folds:
            fl = fl[:max_folds]
        log(f"{asset}: {len(frame)} rows, {len(fl)} folds, test {frame.index[fl[0][1][0]].date()} .. {frame.index[fl[-1][1][-1]].date()}")

        preds = []
        for k, (tr, te) in enumerate(fl):
            train, test = frame.iloc[tr], frame.iloc[te]
            inner_tr, inner_val = inner_split(train, purge)
            btrain, (btest,) = boolean_matrices(train[feature_cols], [test[feature_cols]])
            b_in, (b_val,) = boolean_matrices(inner_tr[feature_cols], [inner_val[feature_cols]])
            shift_har = lognormal_shift(inner_val["y"].to_numpy(), har_forecast(inner_tr, inner_val))
            shift_ridge = lognormal_shift(inner_val["y"].to_numpy(),
                                          ridge_forecast(b_in, inner_tr["y"].to_numpy(), b_val, ridge_alpha))
            block = pd.DataFrame({"fold": k, "y": test["y"], "historical_22": test["historical_22"],
                                  "ewma": test["ewma"], "har_style": har_forecast(train, test),
                                  "ridge_boolean": ridge_forecast(btrain, train["y"].to_numpy(), btest, ridge_alpha)},
                                 index=test.index)
            block["har_style_var"] = block["har_style"] + shift_har
            block["ridge_boolean_var"] = block["ridge_boolean"] + shift_ridge
            if not skip_tm:
                for seed in seeds:
                    pred_tm, sel = tm_fold(out, identity, asset, k, seed, train, test, feature_cols,
                                           candidates, epochs, platform, weighted, purge, resume)
                    block[f"tm_seed{seed}"] = pred_tm
                    block[f"tm_seed{seed}_var"] = pred_tm + sel["lognormal_shift"]
                    selections.setdefault(asset, {}).setdefault(f"seed{seed}", []).append(sel)
            preds.append(block)
        pred = pd.concat(preds)
        pred.to_csv(out / f"predictions-{asset}.csv", index_label="date")
        models = list(BASELINES) + ([f"tm_seed{s}" for s in seeds] if not skip_tm else [])
        y = pred["y"].to_numpy()
        all_metrics[asset] = {
            m: metrics(y, pred[m].to_numpy(), pred[f"{m}_var" if f"{m}_var" in pred else m].to_numpy())
            for m in models
        }

    result = {"metrics": all_metrics, "tm_selection": selections}
    if quick or skip_tm:
        result["decision"] = {"status": "not_evaluated", "reason": "quick smoke test" if quick else "TM skipped"}
    else:
        result["decision"] = decision(all_metrics[primary], seeds)
    atomic_json(out / "metrics.json", result)
    (out / "REPORT.md").write_text(report(spec, result, assets, seeds if not skip_tm else [], quick, skip_tm), encoding="utf-8")
    return result


def tm_fold(out, identity, asset, k, seed, train, test, cols, candidates, epochs, platform, weighted, purge, resume):
    ck = out / "checkpoints" / f"{asset}-seed{seed}-fold{k:03d}.json"
    dates = [d.isoformat() for d in test.index]
    if resume and ck.exists():
        saved = json.loads(ck.read_text(encoding="utf-8"))
        if saved["identity"] != identity or saved["dates"] != dates:
            raise ValueError(f"Checkpoint {ck.name} does not match this run")
        log(f"{asset} seed {seed} fold {k}: restored")
        return np.asarray(saved["predictions"]), saved["selection"]

    started = time.monotonic()
    fold_seed = seed + k
    inner_train, inner_val = inner_split(train, purge)
    y_val = inner_val["y"].to_numpy()
    b_in, (b_val,) = boolean_matrices(inner_train[cols], [inner_val[cols]])
    scores, shifts = [], []
    for c in candidates:
        model = fit_tm(b_in, inner_train["y"].to_numpy(), c, epochs, fold_seed, platform, weighted)
        val_pred = tm_predict(model, b_val)
        shifts.append(lognormal_shift(y_val, val_pred))
        scores.append(qlike(y_val, val_pred + shifts[-1]))
    best = int(np.argmin(scores))
    b_full, (b_test,) = boolean_matrices(train[cols], [test[cols]])
    model = fit_tm(b_full, train["y"].to_numpy(), candidates[best], epochs, fold_seed, platform, weighted)
    pred = tm_predict(model, b_test)
    selection = {"fold": k, "chosen": candidates[best], "validation_qlike": scores,
                 "lognormal_shift": shifts[best], "seconds": round(time.monotonic() - started, 1)}
    ck.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(ck, {"identity": identity, "dates": dates, "predictions": pred.tolist(), "selection": selection})
    log(f"{asset} seed {seed} fold {k}: chose {candidates[best]} in {selection['seconds']}s")
    return pred, selection


def report(spec: dict, result: dict, assets: list[str], seeds: list[int], quick: bool, skip_tm: bool) -> str:
    lines = [f"# {spec['name']}", ""]
    if quick:
        lines += ["> **SMOKE TEST ONLY.** Shortened run (1 seed, 1 setting, 2 epochs, 3 folds). Not a valid result.", ""]
    if skip_tm:
        lines += ["> Baselines only; the Tsetlin Machine was skipped. No decision is made.", ""]
    dec = result["decision"]
    if dec["status"] in ("pass", "fail"):
        lines += [f"## Decision ({spec['data']['primary_asset']}): **{dec['status'].upper()}**", ""]
        for name, ok in dec["checks"].items():
            lines.append(f"- {'PASS' if ok else 'FAIL'}: {name.replace('_', ' ')}")
        lines += ["", f"TM median QLIKE {dec['tm_median_qlike']:.4f}; HAR-style {dec['har_qlike']:.4f}; "
                      f"historical {dec['historical_qlike']:.4f}; EWMA {dec['ewma_qlike']:.4f}.", ""]
    lines += ["Lower QLIKE and MSE are better. Every seed is reported; no seed was selected.",
              "QLIKE scores variance forecasts. Models fit to mean log-volatility (har_style, ridge_boolean, tm) "
              "add their out-of-sample residual variance (log-normal correction) before QLIKE; "
              "historical_22 and ewma estimate variance directly. The uncorrected column is shown for transparency.", ""]
    for asset in assets:
        m = result["metrics"][asset]
        lines += [f"## {asset}", "",
                  "| model | QLIKE | QLIKE uncorrected | MSE log vol | MAE annual vol | MZ slope | top-decile recall |",
                  "|---|--:|--:|--:|--:|--:|--:|"]
        for name, v in m.items():
            lines.append(f"| {name} | {v['qlike']:.4f} | {v['qlike_uncorrected']:.4f} | {v['mse_log_vol']:.4f} | "
                         f"{v['mae_annual_vol']:.4f} | {v['mz_slope']:.3f} | {v['top_decile_recall']:.3f} |")
        lines += ["", f"Test observations: {next(iter(m.values()))['observations']}", ""]
    lines += ["Data: Tiingo current-vintage adjusted closes, internal use only. Prices after the development end "
              "were dropped before any calculation. The holdout was not touched."]
    return "\n".join(lines) + "\n"
