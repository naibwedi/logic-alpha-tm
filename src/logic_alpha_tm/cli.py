import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import ResearchConfig
from .data import load_prices_csv, synthetic_prices
from .experiments import run_benchmark, sha256_file
from .pipeline import run_research
from .providers import download_massive_prices, download_tiingo_prices


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="LogicAlpha-TM research runner")
    sub = root.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run deterministic synthetic validation")
    demo.add_argument("--output", default="results")
    demo.add_argument("--model", choices=("bernoulli", "logistic", "boosted_tree", "tmu"), default="bernoulli")
    run = sub.add_parser("run", help="run on a wide adjusted-close CSV")
    run.add_argument("--csv", required=True)
    run.add_argument("--output", default="results")
    run.add_argument("--model", choices=("bernoulli", "logistic", "boosted_tree", "tmu"), default="bernoulli")
    benchmark = sub.add_parser("benchmark", help="run a frozen multi-model development or holdout benchmark")
    benchmark.add_argument("--csv", required=True)
    benchmark.add_argument("--spec", default="experiments/real-market-v0.2.json")
    benchmark.add_argument("--phase", choices=("development", "holdout"), default="development")
    benchmark.add_argument("--output", default="results/real-benchmark")
    benchmark.add_argument("--unlock-holdout", action="store_true")
    benchmark.add_argument(
        "--tmu-platform",
        choices=("CPU", "CUDA"),
        default="CPU",
        help="TMU execution backend; use CUDA only on a configured NVIDIA runtime",
    )
    download = sub.add_parser("download-massive", help="download licensed unadjusted daily bars")
    download.add_argument("--start", required=True, help="inclusive date (YYYY-MM-DD)")
    download.add_argument("--end", required=True, help="inclusive date (YYYY-MM-DD)")
    download.add_argument("--output", default="data/raw/massive-prices.csv")
    download.add_argument("--vendor-plan", default="not-recorded", help="non-secret licensed plan label for the manifest")
    tiingo = sub.add_parser("download-tiingo", help="download Tiingo EOD prices for internal research")
    tiingo.add_argument("--start", required=True, help="inclusive date (YYYY-MM-DD)")
    tiingo.add_argument("--end", required=True, help="inclusive date (YYYY-MM-DD)")
    tiingo.add_argument("--output", default="data/raw/tiingo-prices.csv")
    tiingo.add_argument("--allow-partial-history", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "download-tiingo":
        api_token = os.environ.get("TIINGO_API_TOKEN")
        if not api_token:
            raise SystemExit(
                "TIINGO_API_TOKEN is not set. Set it locally; never commit or paste the token into a report."
            )
        adjusted, raw, actions, availability = download_tiingo_prices(
            args.start, args.end, api_token
        )
        requested_start = pd.Timestamp(args.start)
        requested_end = pd.Timestamp(args.end)
        tolerance = pd.Timedelta(days=10)
        if not args.allow_partial_history and (
            adjusted.index.min() > requested_start + tolerance
            or adjusted.index.max() < requested_end - tolerance
        ):
            raise SystemExit(
                "Tiingo returned partial history: "
                f"{adjusted.index.min().date()} through {adjusted.index.max().date()}. "
                "Check ticker entitlement or use --allow-partial-history deliberately."
            )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        raw_path = output.with_name(f"{output.stem}.raw.csv")
        actions_path = output.with_name(f"{output.stem}.corporate-actions.csv")
        availability_path = output.with_name(f"{output.stem}.available-at.csv")
        adjusted.rename_axis("date").to_csv(output)
        raw.rename_axis("date").to_csv(raw_path)
        actions.to_csv(actions_path, index=False)
        availability.rename_axis("observation_at").to_csv(availability_path)
        manifest = {
            "prices": str(output),
            "price_basis": "Tiingo current-vintage adjusted close",
            "prices_sha256": sha256_file(output),
            "raw_prices": str(raw_path),
            "raw_prices_sha256": sha256_file(raw_path),
            "corporate_actions": str(actions_path),
            "corporate_actions_sha256": sha256_file(actions_path),
            "availability": str(availability_path),
            "availability_sha256": sha256_file(availability_path),
            "rows": len(adjusted),
            "actual_start": str(adjusted.index.min().date()),
            "actual_end": str(adjusted.index.max().date()),
            "requested_start": args.start,
            "requested_end": args.end,
            "tickers": list(adjusted.columns),
            "source": "tiingo-end-of-day",
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "license": "Internal use only; do not redistribute downloaded Tiingo data.",
            "availability_assumption": "20:00 America/New_York on observation date",
            "limitation": "Current-vintage adjusted history is not a historical revision archive.",
        }
        manifest_path = output.with_name(f"{output.stem}.manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return
    if args.command == "download-massive":
        api_key = os.environ.get("MASSIVE_API_KEY")
        if not api_key:
            raise SystemExit(
                "MASSIVE_API_KEY is not set. Set it locally; never commit or paste the key into a report."
            )
        prices, availability = download_massive_prices(args.start, args.end, api_key)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        prices.rename_axis("date").to_csv(output)
        availability.rename_axis("observation_at").to_csv(
            output.with_name(f"{output.stem}.available-at.csv")
        )
        availability_path = output.with_name(f"{output.stem}.available-at.csv")
        manifest = {
            "prices": str(output),
            "prices_sha256": sha256_file(output),
            "availability": str(availability_path),
            "availability_sha256": sha256_file(availability_path),
            "rows": len(prices),
            "start": args.start,
            "end": args.end,
            "tickers": list(prices.columns),
            "adjusted": False,
            "source": "massive-us-stocks-daily-aggregates",
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "vendor_plan": args.vendor_plan,
            "availability_assumption": "16:15 America/New_York on observation date",
        }
        manifest_path = output.with_name(f"{output.stem}.manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return
    if args.command == "benchmark":
        try:
            comparison = run_benchmark(
                args.csv,
                args.spec,
                args.output,
                args.phase,
                args.unlock_holdout,
                args.tmu_platform,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(comparison.to_string(index=False))
        return
    config = ResearchConfig()
    if args.command == "demo":
        prices, kind = synthetic_prices(seed=config.seed), "synthetic validation"
    else:
        prices, kind = load_prices_csv(args.csv), "user CSV"
    print(json.dumps(run_research(prices, args.output, config, args.model, kind), indent=2))


if __name__ == "__main__":
    main()
