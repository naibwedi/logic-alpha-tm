import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .config import ResearchConfig
from .data import load_prices_csv, synthetic_prices
from .experiments import run_benchmark, sha256_file
from .pipeline import run_research
from .providers import download_massive_prices


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
    download = sub.add_parser("download-massive", help="download licensed unadjusted daily bars")
    download.add_argument("--start", required=True, help="inclusive date (YYYY-MM-DD)")
    download.add_argument("--end", required=True, help="inclusive date (YYYY-MM-DD)")
    download.add_argument("--output", default="data/raw/massive-prices.csv")
    download.add_argument("--vendor-plan", default="not-recorded", help="non-secret licensed plan label for the manifest")
    return root


def main() -> None:
    args = parser().parse_args()
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
                args.csv, args.spec, args.output, args.phase, args.unlock_holdout
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
