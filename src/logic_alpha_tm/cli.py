import argparse
import json
import os
from pathlib import Path

from .config import ResearchConfig
from .data import load_prices_csv, synthetic_prices
from .pipeline import run_research
from .providers import download_massive_prices


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="LogicAlpha-TM research runner")
    sub = root.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run deterministic synthetic validation")
    demo.add_argument("--output", default="results")
    demo.add_argument("--model", choices=("bernoulli", "tmu"), default="bernoulli")
    run = sub.add_parser("run", help="run on a wide adjusted-close CSV")
    run.add_argument("--csv", required=True)
    run.add_argument("--output", default="results")
    run.add_argument("--model", choices=("bernoulli", "tmu"), default="bernoulli")
    download = sub.add_parser("download-massive", help="download licensed unadjusted daily bars")
    download.add_argument("--start", required=True, help="inclusive date (YYYY-MM-DD)")
    download.add_argument("--end", required=True, help="inclusive date (YYYY-MM-DD)")
    download.add_argument("--output", default="data/raw/massive-prices.csv")
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
        print(json.dumps({"prices": str(output), "rows": len(prices), "adjusted": False}, indent=2))
        return
    config = ResearchConfig()
    if args.command == "demo":
        prices, kind = synthetic_prices(seed=config.seed), "synthetic validation"
    else:
        prices, kind = load_prices_csv(args.csv), "user CSV"
    print(json.dumps(run_research(prices, args.output, config, args.model, kind), indent=2))


if __name__ == "__main__":
    main()
