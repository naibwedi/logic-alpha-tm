import argparse
import json

from .config import ResearchConfig
from .data import load_prices_csv, synthetic_prices
from .pipeline import run_research


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
    return root


def main() -> None:
    args = parser().parse_args()
    config = ResearchConfig()
    if args.command == "demo":
        prices, kind = synthetic_prices(seed=config.seed), "synthetic validation"
    else:
        prices, kind = load_prices_csv(args.csv), "user CSV"
    print(json.dumps(run_research(prices, args.output, config, args.model, kind), indent=2))


if __name__ == "__main__":
    main()

