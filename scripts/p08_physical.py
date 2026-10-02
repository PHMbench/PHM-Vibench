"""P08 research commands using PHMFactory's one public YAML resolver."""
from __future__ import annotations

import argparse
from pathlib import Path

from phmfactory.config import analyze_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("data-check", "smoke", "overfit", "tune", "compare", "benchmark", "ablate"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-config")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arms", help="Comma-separated B0,B1,F01,P0,TOKEN,LATE")
    parser.add_argument("--selection", type=Path, help="Tuning root; for ablate, comparison/benchmark root")
    args = parser.parse_args(argv)
    analysis = analyze_config(args.config, local_config=args.local_config, override_values=args.override)
    config = analysis.runtime_config()
    from src.task_factory.task.DG.p08_physical import execute
    output = args.output or Path(config["environment"]["output_dir"]) / args.command
    result = execute(config, args.command, output, target=args.target, seed=args.seed,
                     arms=args.arms.split(",") if args.arms else None, selection=args.selection)
    print(f"{result['status']}: {output} ({result['evidence_kind']}; no industrial claim accepted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
