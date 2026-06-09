"""Run a small generative robustness sweep through the public main.py entrypoint."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _method_id(config: str) -> str:
    return Path(config).stem.replace("dummy_generative_", "")


def run_sweep(configs: list[str] | str, seeds: list[int], steps: list[int], out_csv: Path) -> int:
    if isinstance(configs, str):
        config_list = [configs]
    else:
        config_list = list(configs)
    if not config_list:
        raise ValueError("run_sweep requires at least one config")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for config in config_list:
        method = _method_id(config)
        for seed in seeds:
            for num_steps in steps:
                cmd = [
                    sys.executable,
                    "main.py",
                    "--config",
                    config,
                    "--override",
                    f"environment.seed={seed}",
                    "--override",
                    f"task.generative.num_steps={num_steps}",
                ]
                start = time.perf_counter()
                result = subprocess.run(cmd, text=True, capture_output=True)
                rows.append(
                    {
                        "config": config,
                        "method": method,
                        "seed": seed,
                        "num_steps": num_steps,
                        "returncode": result.returncode,
                        "wall_clock_sec": f"{time.perf_counter() - start:.6f}",
                        "stdout_tail": result.stdout[-500:],
                        "stderr_tail": result.stderr[-500:],
                    }
                )
                if result.returncode != 0:
                    break
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "config",
                "method",
                "seed",
                "num_steps",
                "returncode",
                "wall_clock_sec",
                "stdout_tail",
                "stderr_tail",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    failed = [row for row in rows if row["returncode"] != 0]
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a generative config/seed/step sweep.")
    parser.add_argument("--config", default="configs/demo/10_generative/dummy_generative_cfm.yaml")
    parser.add_argument(
        "--configs",
        default=None,
        help="Comma-separated configs for a multi-family sweep. Overrides --config when set.",
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--num_steps", default="4,8,16,32")
    parser.add_argument("--out_csv", default="results/generative_sweep/sweep_summary.csv")
    args = parser.parse_args()
    code = run_sweep(
        configs=_parse_str_list(args.configs) if args.configs else args.config,
        seeds=_parse_int_list(args.seeds),
        steps=_parse_int_list(args.num_steps),
        out_csv=Path(args.out_csv),
    )
    print(f"[sweep] summary -> {args.out_csv}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
