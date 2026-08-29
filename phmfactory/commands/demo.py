"""One-command wrapper around the maintained offline Dummy smoke."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from importlib import resources
from pathlib import Path
from typing import Any


DEFAULT_OVERRIDES = (
    "trainer.num_epochs=1",
    "trainer.device=cpu",
    "data.num_workers=0",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phmfactory demo",
        description="Run the repository-shipped offline Dummy smoke.",
    )
    parser.add_argument("--notes", default="", help="Experiment notes.")
    parser.add_argument(
        "--override",
        action="append",
        help="Additional key=value override; later values replace demo defaults.",
    )
    return parser


def run(
    argv: Sequence[str],
    *,
    experiment_runner: Callable[[argparse.Namespace], Any],
) -> Any:
    args = build_parser().parse_args(list(argv))
    packaged_data_dir = Path(str(resources.files("data"))).resolve()
    experiment_args = argparse.Namespace(
        config="smoke",
        config_path=None,
        notes=args.notes,
        override=[
            f"data.data_dir={packaged_data_dir}",
            *DEFAULT_OVERRIDES,
            *(args.override or ()),
        ],
        allow_experimental=False,
    )
    return experiment_runner(experiment_args)
