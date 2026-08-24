"""Download the maintained CWRU bundle and run a one-epoch CPU demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phmfactory import cli
from phmfactory.data_sources import download_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("huggingface", "modelscope"),
        default="huggingface",
    )
    parser.add_argument("--destination", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    return parser


def main() -> object:
    args = build_parser().parse_args()
    bundle = download_bundle(
        "cwru-demo-v1",
        source=args.source,
        destination=args.destination,
        revision=args.revision,
        force=args.force,
    )
    if args.download_only:
        return bundle

    data_dir = json.dumps(str(bundle.directory))
    output_dir = json.dumps(str(Path("outputs") / "demo" / "cwru_fault_diagnosis"))
    return cli.main(
        [
            "--config",
            "configs/demo/01_cross_domain/cwru_dg.yaml",
            "--override",
            f"data.data_dir={data_dir}",
            "--override",
            "data.num_workers=0",
            "--override",
            "data.batch_size=32",
            "--override",
            "data.window_size=1024",
            "--override",
            "data.stride=1024",
            "--override",
            "data.num_window=8",
            "--override",
            "trainer.device=cpu",
            "--override",
            "trainer.devices=1",
            "--override",
            "trainer.num_epochs=1",
            "--override",
            f"environment.output_dir={output_dir}",
        ]
    )


if __name__ == "__main__":
    main()
