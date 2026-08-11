"""Validate the C01 grouped real-data protocol without training a model."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from src.Pipeline_01_Fault_Diagnosis import write_p01_data_protocol_summary
from src.configs.config_utils import load_config
from src.data_factory import build_data
from src.model_factory.X_model.P01Alignment import Model


def _plain_list(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _probe_loaders(data_factory: Any, model: Model) -> dict[str, Any]:
    result: dict[str, Any] = {}
    metadata = data_factory.get_metadata()
    protocol = data_factory.grouped_protocol
    expected_domains = {
        "train": protocol["source_domains"],
        "val": protocol["source_domains"],
        "test": protocol["target_domains"],
    }
    for split in ("train", "val", "test"):
        pending = set(expected_domains[split])
        environment_batches: dict[str, Any] = {}
        for batch in data_factory.get_dataloader(split):
            waveform = batch["x"]
            if not torch.isfinite(waveform).all():
                raise FloatingPointError(f"{split} loader produced NaN or Inf.")
            if "physical_group_id" not in batch:
                raise ValueError(f"{split} loader omitted physical_group_id.")
            file_ids = _plain_list(batch["file_id"])
            domains = [metadata[file_id]["Domain_id"] for file_id in file_ids]
            for domain in list(pending):
                matching = [index for index, value in enumerate(domains) if value == domain]
                if not matching:
                    continue
                rendered = model.render_2d_view(waveform[matching[:1]])
                environment_batches[str(domain)] = {
                    "waveform_batch_shape": list(waveform.shape),
                    "rendered_shape_one_sample": list(rendered.shape),
                    "matching_file_ids": [file_ids[index] for index in matching],
                    "matching_physical_group_ids": [
                        _plain_list(batch["physical_group_id"])[index]
                        for index in matching
                    ],
                    "matching_labels": [
                        _plain_list(batch["y"])[index] for index in matching
                    ],
                }
                pending.remove(domain)
            if not pending:
                break
        if pending:
            raise ValueError(
                f"{split} loader did not produce configured domain(s) {sorted(pending)}."
            )
        result[split] = {"environment_batches": environment_batches}
    return result


def _require_local_cwru_assets(config: Any) -> dict[str, str]:
    data_root = Path(config.data.data_dir).expanduser().resolve()
    metadata_path = data_root / str(config.data.metadata_file)
    signal_path = data_root / "RM_001_CWRU.h5"
    missing = [str(path) for path in (metadata_path, signal_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "C01 requires pre-existing local CWRU assets and will not download a "
            f"replacement; missing {missing}."
        )
    return {
        "data_root": str(data_root),
        "metadata_file": str(metadata_path),
        "signal_store": str(signal_path),
        "data_release_boundary": (
            "Official CWRU 12 kHz drive-end/fan-end condition tables; the source "
            "site publishes no numbered release or specimen serial identifiers."
        ),
    }


def validate(
    config_path: str | Path,
    output_path: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> Path:
    config = load_config(config_path)
    if data_dir is not None:
        config.data.data_dir = str(Path(data_dir).expanduser().resolve())
    data_provenance = _require_local_cwru_assets(config)
    seed = int(config.environment.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data_factory = build_data(config.data, config.task)
    try:
        renderer_identities = {}
        models = {}
        for condition in ("M2", "M3", "M4", "M5"):
            model_args = copy.deepcopy(config.model)
            model_args.condition = condition
            model = Model(model_args)
            renderer_identities[condition] = model.renderer_identity()
            models[condition] = model
        if len({repr(value) for value in renderer_identities.values()}) != 1:
            raise ValueError("M2--M5 do not consume one identical renderer identity.")

        probe = _probe_loaders(data_factory, models["M5"])
        probe["renderer_equal_across_M2_M3_M4_M5"] = True
        target = Path(output_path) if output_path is not None else (
            Path(config.environment.output_dir) / "data_protocol_summary.json"
        )
        return write_p01_data_protocol_summary(
            target,
            data_factory,
            config.model,
            model=models["M5"],
            loader_probe=probe,
            provenance={
                "config_path": str(config_path),
                **data_provenance,
            },
        )
    finally:
        close = getattr(getattr(data_factory, "data", None), "close", None)
        if callable(close):
            close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/p01/p01_c01_grouped_protocol.yaml",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override data.data_dir; assets must already exist locally.",
    )
    args = parser.parse_args()
    path = validate(args.config, args.output, data_dir=args.data_dir)
    print(f"[C01 PASS] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
