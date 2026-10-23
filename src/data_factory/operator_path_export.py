"""Export explicitly labelled records for bounded operator-path experiments.

Readers, channel selection and chronological complete windows retain the existing
P07 data contract. Source-only exports require train/val records and reject target
metadata before any waveform is read. No normalization or resampling is applied.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
import re
import traceback

import numpy as np

READERS = Path(__file__).resolve().parent / "reader"


def safe_identifier(value: str, key: str) -> None:
    if not isinstance(value, str) or not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-." for c in value) or value in {".", ".."}:
        raise ValueError(f"Use a nonempty safe identifier for {key}")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def validate_data(data: dict, domain_holdout: bool, *,
                  required_splits: tuple[str, ...] = ("train", "val", "test")) -> None:
    required = {"x", "y", "unit_id", "split", "domain", "sampling_rate"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Dataset lacks explicit fields: {sorted(missing)}")
    x, y = data["x"], data["y"]
    if x.ndim != 3 or x.shape[1] < 32 or x.shape[2] < 1 or not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all():
        raise ValueError("x must be finite (N,L>=32,C>=1); no layout or signal repair is attempted")
    n = len(x)
    for key in ("y", "unit_id", "split", "domain"):
        if data[key].shape != (n,):
            raise ValueError(f"{key} must have shape ({n},)")
    if not np.issubdtype(y.dtype, np.integer) or set(y) != set(range(len(np.unique(y)))):
        raise ValueError("Labels must be explicit contiguous integers 0..C-1")
    if len(np.unique(y)) < 2:
        raise ValueError("Classification and winner-rival comparisons need at least two classes")
    if set(data["split"]) != set(required_splits):
        raise ValueError(f"Explicit partitions {required_splits} are required")
    fs = np.asarray(data["sampling_rate"])
    if fs.size != 1 or not np.isfinite(fs).all() or float(fs.reshape(-1)[0]) <= 0:
        raise ValueError("One positive sampling rate is required; resampling is not automatic")
    train_labels = set(y[data["split"] == "train"])
    for split in required_splits[1:]:
        if set(y[data["split"] == split]) != train_labels:
            raise ValueError(f"{split} class support differs from train; declare a different task explicitly")
    for unit in np.unique(data["unit_id"]):
        indices = data["unit_id"] == unit
        if len(set(data["split"][indices])) != 1:
            raise ValueError(f"Physical/record unit {unit!r} crosses partitions")
        if len(set(y[indices])) != 1:
            raise ValueError(f"Unit {unit!r} must have one label; use an explicitly defined longitudinal task otherwise")
    if domain_holdout:
        domain_sets = [set(data["domain"][data["split"] == split]) for split in required_splits]
        if any(domain_sets[i] & domain_sets[j] for i in range(len(domain_sets)) for j in range(i)):
            raise ValueError("Domain-held-out protocol has a domain crossing partitions")
        if len(domain_sets[0]) < 2:
            raise ValueError("Multi-source DG needs at least two training domains")
    for key in ("domain", "unit_id"):
        if any(not str(value).strip() for value in data[key]):
            raise ValueError(f"{key} contains an empty identifier")


def read_signal(reader: str, path: Path) -> np.ndarray:
    """No layout guessing: a reader or explicit NPY must return time × channels."""
    if reader == "npy":
        if path.suffix.lower() != ".npy":
            raise ValueError("The npy reader accepts an explicit .npy array only")
        signal = np.load(path, allow_pickle=False)
    else:
        if not re.fullmatch(r"RM_[0-9]{3}_[A-Za-z0-9_]+", reader) or not (READERS / f"{reader}.py").is_file():
            raise ValueError(f"Reader must name an existing PHMFactory RM_NNN module: {reader!r}")
        module = importlib.import_module(f"src.data_factory.reader.{reader}")
        if not callable(getattr(module, "read", None)):
            raise ValueError(f"Existing reader {reader} has no read() entry")
        signal = np.asarray(module.read(str(path)))
    if signal.ndim != 2 or not np.issubdtype(signal.dtype, np.number) or np.iscomplexobj(signal) or not np.isfinite(signal).all():
        raise ValueError(f"Reader must return finite real numeric (time, channels): {path}")
    return signal


def resolve_fold(fold: dict, base: Path, *, source_only: bool = False) -> dict:
    required = {"dataset", "fold", "reader", "window_size", "stride", "channels", "sampling_rate",
                "source_version", "provenance_notes", "records"}
    if not isinstance(fold, dict) or not required <= fold.keys():
        raise ValueError(f"Each fold requires explicit fields: {sorted(required)}")
    for key in ("dataset", "fold"):
        safe_identifier(fold[key], key)
    for key in ("source_version", "provenance_notes"):
        if not isinstance(fold[key], str) or not fold[key].strip():
            raise ValueError(f"{key} must document the raw source and export decisions")
    reader = fold["reader"]
    if not isinstance(reader, str) or (reader != "npy" and (
            not re.fullmatch(r"RM_[0-9]{3}_[A-Za-z0-9_]+", reader) or not (READERS / f"{reader}.py").is_file())):
        raise ValueError("Choose npy or an existing PHMFactory RM_NNN reader; arbitrary import paths are forbidden")
    if type(fold["window_size"]) is not int or fold["window_size"] < 32:
        raise ValueError("window_size must be an integer >=32")
    if type(fold["stride"]) is not int or fold["stride"] < 1:
        raise ValueError("stride must be a positive integer")
    channels = fold["channels"]
    if not isinstance(channels, list) or not channels or any(type(c) is not int or c < 0 for c in channels) or len(channels) != len(set(channels)):
        raise ValueError("channels must explicitly list distinct nonnegative indices in output order")
    rate = fold["sampling_rate"]
    if type(rate) not in (int, float) or not np.isfinite(rate) or rate <= 0:
        raise ValueError("sampling_rate must be positive and finite; resampling is not performed")
    records = fold["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a nonempty explicit metadata list")
    required_splits = ("train", "val") if source_only else ("train", "val", "test")
    paths, units, domain_partition, labels = set(), {}, {}, {s: set() for s in required_splits}
    resolved = []
    for record in records:
        if not isinstance(record, dict) or not {"path", "label", "unit_id", "domain", "split"} <= record.keys():
            raise ValueError("Each record needs path, label, unit_id, domain and split before windowing")
        if type(record["label"]) is not int or record["label"] < 0:
            raise ValueError("Record labels must be explicit nonnegative integers")
        if record["split"] not in labels:
            raise ValueError("Source-only export permits train/val records only; target records are forbidden"
                             if source_only else "Record split must be train, val or test")
        for key in ("path", "unit_id", "domain"):
            if not isinstance(record[key], str) or not record[key].strip():
                raise ValueError(f"Record {key} must be a nonempty string")
        path = Path(record["path"]).expanduser()
        path = (base / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path in paths:
            raise ValueError(f"One raw record cannot be duplicated or relabelled within a fold: {path}")
        paths.add(path)
        unit_state = (record["label"], record["split"])
        if record["unit_id"] in units and units[record["unit_id"]] != unit_state:
            raise ValueError("A physical unit crosses labels or partitions before windowing")
        units[record["unit_id"]] = unit_state
        if record["domain"] in domain_partition and domain_partition[record["domain"]] != record["split"]:
            raise ValueError("A domain crosses partitions before windowing")
        domain_partition[record["domain"]] = record["split"]
        labels[record["split"]].add(record["label"])
        resolved.append(record | {"path": str(path)})
    if len(labels["train"]) < 2 or labels["train"] != set(range(len(labels["train"]))):
        raise ValueError("Training labels must cover at least two contiguous integer classes")
    if any(labels[split] != labels["train"] for split in required_splits[1:]):
        raise ValueError("Every partition must retain the same class support")
    if sum(split == "train" for split in domain_partition.values()) < 2:
        raise ValueError("Multi-source DG requires at least two training domains")
    return fold | {"records": resolved}


def export_spec(spec_path: Path, output: Path, *, source_only: bool = False) -> Path:
    """Write a new derived directory; leave raw inputs and existing outputs untouched."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an export: {output}")
    output.mkdir(parents=True)
    try:
        original = spec_path.read_text(encoding="utf-8")
        (output / "spec.json").write_text(original, encoding="utf-8")
        spec = json.loads(original)
        if not isinstance(spec, dict) or not isinstance(spec.get("folds"), list) or not spec["folds"]:
            raise ValueError("Export spec requires a nonempty folds list")
        folds = [resolve_fold(fold, spec_path.resolve().parent, source_only=source_only) for fold in spec["folds"]]
        identities = [(fold["dataset"], fold["fold"]) for fold in folds]
        if len(identities) != len(set(identities)):
            raise ValueError("Duplicate dataset/fold export identity")
        write_json(output / "resolved_spec.json", {"folds": folds, "source_spec": str(spec_path.resolve()),
            "transform": "explicit channel selection; chronological complete windows; float32 conversion; no normalization or resampling",
            "source_only": source_only})
        manifest = []
        with (output / "window_records.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["dataset", "fold", "window", "path", "start", "stop", "unit_id", "domain", "split", "label"])
            writer.writeheader()
            for fold in folds:
                windows, labels, units, domains, splits = [], [], [], [], []
                for record in fold["records"]:
                    signal = read_signal(fold["reader"], Path(record["path"]))
                    if not isinstance(signal, np.ndarray) or signal.ndim != 2 or not np.issubdtype(signal.dtype, np.number) or np.iscomplexobj(signal) or not np.isfinite(signal).all():
                        raise ValueError(f"Reader must return finite real numeric (time, channels): {record['path']}")
                    if max(fold["channels"]) >= signal.shape[1]:
                        raise ValueError(f"Requested channel is absent: {record['path']}")
                    if signal.shape[0] < fold["window_size"]:
                        raise ValueError(f"Record is shorter than one declared window: {record['path']}")
                    for start in range(0, signal.shape[0] - fold["window_size"] + 1, fold["stride"]):
                        stop = start + fold["window_size"]
                        windows.append(signal[start:stop, fold["channels"]].astype(np.float32))
                        labels.append(record["label"])
                        units.append(record["unit_id"])
                        domains.append(record["domain"])
                        splits.append(record["split"])
                        writer.writerow(dict(dataset=fold["dataset"], fold=fold["fold"], window=len(windows)-1,
                            path=record["path"], start=start, stop=stop, unit_id=record["unit_id"],
                            domain=record["domain"], split=record["split"], label=record["label"]))
                data = dict(x=np.stack(windows), y=np.asarray(labels, dtype=np.int64), unit_id=np.asarray(units),
                    domain=np.asarray(domains), split=np.asarray(splits), sampling_rate=np.asarray(fold["sampling_rate"]))
                validate_data(data, domain_holdout=True,
                              required_splits=("train", "val") if source_only else ("train", "val", "test"))
                target = output / fold["dataset"] / f"{fold['fold']}.npz"
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target, **data)
                manifest.append(dict(dataset=fold["dataset"], fold=fold["fold"], path=str(target.relative_to(output))))
                stream.flush()
        result = output / "manifest.json"
        write_json(result, {"folds": manifest})
        write_json(output / "status.json", {"status": "completed", "folds": len(folds), "raw_inputs_modified": False})
        return result
    except Exception as error:
        write_json(output / "failure.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def export_source(manifest: Path, output: Path) -> Path:
    """Export an explicit train/val raw-record spec without reading target signals."""
    return export_spec(manifest, output, source_only=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-only", action="store_true", help="Require train/val records only")
    args = parser.parse_args()
    print(export_spec(args.spec, args.output, source_only=args.source_only))


if __name__ == "__main__":
    main()
