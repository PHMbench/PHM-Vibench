"""Existing H5 selection and deterministic acquisition window access for P01."""
from __future__ import annotations
import importlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch


def _vibench_id(value):
    """Vibench H5 keys are the string form of integer acquisition IDs."""
    if pd.isna(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError("Vibench Id must be a non-null integer")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid Vibench Id: {value!r}") from error
    if isinstance(value, str):
        if value != str(number):
            raise ValueError(f"noncanonical Vibench Id: {value!r}")
    elif value != number:
        raise ValueError(f"fractional Vibench Id: {value!r}")
    return str(number)


def _h5_metadata_frame(dataset, frame, mapping):
    """Select existing rows and optionally join only missing grouping columns."""
    id_column = mapping.get("id")
    if id_column is None or id_column not in frame:
        raise ValueError("vibench_h5 requires columns.id mapped to the existing Id column")
    frame = frame.copy()
    frame[id_column] = frame[id_column].map(_vibench_id)
    if frame[id_column].duplicated().any():
        raise ValueError("duplicate Vibench Id in metadata")
    for column, values in dataset.get("select", {}).items():
        if column not in frame or not isinstance(values, list) or not values:
            raise ValueError("select requires an existing column and a nonempty value list")
        if column == id_column:
            values = [_vibench_id(value) for value in values]
        frame = frame[frame[column].isin(values)]
    if frame.empty:
        raise ValueError("declared Vibench selection contains no records")
    if dataset.get("protocol_file"):
        path = Path(dataset["protocol_file"]).expanduser().resolve()
        protocol = pd.read_csv(path, dtype={id_column: str})
        grouping = [mapping.get("unit_id"), mapping.get("split")]
        if None in grouping or set(protocol.columns) != {id_column, *grouping}:
            raise ValueError("protocol_file may contain only Id, the unit column and the split column")
        if any(column in frame for column in grouping):
            raise ValueError("protocol_file cannot overwrite existing metadata grouping columns")
        protocol[id_column] = protocol[id_column].map(_vibench_id)
        if protocol[id_column].duplicated().any():
            raise ValueError("duplicate Id in protocol_file")
        frame = frame.merge(protocol, on=id_column, how="left", validate="one_to_one")
        if frame[grouping].isna().any().any():
            raise ValueError("protocol_file is missing selected Id assignments")
    return frame


def window_record(record, dataset, data_cfg):
    if dataset["format"] == "vibench_h5":
        from src.data_factory.H5DataDict import H5DataDict
        with H5DataDict(record["path"]) as data:
            signal = data[record["h5_key"]]
    elif dataset["format"] == "npz":
        with np.load(record["path"], allow_pickle=False) as archive:
            signal = archive[data_cfg.get("array_key", "x")]
    else:
        reader = importlib.import_module("src.data_factory.reader." + str(record["reader"]))
        signal = reader.read(
            record["path"], SimpleNamespace(**dataset.get("reader_args", {}))
        )

    for axis in sorted(data_cfg.get("squeeze_axes", []), reverse=True):
        signal = np.squeeze(signal, axis=int(axis))
    layout = data_cfg["layout"]
    if layout == "L":
        if signal.ndim != 1:
            raise ValueError(f"expected L, got {signal.shape}")
        signal = signal[:, None]
    elif layout == "CL":
        if signal.ndim != 2:
            raise ValueError(f"expected CL, got {signal.shape}")
        signal = signal.T
    elif layout != "LC" or signal.ndim != 2:
        raise ValueError(f"expected declared layout {layout}, got {signal.shape}")

    if "channel_indices" in data_cfg:
        channels = data_cfg["channel_indices"]
        if (not isinstance(channels, list) or not channels or
                any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < signal.shape[1]
                    for i in channels) or len(set(channels)) != len(channels)):
            raise ValueError("channel_indices must name unique, observed zero-based LC channels")
        signal = signal[:, channels]

    signal = np.asarray(signal, dtype=np.float32)
    if not np.isfinite(signal).all():
        raise ValueError("nonfinite raw signal")
    length = int(data_cfg["window_size"])
    count = int(data_cfg["windows_per_unit"])
    if len(signal) < length or count < 1:
        raise ValueError("record too short or invalid window count")
    starts = np.linspace(0, len(signal) - length, count, dtype=np.int64)
    if len(np.unique(starts)) != count:
        raise ValueError("requested duplicate windows; reduce windows_per_unit")
    return torch.from_numpy(np.stack([signal[start : start + length] for start in starts]))


def materialize(records, dataset, cfg):
    return [dict(record, x=window_record(record, dataset, cfg["data"])) for record in records]


def pack_batch(units, device):
    x = torch.cat([unit["x"] for unit in units]).to(device)
    counts = [len(unit["x"]) for unit in units]
    y = torch.cat(
        [
            torch.full((count,), unit["label"], dtype=torch.long)
            for unit, count in zip(units, counts)
        ]
    ).to(device)
    unit_ids = torch.cat(
        [
            torch.full((count,), index, dtype=torch.long)
            for index, count in enumerate(counts)
        ]
    ).to(device)
    metadata = {
        key: torch.cat(
            [torch.full((count,), unit[key]) for unit, count in zip(units, counts)]
        ).to(device)
        for key in ("sample_rate_hz", "rotation_speed_rpm")
    }
    return x, y, unit_ids, metadata
