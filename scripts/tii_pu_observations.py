"""Read-only Goal07 observations of the fixed PU original MAT population.

This diagnostic does not call a training reader, touch H5, infer physical units,
or qualify the sampling clock. Run with ``python -m scripts.tii_pu_observations``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat
from scipy.io.matlab import MatReadError

from src.data_factory.data_utils import read_metadata_table
from src.utils.identifiers import validate_identifiers


DATASET = "RM_027_PU"
DATASET_ID = 20
EXPECTED_RECORDS = 2559
SENTINEL_ID = 47567
OUTPUTS = ("record_observations.csv", "group_summary.csv", "observation_summary.json")


def matched_records(metadata: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    """Use the existing inventory as the identity/group map, before any MAT read."""
    selected = metadata.loc[(metadata.Name == DATASET) | (metadata.Dataset_id == DATASET_ID)].copy()
    known = inventory.loc[(inventory.dataset == DATASET) | (inventory.dataset_id == DATASET_ID)
                          | (inventory.file == f"{DATASET}.h5")].copy()
    for frame, fields in ((selected, ("Id", "Name", "File", "Dataset_id")),
                          (known, ("metadata_id", "recording_id", "original_recording", "group", "dataset_id"))):
        for field in fields:
            validate_identifiers(frame[field], field)
    for values in (selected.Id, known.metadata_id, known.recording_id):
        if any(not isinstance(x, (int, float, np.integer, np.floating)) or x != int(x) for x in values):
            raise ValueError("PU metadata identities must be finite integral numeric values")
        if values.duplicated().any():
            raise ValueError("PU metadata/inventory identities must be unique")
    if not (selected.Name.eq(DATASET).all() and selected.Dataset_id.eq(DATASET_ID).all()
            and known.dataset.eq(DATASET).all() and known.dataset_id.eq(DATASET_ID).all()
            and known.file.eq(f"{DATASET}.h5").all()):
        raise ValueError("PU Name, dataset_id and inventory container disagree")
    if set(selected.Id) != set(known.metadata_id):
        raise ValueError("PU metadata/inventory ID coverage differs")
    if not known.metadata_id.eq(known.recording_id).all():
        raise ValueError("PU inventory recording_id differs from metadata_id")
    joined = selected.merge(known, left_on="Id", right_on="metadata_id", validate="one_to_one")
    if not joined.File.eq(joined.original_recording).all():
        raise ValueError("PU metadata File differs from existing inventory original_recording")
    if joined.File.duplicated().any():
        raise ValueError("PU original recording paths are not unique")
    for row in joined.itertuples(index=False):
        path = Path(row.File)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"PU original recording path is not relative: {row.File!r}")
        if not (row.Sample_rate == row.metadata_rate and row.Sample_lenth == row.metadata_length):
            raise ValueError(f"PU metadata rate/length changed since inventory for Id {row.Id}")
        if row.Label != row.label:
            raise ValueError(f"PU metadata label changed since inventory for Id {row.Id}")
    return joined.sort_values("Id", kind="stable")


def _field(struct: object, name: str) -> object:
    if not hasattr(struct, name):
        raise ValueError(f"missing MAT field {name}")
    return getattr(struct, name)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty scalar string")
    return value


def _unit(struct: object) -> tuple[str | None, str]:
    if not hasattr(struct, "Unit"):
        return None, "absent"
    value = np.asarray(struct.Unit)
    original = json.dumps(value.tolist(), ensure_ascii=False, allow_nan=False)
    blank = value.size == 0 or (value.ndim == 0 and isinstance(value.item(), str) and not value.item().strip())
    return original, "empty" if blank else "present"


def _vector(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.size < 2 or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric 1D vector with >=2 entries; shape={array.shape}, dtype={array.dtype}")
    return array


def observe_record(path: Path, identity: dict[str, object], source_error: str | None = None) -> dict[str, object]:
    """Retain one row even when original parsing or channel association fails."""
    row = dict(identity, observation_status="failed", error="", time_association_status="unresolved",
               channel_name=None, y_index_1based=None, x_index_1based=None, time_field=None,
               signal_raster=None, time_raster=None, time_name_original=None, down_sampling_original=None,
               n_signal=None, n_time=None, signal_dtype=None, time_dtype=None,
               signal_unit_original=None, signal_unit_status="not_inspected", time_unit_original=None,
               time_unit_status="not_inspected", signal_finite=None, time_finite=None,
               first_time=None, last_time=None, duplicate_time_steps=None, reversed_time_steps=None,
               dt_min=None, dt_median=None, dt_max=None, dt_std=None, stored_dt_exactly_nonuniform=None,
               span_average_rate_per_stored_unit=None, span_average_rate_hz=None,
               rate_difference_hz=None, rate_hz_status="unresolved", length_difference=None,
               unit_conversion_status="UNRESOLVED; no conversion applied", inspection="new_raw_mat_check")
    try:
        if source_error:
            raise ValueError(source_error)
        original = loadmat(path, squeeze_me=True, struct_as_record=False)[path.stem]
        channels = np.atleast_1d(_field(original, "Y"))
        matches = [(i, y) for i, y in enumerate(channels, 1)
                   if isinstance(getattr(y, "Name", None), str) and y.Name == "vibration_1"]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one Y.Name='vibration_1'; observed {len(matches)}")
        index, signal = matches[0]
        row.update(channel_name=signal.Name, y_index_1based=index)
        row["signal_unit_original"], row["signal_unit_status"] = _unit(signal)
        row["down_sampling_original"] = json.dumps(np.asarray(_field(signal, "DownSampling")).tolist())
        values = _vector(_field(signal, "Data"), "vibration_1.Data")
        row.update(n_signal=values.size, signal_dtype=str(values.dtype), signal_finite=bool(np.isfinite(values).all()),
                   length_difference=int(values.size) - row["metadata_length"])
        axes = np.atleast_1d(_field(original, "X"))
        xindex = np.asarray(_field(signal, "XIndex"))
        if xindex.ndim != 0 or xindex.dtype.kind not in "fiu" or not np.isfinite(xindex):
            raise ValueError("vibration_1.XIndex must be a finite scalar integer")
        number = xindex.item()
        if number != int(number) or not 1 <= number <= len(axes):
            raise ValueError(f"vibration_1.XIndex={number!r} outside integer 1..{len(axes)}")
        row.update(x_index_1based=int(number), time_field=f"X[{int(number)}].Data (MATLAB 1-based)")
        axis = axes[int(number) - 1]
        row["time_name_original"] = json.dumps(np.asarray(_field(axis, "Name")).tolist())
        row["time_unit_original"], row["time_unit_status"] = _unit(axis)
        row["signal_raster"] = _text(_field(signal, "Raster"), "Y.Raster")
        row["time_raster"] = _text(_field(axis, "Raster"), "X.Raster")
        if row["signal_raster"] != row["time_raster"]:
            raise ValueError(f"Y.Raster={row['signal_raster']!r} differs from X.Raster={row['time_raster']!r}")
        timestamps = _vector(_field(axis, "Data"), "linked X.Data")
        finite = bool(np.isfinite(timestamps).all())
        row.update(n_time=timestamps.size, time_dtype=str(timestamps.dtype), time_finite=finite)
        if len(values) != len(timestamps):
            raise ValueError(f"linked signal/time lengths differ: {len(values)} versus {len(timestamps)}")
        row["time_association_status"] = "mapped_by_unique_name_xindex_raster_length"
        row.update(first_time=float(timestamps[0]) if np.isfinite(timestamps[0]) else None,
                   last_time=float(timestamps[-1]) if np.isfinite(timestamps[-1]) else None)
        if not finite or not row["signal_finite"]:
            raise ValueError("linked signal/time contains nonfinite values")
        # Promote integers before differencing to avoid unsigned or narrow-integer wrap.
        times = timestamps.astype(np.float64, copy=False)
        dt = np.diff(times)
        row.update(first_time=float(times[0]), last_time=float(times[-1]),
                   duplicate_time_steps=int(np.count_nonzero(dt == 0)), reversed_time_steps=int(np.count_nonzero(dt < 0)),
                   dt_min=float(dt.min()), dt_median=float(np.median(dt)), dt_max=float(dt.max()),
                   dt_std=float(dt.std()), stored_dt_exactly_nonuniform=bool(np.any(dt != dt[0])))
        if np.any(dt <= 0):
            raise ValueError("linked stored timestamps are not strictly increasing")
        rate = float((len(times) - 1) / (times[-1] - times[0]))
        row.update(span_average_rate_per_stored_unit=rate, span_average_rate_hz=rate,
                   rate_difference_hz=rate - row["metadata_fs"],
                   rate_hz_status="conditional_on_X.Data_being_seconds; physical_clock_UNRESOLVED",
                   observation_status="observed")
    except (OSError, MatReadError, ValueError, TypeError, KeyError, IndexError, NotImplementedError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:1000]
    return row


def group_summary(observations: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for (bearing, condition), part in observations.groupby(["bearing_id", "operating_condition"], sort=True, dropna=False):
        rates = part.span_average_rate_per_stored_unit.dropna()
        summaries.append(dict(bearing_id=bearing, operating_condition=condition, records=len(part),
            observed=int(part.observation_status.eq("observed").sum()), failed=int(part.observation_status.eq("failed").sum()),
            mapping_failures=int(part.time_association_status.eq("unresolved").sum()),
            signal_unit_empty_or_absent=int(part.signal_unit_status.isin(["empty", "absent"]).sum()),
            time_unit_empty_or_absent=int(part.time_unit_status.isin(["empty", "absent"]).sum()),
            stored_axis_nonuniform_records=int(part.stored_dt_exactly_nonuniform.eq(True).sum()),
            duplicate_time_steps=int(part.duplicate_time_steps.sum()), reversed_time_steps=int(part.reversed_time_steps.sum()),
            rate_per_stored_unit_min=None if rates.empty else float(rates.min()),
            rate_per_stored_unit_max=None if rates.empty else float(rates.max())))
    return pd.DataFrame(summaries)


def run(args: argparse.Namespace, argv: list[str]) -> int:
    started, usage_before = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
    root, output = args.data_root.resolve(), args.output_dir.resolve()
    if output.is_relative_to(root):
        raise ValueError("observation outputs must be outside the immutable data root")
    if any((output / name).exists() or (output / name).is_symlink() for name in OUTPUTS):
        raise FileExistsError(f"refusing to overwrite observation output in {output}")
    selected = matched_records(read_metadata_table(root / args.metadata_file), pd.read_csv(args.inventory))
    if len(selected) != EXPECTED_RECORDS or SENTINEL_ID not in set(selected.Id):
        raise ValueError(f"fixed PU population requires {EXPECTED_RECORDS} records including sentinel Id {SENTINEL_ID}; observed {len(selected)}")
    selected = pd.concat([selected.loc[selected.Id == SENTINEL_ID], selected.loc[selected.Id != SENTINEL_ID]])
    records = []
    for count, native in enumerate(selected.itertuples(index=False), 1):
        stem = Path(native.File).stem
        suffix = f"_{native.group}_"
        source_error = None
        condition = stem.split(suffix)[0] if stem.count(suffix) == 1 else None
        if condition is None or str(Path(native.File).parts[0]) != str(native.group):
            source_error = f"bearing/operating condition conflicts with inventory group: {native.File!r} / {native.group!r}"
        path = root / "raw" / DATASET / native.File
        if not path.resolve().is_relative_to(root / "raw" / DATASET):
            source_error = f"original recording resolves outside immutable PU directory: {native.File!r}"
        identity = dict(metadata_id=int(native.Id), original_file=native.File, bearing_id=native.group,
                        operating_condition=condition, metadata_fs=native.Sample_rate,
                        metadata_length=native.Sample_lenth)
        records.append(observe_record(path, identity, source_error))
        if count == 1 or count % 100 == 0 or count == len(selected):
            print(f"PU observations {count}/{len(selected)}; Id {int(native.Id)}; {records[-1]['observation_status']}", flush=True)
    observations = pd.DataFrame(records)
    output.mkdir(parents=True, exist_ok=True)
    observations.to_csv(output / OUTPUTS[0], index=False, mode="x")
    group_summary(observations).to_csv(output / OUTPUTS[1], index=False, mode="x")
    failures = int(observations.observation_status.eq("failed").sum())
    usage = resource.getrusage(resource.RUSAGE_SELF)
    summary = dict(status="completed_with_failures" if failures else "completed", qualification="UNRESOLVED",
        dataset=DATASET, dataset_id=DATASET_ID, sentinel_id=SENTINEL_ID, expected_records=EXPECTED_RECORDS,
        inspected_records=len(records), failed_records=failures, coverage_complete=len(records) == EXPECTED_RECORDS,
        data_root=str(root), metadata_file=str(root / args.metadata_file), inventory=str(args.inventory.resolve()),
        output_dir=str(output), argv=argv, code_commit=args.code_commit,
        elapsed_seconds=time.monotonic() - started, cpu_user_seconds=usage.ru_utime - usage_before.ru_utime,
        cpu_system_seconds=usage.ru_stime - usage_before.ru_stime,
        peak_rss_bytes=int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)),
        environment=dict(platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
                         scipy=scipy.__version__, pandas=pd.__version__),
        interpretation="Stored timestamp exact nonuniformity is descriptive; no clock tolerance is assumed. "
                       "span_average_rate_hz and rate_difference_hz are conditional on X.Data being seconds. "
                       "Blank or absent Unit is preserved; physical clock, scale and acquisition qualification remain unresolved.")
    with (output / OUTPUTS[2]).open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--metadata-file", default="metadata.xlsx")
    parser.add_argument("--code-commit", help="ordinary source revision recorded by the invoking operator")
    args = parser.parse_args(argv)
    try:
        return run(args, list(sys.argv) if argv is None else ["scripts.tii_pu_observations", *argv])
    except (OSError, ValueError, KeyError) as exc:
        print(f"PU observation request failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
