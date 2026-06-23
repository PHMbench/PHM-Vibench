"""Generate metadata Excel for RM_101_THU_GEARBOX (MCC5-THU)."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py

import pandas as pd


FAULT_LABEL_MAP = {
    "health": 0,
    "gear_pitting": 1,
    "gear_wear": 2,
    "miss_teeth": 3,
    "teeth_break": 4,
    "teeth_crack": 5,
    "teeth_break_and_bearing_inner": 6,
    "teeth_break_and_bearing_outer": 7,
}

SEVERITY_MAP = {"L": 0, "M": 1, "H": 2}
MODE_ORDER = ["speed_circulation", "torque_circulation"]
TORQUE_ORDER = [10, 20]
RPM_ORDER = [1000, 2000, 3000]

METADATA_COLUMNS = [
    "Id",
    "Dataset_id",
    "Name",
    "Description",
    "TYPE",
    "File",
    "Visiable",
    "Label",
    "Label_Description",
    "Fault_level",
    "RUL_label",
    "RUL_label_description",
    "Domain_id",
    "Domain_description",
    "Sample_rate",
    "Sample_lenth",
    "Channel",
    "Fault_Diagnosis",
    "Anomaly_Detection",
    "Remaining_Life",
]


@dataclass
class ParsedFile:
    file_name: str
    fault_type: str
    severity: Optional[str]
    mode: str
    torque_nm: int
    rpm: int


def parse_filename(file_name: str) -> ParsedFile:
    stem = Path(file_name).stem
    tokens = stem.split("_")
    if "speed" in tokens:
        mode_idx = tokens.index("speed")
        mode = "speed_circulation"
    elif "torque" in tokens:
        mode_idx = tokens.index("torque")
        mode = "torque_circulation"
    else:
        raise ValueError(f"Cannot detect mode from file name: {file_name}")

    fault_tokens = tokens[:mode_idx]
    if not fault_tokens:
        raise ValueError(f"Cannot detect fault type from file name: {file_name}")

    severity = None
    if fault_tokens[-1] in SEVERITY_MAP:
        severity = fault_tokens[-1]
        fault_type = "_".join(fault_tokens[:-1])
    else:
        fault_type = "_".join(fault_tokens)

    if fault_type not in FAULT_LABEL_MAP:
        raise ValueError(f"Unknown fault type '{fault_type}' in file: {file_name}")

    condition_tokens = tokens[mode_idx + 2 :]
    if mode == "speed_circulation":
        if len(condition_tokens) != 1:
            raise ValueError(f"Invalid speed condition format in file: {file_name}")
        m = re.match(r"^(\d+)Nm-(\d+)rpm$", condition_tokens[0])
        if not m:
            raise ValueError(f"Invalid speed condition token '{condition_tokens[0]}' in file: {file_name}")
        torque_nm = int(m.group(1))
        rpm = int(m.group(2))
    else:
        if len(condition_tokens) != 2:
            raise ValueError(f"Invalid torque condition format in file: {file_name}")
        m1 = re.match(r"^(\d+)rpm$", condition_tokens[0])
        m2 = re.match(r"^(\d+)Nm$", condition_tokens[1])
        if not (m1 and m2):
            raise ValueError(f"Invalid torque condition tokens '{condition_tokens}' in file: {file_name}")
        rpm = int(m1.group(1))
        torque_nm = int(m2.group(1))

    if mode not in MODE_ORDER or torque_nm not in TORQUE_ORDER or rpm not in RPM_ORDER:
        raise ValueError(
            f"Unsupported domain combo mode={mode}, torque={torque_nm}, rpm={rpm} in file: {file_name}"
        )

    return ParsedFile(
        file_name=file_name,
        fault_type=fault_type,
        severity=severity,
        mode=mode,
        torque_nm=torque_nm,
        rpm=rpm,
    )


def domain_id(mode: str, torque_nm: int, rpm: int) -> int:
    mode_idx = MODE_ORDER.index(mode)
    torque_idx = TORQUE_ORDER.index(torque_nm)
    rpm_idx = RPM_ORDER.index(rpm)
    return mode_idx * (len(TORQUE_ORDER) * len(RPM_ORDER)) + torque_idx * len(RPM_ORDER) + rpm_idx


def count_rows_fast(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def build_metadata(
    raw_dir: Path,
    dataset_id: int,
    dataset_name: str,
    start_id: int,
    sample_rate: int,
    channel_count: int,
    count_sample_length: bool,
    skip_copy: bool,
) -> pd.DataFrame:
    csv_files = sorted([p.name for p in raw_dir.glob("*.csv")])
    if skip_copy:
        csv_files = [f for f in csv_files if " copy" not in Path(f).stem]

    rows = []
    for row_index, file_name in enumerate(csv_files):
        parsed = parse_filename(file_name)
        sample_length = count_rows_fast(raw_dir / file_name) if count_sample_length else 768000
        d_id = domain_id(parsed.mode, parsed.torque_nm, parsed.rpm)

        rows.append(
            {
                "Id": start_id + row_index,
                "Dataset_id": dataset_id,
                "Name": dataset_name,
                "Description": "",
                "TYPE": "csv",
                "File": file_name,
                "Visiable": 1,
                "Label": FAULT_LABEL_MAP[parsed.fault_type],
                "Label_Description": parsed.fault_type,
                "Fault_level": SEVERITY_MAP.get(parsed.severity, 0),
                "RUL_label": 0,
                "RUL_label_description": "",
                "Domain_id": d_id,
                "Domain_description": f"{parsed.mode}; torque={parsed.torque_nm}Nm; speed={parsed.rpm}rpm",
                "Sample_rate": sample_rate,
                "Sample_lenth": sample_length,
                "Channel": channel_count,
                "Fault_Diagnosis": 1,
                "Anomaly_Detection": 1,
                "Remaining_Life": 0,
            }
        )

    df = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    return df


def find_id_conflicts(df: pd.DataFrame, cache_h5: Path) -> list[int]:
    if not cache_h5.exists():
        return []
    id_set = {str(int(v)) for v in df["Id"].tolist()}
    with h5py.File(cache_h5, "r") as h5f:
        overlaps = sorted(int(k) for k in h5f.keys() if k in id_set)
    return overlaps


def parse_args():
    parser = argparse.ArgumentParser(description="Generate gear_metadata.xlsx for RM_101_THU_GEARBOX.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("/home/user/data/PHMbenchdata/PHM-Vibench/raw/RM_101_THU_GEARBOX"),
        help="Directory containing MCC5-THU CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/user/data/PHMbenchdata/PHM-Vibench/gear_metadata.xlsx"),
        help="Output Excel metadata path.",
    )
    parser.add_argument("--dataset-id", type=int, default=101, help="Metadata Dataset_id.")
    parser.add_argument("--name", type=str, default="RM_101_THU_GEARBOX", help="Metadata Name field.")
    parser.add_argument(
        "--start-id",
        type=int,
        default=101001,
        help="Start value for global-unique Id column. Generated IDs are contiguous.",
    )
    parser.add_argument("--sample-rate", type=int, default=12800, help="Sample rate stored in metadata.")
    parser.add_argument("--channel-count", type=int, default=8, help="Channel count stored in metadata.")
    parser.add_argument(
        "--count-sample-length",
        action="store_true",
        help="Count each CSV row number for Sample_lenth (slower, but exact).",
    )
    parser.add_argument(
        "--keep-copy-files",
        action="store_true",
        help="Include files with ' copy' in filename. Default is to skip them.",
    )
    parser.add_argument(
        "--check-cache-path",
        type=Path,
        default=None,
        help="Optional cache.h5 path used to block conflicting Id keys before writing metadata.",
    )
    parser.add_argument(
        "--skip-cache-conflict-check",
        action="store_true",
        help="Skip collision check against cache.h5 keys.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {args.raw_dir}")

    df = build_metadata(
        raw_dir=args.raw_dir,
        dataset_id=args.dataset_id,
        dataset_name=args.name,
        start_id=args.start_id,
        sample_rate=args.sample_rate,
        channel_count=args.channel_count,
        count_sample_length=args.count_sample_length,
        skip_copy=not args.keep_copy_files,
    )

    if not args.skip_cache_conflict_check:
        check_path = args.check_cache_path
        if check_path is None:
            candidate = args.output.parent / "cache.h5"
            if candidate.exists():
                check_path = candidate
        if check_path is not None and check_path.exists():
            conflicts = find_id_conflicts(df, check_path)
            if conflicts:
                preview = conflicts[:10]
                raise RuntimeError(
                    f"Id conflict detected against cache {check_path}. "
                    f"Conflicting keys count={len(conflicts)}, sample={preview}. "
                    "Use a different --start-id."
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(args.output, index=False)

    summary = df.groupby("Label").size().to_dict()
    print(f"[SUCCESS] metadata rows: {len(df)}")
    print(f"[SUCCESS] output: {args.output}")
    print(f"[INFO] id range: {int(df['Id'].min())}..{int(df['Id'].max())}")
    print(f"[INFO] label distribution: {summary}")
    print(f"[INFO] domain ids: {sorted(df['Domain_id'].unique().tolist())}")


if __name__ == "__main__":
    main()
