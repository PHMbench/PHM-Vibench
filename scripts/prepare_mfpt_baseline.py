"""Prepare the pinned public MFPT bearing-test-rig subset for PHMFactory."""

from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

from src.data_factory.reader.RM_007_MFPT import read_record


PROVIDER_REPOSITORY = "https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data"
PROVIDER_REVISION = "d3efefb6ce84fa1ee6c0311f80f7c89cf903ad1d"
PROVIDER_ARCHIVE = f"{PROVIDER_REPOSITORY}/archive/{PROVIDER_REVISION}.zip"
DATASET_NAME = "RM_007_MFPT"
DATASET_ID = 7
LICENSE = "CC BY-NC-SA 4.0"

TRAIN_FILES = (
    "baseline_1.mat",
    "baseline_2.mat",
    "InnerRaceFault_vload_1.mat",
    "InnerRaceFault_vload_2.mat",
    "InnerRaceFault_vload_3.mat",
    "InnerRaceFault_vload_4.mat",
    "InnerRaceFault_vload_5.mat",
    "OuterRaceFault_1.mat",
    "OuterRaceFault_2.mat",
    "OuterRaceFault_vload_1.mat",
    "OuterRaceFault_vload_2.mat",
    "OuterRaceFault_vload_3.mat",
    "OuterRaceFault_vload_4.mat",
    "OuterRaceFault_vload_5.mat",
)

TEST_FILES = (
    "baseline_3.mat",
    "InnerRaceFault_vload_6.mat",
    "InnerRaceFault_vload_7.mat",
    "OuterRaceFault_3.mat",
    "OuterRaceFault_vload_6.mat",
    "OuterRaceFault_vload_7.mat",
)

EXPECTED_FILES = {
    *(f"train_data/{name}" for name in TRAIN_FILES),
    *(f"test_data/{name}" for name in TEST_FILES),
}

FIELDNAMES = (
    "Id",
    "Dataset_id",
    "Name",
    "Type",
    "File",
    "Visiable",
    "Label",
    "Label_Description",
    "Domain_id",
    "Provider_Split",
    "Working_Condition_description",
    "Sample_Rate",
    "Shaft_Rate_Hz",
    "Load",
    "BPFO",
    "BPFI",
    "FTF",
    "BSF",
    "Length",
    "Channels",
    "Description",
    "Fault_Diagnosis",
    "Anomaly_Detection",
    "Remaining_Life",
    "Digital_Twin_Prediction",
    "Source_Repository",
    "Source_Revision",
    "License",
    "Reference",
    "Benchmark_Table",
)


def _label(filename: str) -> tuple[int, str]:
    if filename.startswith("baseline_"):
        return 0, "Normal"
    if filename.startswith("InnerRaceFault_"):
        return 1, "Inner Race Fault"
    if filename.startswith("OuterRaceFault_"):
        return 2, "Outer Race Fault"
    raise ValueError(f"Unable to derive MFPT label from filename {filename!r}.")


def _expected_entries() -> list[tuple[str, str]]:
    return [
        *(("train_data", filename) for filename in TRAIN_FILES),
        *(("test_data", filename) for filename in TEST_FILES),
    ]


def verify_provider_tree(provider_root: str | Path) -> Path:
    """Require the exact 20-file public bearing-test-rig subset."""

    root = Path(provider_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"MFPT provider root does not exist: {root}")
    observed = {
        path.relative_to(root).as_posix()
        for directory in (root / "train_data", root / "test_data")
        if directory.is_dir()
        for path in directory.glob("*.mat")
    }
    missing = sorted(EXPECTED_FILES - observed)
    unexpected = sorted(observed - EXPECTED_FILES)
    if missing or unexpected:
        raise ValueError(
            "MFPT provider file set does not match the pinned 20-file protocol: "
            f"missing={missing}, unexpected={unexpected}."
        )
    return root


def _download_provider(work_dir: Path) -> Path:
    archive = work_dir / "mfpt-provider.zip"
    request = Request(
        PROVIDER_ARCHIVE,
        headers={"User-Agent": "PHMFactory-MFPT-preparation"},
    )
    try:
        with urlopen(request, timeout=180) as response, archive.open("wb") as target:
            shutil.copyfileobj(response, target)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"Unable to download pinned MFPT provider revision {PROVIDER_REVISION}: {exc}"
        ) from exc

    extracted = work_dir / "provider"
    extracted.mkdir()
    try:
        with ZipFile(archive) as bundle:
            bundle.extractall(extracted)
    except (BadZipFile, OSError) as exc:
        raise ValueError(f"Downloaded MFPT provider archive is invalid: {exc}") from exc

    roots = [path for path in extracted.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise ValueError(
            f"Pinned MFPT archive must contain one repository root, observed={roots}."
        )
    return verify_provider_tree(roots[0])


def _metadata_row(
    *,
    file_id: int,
    split: str,
    filename: str,
    record: dict[str, Any],
    source_revision: str,
) -> dict[str, Any]:
    label, label_description = _label(filename)
    domain_id = 0 if split == "train_data" else 1
    sample_rate = float(record["sample_rate_hz"])
    shaft_rate = float(record["shaft_rate_hz"])
    load = float(record["load"])
    relative_file = f"{split}/{filename}"
    return {
        "Id": file_id,
        "Dataset_id": DATASET_ID,
        "Name": DATASET_NAME,
        "Type": "bearing",
        "File": relative_file,
        "Visiable": True,
        "Label": label,
        "Label_Description": label_description,
        "Domain_id": domain_id,
        "Provider_Split": "train" if domain_id == 0 else "test",
        "Working_Condition_description": (
            f"provider_{'train' if domain_id == 0 else 'test'}; "
            f"shaft_rate_hz={shaft_rate:g}; load={load:g}"
        ),
        "Sample_Rate": sample_rate,
        "Shaft_Rate_Hz": shaft_rate,
        "Load": load,
        "BPFO": float(record["BPFO"]),
        "BPFI": float(record["BPFI"]),
        "FTF": float(record["FTF"]),
        "BSF": float(record["BSF"]),
        "Length": int(record["signal"].shape[0]),
        "Channels": int(record["signal"].shape[1]),
        "Description": "MFPT public bearing-test-rig acceleration signal",
        "Fault_Diagnosis": True,
        "Anomaly_Detection": False,
        "Remaining_Life": False,
        "Digital_Twin_Prediction": False,
        "Source_Repository": PROVIDER_REPOSITORY,
        "Source_Revision": source_revision,
        "License": LICENSE,
        "Reference": PROVIDER_REPOSITORY,
        "Benchmark_Table": "MFPT_Official_Train_Test_v1",
    }


def prepare_dataset(
    provider_root: str | Path,
    output_root: str | Path,
    *,
    source_revision: str,
) -> Path:
    """Copy the exact provider files and derive strict PHMFactory metadata."""

    provider = verify_provider_tree(provider_root)
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"MFPT output already exists: {output}. Choose a new directory; "
            "the preparation command never overwrites user data."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{uuid4().hex}"
    raw_root = staging / "raw" / DATASET_NAME
    rows: list[dict[str, Any]] = []

    try:
        for offset, (split, filename) in enumerate(_expected_entries(), start=1):
            source = provider / split / filename
            destination = raw_root / split / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            record = read_record(destination)
            rows.append(
                _metadata_row(
                    file_id=7000 + offset,
                    split=split,
                    filename=filename,
                    record=record,
                    source_revision=source_revision,
                )
            )

        metadata_path = staging / "metadata_mfpt.csv"
        with metadata_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

        if len(rows) != 20:
            raise RuntimeError(f"MFPT preparation expected 20 rows, observed={len(rows)}.")
        train_rows = [row for row in rows if row["Provider_Split"] == "train"]
        test_rows = [row for row in rows if row["Provider_Split"] == "test"]
        if len(train_rows) != 14 or len(test_rows) != 6:
            raise RuntimeError(
                "MFPT preparation produced the wrong provider split sizes: "
                f"train={len(train_rows)}, test={len(test_rows)}."
            )

        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return output / "metadata_mfpt.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the pinned public MFPT provider revision and prepare the exact "
            "20-file bearing-test-rig protocol for PHMFactory."
        )
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New PHMFactory data root to create, for example data/mfpt.",
    )
    parser.add_argument(
        "--provider-checkout",
        default=None,
        help=(
            "Optional existing provider checkout containing train_data/ and test_data/. "
            "This is intended for offline preparation; metadata records the source as "
            "user-provided-checkout rather than claiming the pinned revision."
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.provider_checkout:
        metadata = prepare_dataset(
            args.provider_checkout,
            args.output,
            source_revision="user-provided-checkout",
        )
    else:
        with tempfile.TemporaryDirectory(prefix="phmfactory-mfpt-") as temp:
            provider = _download_provider(Path(temp))
            metadata = prepare_dataset(
                provider,
                args.output,
                source_revision=PROVIDER_REVISION,
            )
    print(f"mfpt_data_root={metadata.parent}")
    print(f"mfpt_metadata={metadata}")
    print("provider_train_files=14")
    print("provider_test_files=6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
