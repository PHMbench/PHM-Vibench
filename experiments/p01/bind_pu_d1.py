"""Bind the frozen PU D1 to existing metadata and H5 keys, without waveform access."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import random
import subprocess
import sys

import h5py
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CLASSES = ["healthy", "inner_race_or_IR_dominant_mixed", "outer_race_or_OR_dominant_mixed"]
SEEDS = [42, 123, 456]
PARTITIONS = ["update", "validation", "assessment", "test"]
COUNTS = {0: [3, 1, 1, 1], 1: [6, 2, 2, 3], 2: [6, 2, 2, 3]}
DOMAINS = {
    0: "1500rpm, 0.7Nm, 1000N",
    1: "900rpm, 0.7Nm, 1000N",
    2: "1500rpm, 0.1Nm, 1000N",
    3: "1500rpm, 0.7Nm, 400N",
}


def write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bind(data_root: Path, template: Path, output: Path) -> None:
    from experiments.p01.fusion_data import read_records
    from experiments.p01.preflight_fusion import run as preflight

    metadata = data_root / "metadata.xlsx"
    h5_path = data_root / "RM_027_PU.h5"
    readme = data_root / "README.md"
    if not readme.is_file():
        raise FileNotFoundError(readme)
    workbook = pd.ExcelFile(metadata)
    if workbook.sheet_names != ["Sheet1"]:
        raise ValueError(f"The frozen input expects only Sheet1, observed {workbook.sheet_names}")
    frame = pd.read_excel(workbook, sheet_name="Sheet1")
    selected = frame.loc[frame["Name"].eq("RM_027_PU")].copy()
    if len(selected) != 2559 or selected["Id"].duplicated().any():
        raise ValueError("The bound PU population must contain its 2559 unique acquisitions.")
    if set(selected["Sample_rate"]) != {64000} or set(selected["Channel"]) != {3}:
        raise ValueError("PU sampling or stored-channel metadata changed.")
    for domain, description in DOMAINS.items():
        if set(selected.loc[selected["Domain_id"].eq(domain), "Domain_description"]) != {description}:
            raise ValueError(f"PU domain {domain} no longer matches its declared physical condition.")
    selected["unit_id"] = selected["File"].str.split("/").str[0]
    grouped = selected.groupby("unit_id")
    if (grouped["Label"].nunique() != 1).any() or (grouped["Domain_id"].nunique() != 4).any():
        raise ValueError("Each physical bearing must retain one original label and all four conditions.")
    group_labels = grouped["Label"].first()
    rng = random.Random(20260919)
    partition = {}
    for label in range(3):
        groups = sorted(group_labels.index[group_labels.eq(label)].tolist())
        if len(groups) != sum(COUNTS[label]):
            raise ValueError(f"Physical-bearing count for label {label} changed.")
        rng.shuffle(groups)
        offset = 0
        for split, count in zip(PARTITIONS, COUNTS[label]):
            partition.update({group: split for group in groups[offset:offset + count]})
            offset += count
    selected["split"] = selected["unit_id"].map(partition)
    shapes = []
    with h5py.File(h5_path, "r") as signals:
        for row in selected.to_dict("records"):
            key = str(int(row["Id"]))
            if key not in signals:
                raise KeyError(f"Missing PU acquisition {key}")
            item = signals[key]
            shape = item.shape  # Dataset metadata only; never item[:] or item[()].
            if len(shape) != 3 or shape[1:] != (3, 1) or shape[0] < 8193:
                raise ValueError(f"{key}: expected (L, 3, 1) supporting two distinct 8192-point windows; got {shape}")
            shapes.append(dict(Id=key, unit_id=row["unit_id"], split=row["split"],
                               domain=int(row["Domain_id"]), length=shape[0], channels=shape[1],
                               trailing_axis=shape[2], dtype=str(item.dtype)))

    # All outputs are derived, private artifacts, separate from the immutable data.
    output.mkdir(parents=True, exist_ok=False)
    config_root = output / "configs"
    config_root.mkdir()
    write_csv(output / "protocol.csv", [dict(Id=int(row.Id), unit_id=row.unit_id, split=row.split)
              for row in selected.itertuples()])
    write_csv(output / "h5_shapes.csv", shapes)
    dataset = dict(name="D1_PU", format="vibench_h5", metadata_file=str(metadata),
                   h5_file=str(h5_path), select={"Name": ["RM_027_PU"]},
                   columns=dict(id="Id", label="Label", domain="Domain_id", unit_id="unit_id",
                                split="split", sample_rate_hz="Sample_rate", rotation_speed_rpm="Domain_description"),
                   rotation_speed_pattern=r"(?P<rpm>900|1500)rpm, (?:0\.1|0\.7)Nm, (?:400|1000)N",
                   protocol_file=str(output / "protocol.csv"), source_domains=[0, 2, 3], domain_sequence=[1])
    data = dict(model=dict(num_classes=3, class_names=CLASSES),
                data=dict(layout="LC", squeeze_axes=[2], channel_indices=[2], window_size=8192, windows_per_unit=2),
                datasets=[dataset])
    write_yaml(config_root / "data.yaml", data)

    model = yaml.safe_load(template.read_text(encoding="utf-8"))
    model["model"].update(device="cpu", num_classes=3, head_type="linear", head_hidden_dim=16,
                           checkpoint_kind="reference", checkpoint_path=str(output / "reference" / "selected_candidate.pt"),
                           reference_temperature=1.0, head_frobenius_cap=5.0)
    model["model"]["reference_config"].update(in_channels=1, in_dim=8192, out_dim=8192, num_classes=3)
    model["loss"].update(tau=1.0, brier_weight=0.25, domain_temperature=0.25,
                         lambda_delta=0.1, reduction="worst_source", risk_reference="relative",
                         consistency_target="correction")
    model["assessment"].update(delta=0.05, candidate_count=4, target_scope="same_distribution")
    write_yaml(config_root / "model.yaml", model)
    reference = copy.deepcopy(model["model"]["reference_config"])
    reference.update(type="X_model", name="TSPN", device="cpu")
    write_yaml(config_root / "p0.yaml", {"model": reference})

    arms = {}
    for arm in ["MLP16", "O", "UO", "RO", "RC"]:
        config = copy.deepcopy(model)
        if arm in {"MLP16", "O"}:
            config["loss"].update(reduction="mean_source", lambda_delta=0.0)
        if arm == "MLP16":
            config["model"].update(branches=[], head_type="mlp")
        if arm in {"UO", "RO"}:
            config["loss"]["consistency_target"] = "candidate"
        if arm == "UO":
            config["loss"]["risk_reference"] = "absolute"
        arms[arm] = config
        write_yaml(config_root / "arms" / f"{arm}.yaml", config)

    diagnostics = {}
    for name, width in [("linear", None), ("MLP8", 8), ("MLP32", 32)]:
        config = copy.deepcopy(arms["MLP16"])
        config["model"].update(head_type="linear" if width is None else "mlp", head_hidden_dim=width or 16)
        diagnostics[name] = config
    config = copy.deepcopy(arms["O"])
    config["model"]["head_type"] = "mlp"
    diagnostics["MLP16_all"] = config
    for branch in model["model"]["branches"]:
        config = copy.deepcopy(arms["O"])
        config["model"]["branches"] = [copy.deepcopy(branch)]
        diagnostics[f"only_{branch['name']}"] = config
    for name, config in diagnostics.items():
        write_yaml(config_root / "diagnostics" / f"{name}.yaml", config)

    def checkpoint(arm: str, seed: int) -> str:
        return str(output / "core" / arm / f"seed_{seed}" / "selected_candidate.pt")

    final_candidates = [dict(name=f"{arm}_seed_{seed}", arm=arm, seed=seed, kind="model", checkpoint=checkpoint(arm, seed))
                        for arm in arms for seed in SEEDS]
    plan = dict(data_config=str(config_root / "data.yaml"), dataset="D1_PU", class_names=CLASSES,
                mode="independent", rule="moments", bound="bernstein", scope="source_mixture",
                delta_total=0.05, delta_shift=0.0, selection_alpha_grid=[i / 10 for i in range(11)],
                d1_controls=dict(tau=1.0, selection_predictor="candidate", selection_brier_weight=0.25,
                                 nonlinear_hidden_dim=16, seed=42),
                reference_development_group_files=[], development_group_files=[],
                candidates=[dict(name="temperature", kind="temperature", checkpoint=checkpoint("MLP16", 42),
                                 temperature_grid=[0.5, 0.75, 1.0, 1.5, 2.0])]
                           + [dict(name=arm, kind="model", checkpoint=checkpoint(arm, 42)) for arm in ["MLP16", "O", "RC"]],
                final_candidates=final_candidates)
    plan_path = config_root / "d1_plan.yaml"
    write_yaml(plan_path, plan)
    records = read_records(dataset, data)
    if len(records) != 2559:
        raise ValueError("The maintained metadata reader did not preserve the bound population.")

    benchmark_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (output / "command.json").write_text(json.dumps(dict(argv=sys.argv, python=sys.executable,
         benchmark_commit_at_binding=benchmark_commit, mode="metadata_and_h5_shapes_only"), indent=2) + "\n")
    assignments = "\n".join(f"- {split}: {', '.join(sorted(g for g, s in partition.items() if s == split))}"
                              for split in PARTITIONS)
    branches = ", ".join(branch["name"] for branch in model["model"]["branches"])
    (output / "input_binding.md").write_text(f"""# D1 PU input binding

Dataset selected once from physical identity, class/condition coverage and local H5 availability; no performance screening.
README: `{readme}`. Metadata: `{metadata}`, Sheet1 (49,855 rows, 20 columns).
Selected Name=RM_027_PU: 2,559 unique acquisitions, 32 physical bearings, original ordered labels {CLASSES}.
H5: `{h5_path}`; direct canonical integer Id keys. Shape metadata only was inspected.
Original H5 shape (L,3,1), float64; observed L={min(r['length'] for r in shapes)}…{max(r['length'] for r in shapes)}.
Explicit singleton squeeze axis 2, layout LC, original vibration channel 2; reader channel order is current1/current2/vibration.

Global group is the first File path component (bearing code), supported by the specimen profile and measuring_log PDFs
under `{data_root / 'raw' / 'RM_027_PU'}`. In particular K001/measuring_log_K001.pdf names bearing K001 and
64 kHz vibration/current sampling. Trial numbers are acquisitions, never independent groups.
Cross-condition acquisitions of one bearing stay in one partition. Stratified sorted bearing lists were shuffled
in original label order 0,1,2 using one Python random.Random(20260919); counts by label are {COUNTS}.
{assignments}

Source domains 0/2/3: 1500 rpm at (0.7 Nm,1000 N)/(0.1 Nm,1000 N)/(0.7 Nm,400 N).
Unseen target domain 1: 900 rpm,0.7 Nm,1000 N. RPM is extracted by a declared strict full-match from existing
Domain_description values; it is the documented operating-condition speed, not a newly measured speed trace.
Sample_rate is the existing 64000 Hz value. No metadata label, unit, waveform, or original file was rewritten.
Healthy/IR-dominant/OR-dominant group counts are 6/13/13 in each condition. KB23/KB24 retain label1 and KB27 label2.
KA08/domain2 has 19 acquisitions; other bearing/condition cells have 20. No missing acquisition was synthesized.

Window length 8192 is the next power of two at least ceil(64000/15)=4267 samples: 0.128 s,
1.92 revolutions at target 900 rpm and 3.2 at source 1500 rpm. The maintained window function takes
two evenly spaced windows per acquisition (start/end); no resampling or normalization is added.
The pair is a random circular shift by 1…16 samples, an independent RNG(seed+10000), maximum 0.25 ms.
The physical defect label is invariant to time origin; periodic extension introduces a seam, so this is not
proof of an exact newly acquired physical trace and is not speed/load consistency. Source-only window sanity remains pending.

Reference architecture/filter arguments and operator definitions come unchanged from `{template}` except
the explicit 8192-point input/output interval. Approved branches: {branches}.
Reference checkpoint planned at `{output / 'reference' / 'selected_candidate.pt'}`; no compatible checkpoint
or completed clean reference-development history exists in this binding. Reference temperature fixed to 1.0.
No placeholder weights or fabricated history CSV were created. Source-only clean reference training must happen first.
Independent assessment mode is predeclared for that future clean run; independence is NOT yet established.
History lists remain empty until actual run histories exist; initial preflight must remain blocked on missing history.
Core slots are MLP16/O/UO/RO/RC × seeds42/123/456. K=4 assessment bank is seed42 temperature/MLP16/O/RC.
UO/RO and other seeds are direct empirical comparisons. Diagnostics are source-validation seed42 only.
Training, assessment, permanent test, and source waveform sanity have not been performed by this binding command.
Artifacts here contain private acquisition/group identifiers and are not authorized for public upload.
Benchmark HEAD at binding: {benchmark_commit}; later execution must record its actual implementation commit.
""", encoding="utf-8")
    try:
        preflight(plan_path, config_root / "model.yaml", output / "preflight_initial")
    except ValueError as error:
        expected = "Complete reference history has not been supplied; choose empirical explicitly or retrain cleanly."
        if str(error) != expected:
            raise
        (output / "preflight_initial" / "failure.json").write_text(json.dumps(dict(
            status="blocked", stage="preflight", reason=str(error), exit_status=1,
            waveform_access=False, reference_training_performed=False), indent=2) + "\n")
        print(f"Metadata binding completed; expected preflight blocker: {error}")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bind(args.data_root.resolve(), args.template.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
