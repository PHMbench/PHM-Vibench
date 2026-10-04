"""Independently recompute complete-run primary metrics from native predictions.

Uses saved logits and acquisition IDs only: no model, support prototypes or
checkpoint selection. Effects are paired within declared episodes, not estimates
of between-machine uncertainty. Selectively adapted from the remote summary.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ARMS = {"E1": ("A2", "A3", "A4", "A6", "A7"),
        "E2": ("A2", "A3", "A4", "A7"),
        "E3": ("A4", "A6", "A7", "A8", "A7_prior_only", "A7_init_only")}
IDENTITY = ("dataset", "fold", "shots", "seed", "draw")
PRIMARY = ("base_acc", "novel_acc", "joint_acc", "harmonic", "a0_base_acc", "signed_base_accuracy_loss", "base_only_acc", "intrusion")


def _identity(row: dict) -> tuple[str, ...]:
    return tuple(str(row.get(field, "UNSPECIFIED")) if field == "dataset" else str(row[field]) for field in IDENTITY)


def _key(row: dict) -> tuple[str, ...]:
    return _identity(row) + (str(row["arm"]), str(row["step"]))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _evaluation(path: Path, class_order: list[int]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {field: data[field] for field in ("logits", "label", "group", "class_order",
                                                  "representation", "source_representation", "source_base_logits")}
    labels, groups, logits = arrays["label"], arrays["group"], arrays["logits"]
    if any(a.ndim != 1 or a.dtype.kind not in "iu" for a in (labels, groups, arrays["class_order"])):
        raise ValueError("Native labels, groups and class order must be integer vectors.")
    if not len(labels) or len(groups) != len(labels) or logits.shape != (len(labels), len(class_order)):
        raise ValueError("Native evaluation arrays have incompatible shapes.")
    if not np.array_equal(arrays["class_order"], class_order) or set(labels.tolist()) != set(class_order):
        raise ValueError("Native label/class order differs from the declared base/novel ontology.")
    for field in ("logits", "representation", "source_representation", "source_base_logits"):
        values = arrays[field]
        if values.ndim != 2 or len(values) != len(labels) or not np.isfinite(values).all():
            raise ValueError(f"Invalid or nonfinite native {field}.")
    for group in np.unique(groups):
        if len(np.unique(labels[groups == group])) != 1:
            raise ValueError("A query acquisition has multiple labels.")
    return arrays


def _balanced(labels: np.ndarray, groups: np.ndarray, classes: list[int], success: np.ndarray) -> float:
    per_class = []
    for label in classes:
        selected = labels == label
        class_groups = np.unique(groups[selected])
        if not len(class_groups):
            raise ValueError("Missing query class.")
        per_class.append(np.mean([success[selected & (groups == group)].mean() for group in class_groups]))
    return float(np.mean(per_class))


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(directory: Path, out: Path | None = None) -> Path:
    directory = directory.resolve()
    status = json.loads((directory / "status.json").read_text())
    cfg = json.loads((directory / "config.resolved.json").read_text())
    experiment = status.get("experiment")
    if status.get("status") != "completed" or status.get("query_evaluated") is not True or cfg.get("query_evaluated") is not True:
        raise ValueError("Only completed query-evaluated runs can be summarized; retain incomplete/failed rows.")
    if experiment not in ARMS or set(cfg.get("actual_arms", [])) != set(ARMS[experiment]) or len(cfg["actual_arms"]) != len(ARMS[experiment]):
        raise ValueError("Require the complete formal experiment arm grid, including A4; tuning subsets are not benchmarks.")
    destination = out.resolve() if out else directory / "summary"
    if destination.exists():
        raise FileExistsError("Summary output exists; choose a new directory without overwriting prior analysis.")
    expected, references = {}, {}
    for index, episode in enumerate(cfg["episodes"]):
        identity = _identity(episode)
        if identity in references:
            raise ValueError("Duplicate episode identity in resolved configuration.")
        folder = directory / f"episode-{index:04d}"
        spec = json.loads((folder / "source.settings.json").read_text())
        settings = json.loads((folder / "adaptation.settings.json").read_text())
        base, novel = spec["base_classes"], episode["novel_classes"]
        if not base or not novel or len(set(base + novel)) != len(base + novel):
            raise ValueError("Base/novel ontology must be nonempty and disjoint.")
        reference = _evaluation(folder / "A4-0.evaluation.npz", base + novel)
        if reference["source_base_logits"].shape != (len(reference["label"]), len(base)):
            raise ValueError("Native source-base logits do not match the declared base class order.")
        if not np.allclose(reference["representation"], reference["source_representation"], atol=1e-6, rtol=1e-6):
            raise ValueError("A4 must retain the saved source representation.")
        references[identity] = (base, novel, reference)
        for arm in ARMS[experiment]:
            family = "A7" if arm.startswith("A7_") else arm
            steps = [0] if arm == "A4" else settings["trajectory_steps"][arm] if experiment == "E2" and arm in {"A2", "A3"} else [settings["steps"][family]]
            if not steps or len(set(steps)) != len(steps) or any(type(step) is not int or step < 0 or step > settings["steps"][family] for step in steps):
                raise ValueError("Invalid or duplicate source-selected checkpoint schedule.")
            for step in steps:
                expected[identity + (arm, str(step))] = folder / f"{arm}-{step}.evaluation.npz"
    if not expected:
        raise ValueError("Resolved configuration has no episodes.")
    metrics = _read_csv(directory / "metrics.csv")
    if len(metrics) != len({_key(row) for row in metrics}) or {_key(row) for row in metrics} != set(expected):
        raise ValueError("Missing, duplicate or unplanned metric endpoints; complete grid required.")
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in _read_csv(directory / "predictions.csv"):
        buckets[_key(row)].append(row)
    if set(buckets) != set(expected):
        raise ValueError("Missing or unplanned prediction endpoints.")
    recomputed = []
    for metric in metrics:
        key = _key(metric)
        base, novel, reference = references[key[:5]]
        native = _evaluation(expected[key], base + novel)
        for field in ("label", "group", "class_order", "source_representation", "source_base_logits"):
            if not np.array_equal(native[field], reference[field]):
                raise ValueError("Paired arms do not share native query observations or source representations.")
        rows = buckets[key]
        observations = [int(row["observation"]) for row in rows]
        if len(observations) != len(set(observations)) or set(observations) != set(range(len(native["label"]))):
            raise ValueError("Duplicate, missing or truncated prediction observations.")
        order = np.asarray(base + novel)
        joint = order[native["logits"].argmax(axis=1)]
        base_prediction = np.asarray(base)[native["logits"][:, :len(base)].argmax(axis=1)]
        source_prediction = np.asarray(base)[reference["source_base_logits"].argmax(axis=1)]
        for row in rows:
            index = int(row["observation"])
            actual = tuple(int(row[field]) for field in ("label", "group", "joint_prediction", "base_prediction", "source_base_prediction"))
            required = (int(native["label"][index]), int(native["group"][index]), int(joint[index]), int(base_prediction[index]), int(source_prediction[index]))
            if actual != required:
                raise ValueError("CSV predictions or source-base mapping differ from native scores.")
        labels, groups = native["label"], native["group"]
        ab = _balanced(labels, groups, base, joint == labels)
        an = _balanced(labels, groups, novel, joint == labels)
        values = {"base_acc": ab, "novel_acc": an,
                  "joint_acc": (len(base)*ab + len(novel)*an)/len(base + novel), "harmonic": 2 * ab * an / (ab + an) if ab + an else 0.,
                  "a0_base_acc": _balanced(labels, groups, base, source_prediction == labels),
                  "base_only_acc": _balanced(labels, groups, base, base_prediction == labels),
                  "intrusion": _balanced(labels, groups, base, (base_prediction == labels) & np.isin(joint, novel))}
        values["signed_base_accuracy_loss"] = values["a0_base_acc"] - ab
        if any(not np.isfinite(float(metric[field])) or abs(float(metric[field]) - value) > 2e-6 for field, value in values.items()):
            raise ValueError("Native predictions do not reproduce the primary metrics.")
        if abs(values["base_only_acc"] - ab - values["intrusion"]) > 2e-6:
            raise ValueError("Joint/base-only/novel-intrusion error decomposition failed.")
        recomputed.append({**dict(zip(IDENTITY + ("arm", "step"), key)), **values})
    pairs = []
    if experiment != "E2":
        grouped: dict[tuple, dict] = defaultdict(dict)
        for row in recomputed:
            grouped[_identity(row)][row["arm"]] = row
        for identity, arms in grouped.items():
            method = arms["A7"]
            for control in ARMS[experiment]:
                if control == "A7":
                    continue
                baseline = arms[control]
                pairs.append({**dict(zip(IDENTITY, identity)), "control": control,
                              "method_step": method["step"], "control_step": baseline["step"],
                              **{f"{name}_difference": method[name] - baseline[name] for name in ("base_acc", "novel_acc", "joint_acc", "harmonic", "signed_base_accuracy_loss")}})
    destination.mkdir(parents=True, exist_ok=False)
    _write_csv(destination / "metrics.recomputed.csv", IDENTITY + ("arm", "step") + PRIMARY, recomputed)
    _write_csv(destination / "paired_effects.csv", IDENTITY + ("control", "method_step", "control_step", "base_acc_difference", "novel_acc_difference", "joint_acc_difference", "harmonic_difference", "signed_base_accuracy_loss_difference"), pairs)
    (destination / "summary.json").write_text(json.dumps({
        "source_run": str(directory), "experiment": experiment, "metric_rows": len(recomputed), "paired_rows": len(pairs),
        "source_prediction_reference": "Unscaled native source-base logits, checked against base class order and shared native observation identities",
        "interpretation": "Class/acquisition-balanced descriptive results conditional on declared folds/checkpoints; no checkpoint selection, independence claim or uncertainty estimate.",
        "trajectory_policy": "E2 validates every declared checkpoint and emits no selected-endpoint paired effect."
    }, indent=2) + "\n")
    print(destination)
    return destination


