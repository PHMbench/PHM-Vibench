"""Offline D1 estimates from frozen window predictions; never loads a model or H5.

CLI inputs are the exporter's existing exports.json list, not a second run registry.
Condition sets are explicit, for example {"test":{"source":["0","1"],"unseen":["2"]}}.
All confidence intervals are descriptive, conditional on the frozen predictors.
Source trainer outputs can instead be declared with repeated --source-run ARM:SEED:RUNDIR.
For validation/direct/alpha=1 only, deployed is an in-memory alias of candidate;
this is not an adoption decision and does not modify the source prediction file.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.p01.fusion_data import summarize_rows

SEEDS = (42, 123, 456)
CORE = ("MLP16", "O", "UO", "RO", "RC")
CONTRASTS = {"Delta_pipe": ("O", "MLP16"), "Delta_loss": ("RC", "O"),
             "Delta_prior": ("RO", "UO"), "Delta_response": ("RC", "RO")}
METRICS = ("ce", "brier", "accuracy", "macro_f1")
PREDICTORS = ("raw", "candidate", "deployed")
BOOTSTRAPS = 2000
ANALYSIS_SEED = 20260919


@dataclass
class Artifact:
    spec: dict[str, Any]
    arrays: dict[str, np.ndarray]
    acquisitions: list[dict[str, Any]]
    groups: list[dict[str, Any]]
    classes: list[str]


def load_artifact(spec: Mapping[str, Any]) -> Artifact:
    """Validate stable CE inputs and align observations by literal identity."""
    required = {"name", "arm", "seed", "split", "path", "alpha", "role"}
    if required - spec.keys():
        raise ValueError(f"Prediction descriptor lacks {sorted(required - spec.keys())}")
    if spec["role"] not in {"direct", "source_selected", "raw", "adopted"}:
        raise ValueError("Unknown frozen predictor role.")
    if spec["arm"] == "MLP16_all" and (spec["split"] != "validation" or int(spec["seed"]) != 42):
        raise ValueError("MLP16_all is only the seed42 source-validation diagnostic.")
    alpha = float(spec["alpha"])
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Frozen alpha must lie in [0,1].")
    with np.load(spec["path"], allow_pickle=False) as saved:
        p = {key: saved[key] for key in saved.files}
    missing_deployed = {"deployed_probs", "deployed_log_probs"} - p.keys()
    if missing_deployed:
        if missing_deployed != {"deployed_probs", "deployed_log_probs"} or not (
                spec["split"] == "validation" and spec["role"] == "direct" and alpha == 1):
            raise ValueError("Missing deployed predictions are allowed only for validation/direct/alpha=1.")
        p["deployed_probs"] = p["candidate_probs"]
        p["deployed_log_probs"] = p["candidate_log_probs"]
    if "class_names" not in p:
        if not np.array_equal(p["raw_class_names"], p["candidate_class_names"]):
            raise ValueError("Raw and candidate class order differ.")
        p["class_names"] = p["raw_class_names"]
    names = np.asarray(p["class_names"])
    if names.ndim != 1 or names.dtype.kind not in "US" or len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("Unique ordered literal class names are required.")
    for key in ("raw_class_names", "candidate_class_names"):
        if key in p and not np.array_equal(p[key], names):
            raise ValueError("Class-order aliases disagree.")
    n, classes = len(p["labels"]), len(names)
    labels = np.asarray(p["labels"])
    if not n or labels.shape != (n,) or labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels >= classes)):
        raise ValueError("Integer labels must match the declared class order.")
    identity_fields = ("domains", "group_ids", "acquisition_ids", "window_ids")
    for key in identity_fields:
        if p[key].shape != (n,) or p[key].dtype.kind not in "US" or np.any(p[key] == ""):
            raise ValueError(f"Invalid literal identity field: {key}")
    identities = list(zip(*(p[key].tolist() for key in identity_fields)))
    if len(set(identities)) != n:
        raise ValueError("Repeated window identity in a prediction artifact.")
    order = np.asarray(sorted(range(n), key=identities.__getitem__))
    for key in (*identity_fields, "labels"):
        p[key] = p[key][order]
    for predictor in PREDICTORS:
        prob = np.asarray(p[predictor + "_probs"], dtype=float)
        lp = np.asarray(p[predictor + "_log_probs"], dtype=float)
        if prob.shape != (n, classes) or lp.shape != prob.shape:
            raise ValueError("Window probability/log-probability shape mismatch.")
        if not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)) or not np.allclose(prob.sum(1), 1, atol=1e-6, rtol=0):
            raise ValueError("Probabilities must be finite and already normalized.")
        if not np.isfinite(lp).all() or not np.allclose(np.logaddexp.reduce(lp, axis=1), 0, atol=2e-6, rtol=0):
            raise ValueError("Finite normalized stable log probabilities are required; do not clip CE inputs.")
        if not np.allclose(np.exp(lp), prob, atol=2e-7, rtol=2e-6):
            raise ValueError("Stable log probabilities do not match exported probabilities.")
        p[predictor + "_probs"], p[predictor + "_log_probs"] = prob[order], lp[order]
    mixture = (1 - alpha) * p["raw_probs"] + alpha * p["candidate_probs"]
    if not np.allclose(mixture, p["deployed_probs"], atol=2e-7, rtol=2e-6):
        raise ValueError("Exported deployment differs from its frozen coefficient.")
    for key in ("arm", "seed", "split", "alpha", "role"):
        if key in p and p[key].item() != spec[key]:
            raise ValueError(f"Descriptor and artifact disagree on {key}.")
    acquisitions = acquisition_estimates(p)
    groups = group_estimates(acquisitions, classes)
    recorded = dict(spec)
    for key in ("checkpoint", "benchmark_commit", "code_commit", "model_config", "config_snapshot"):
        if key in p:
            value = p[key].item()
            if key in recorded and recorded[key] != value:
                raise ValueError(f"Descriptor and artifact disagree on {key}.")
            recorded[key] = value
    if missing_deployed:
        recorded["deployed_alias"] = "candidate; validation direct alpha=1, not independent adoption"
    return Artifact(recorded, p, acquisitions, groups, names.tolist())


def source_run_descriptors(runs: Sequence[str]) -> list[dict[str, Any]]:
    """Bind only explicitly named source runs, including requested missing slots."""
    result = []
    for value in runs:
        fields = value.split(":", 2)
        if len(fields) != 3 or not fields[0] or not fields[2]:
            raise ValueError("Declare each source run as ARM:SEED:RUNDIR.")
        arm, seed_text, directory = fields
        seed = int(seed_text)
        root = Path(directory).expanduser().resolve()
        spec = dict(name=f"{arm}_seed_{seed}", arm=arm, seed=seed, split="validation", role="direct", alpha=1.,
                    path=str(root / "selected_source_validation_windows.npz"), source_run=str(root))
        snapshots = {filename: str(root / filename) for filename in
                     ("command.json", "model_config.yaml", "data_config.yaml", "result_scope.json")
                     if (root / filename).is_file()}
        if snapshots:
            spec["source_snapshots"] = snapshots
        result.append(spec)
    return result


def acquisition_estimates(p: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    owners: dict[tuple[str, str], str] = {}
    for i, (domain, group, acquisition) in enumerate(zip(p["domains"], p["group_ids"], p["acquisition_ids"])):
        if owners.setdefault((domain, acquisition), group) != group:
            raise ValueError("An acquisition belongs to multiple physical groups.")
        buckets[(domain, group, acquisition)].append(i)
    rows = []
    classes = p["raw_probs"].shape[1]
    for (domain, group, acquisition), indices in sorted(buckets.items()):
        y = p["labels"][indices]
        if len(set(y.tolist())) != 1:
            raise ValueError("Classification aggregation needs a constant-label acquisition.")
        raw = p["raw_probs"][indices]
        candidate = p["candidate_probs"][indices]
        v = candidate - raw
        onehot = np.eye(classes)[y]
        a = float(np.mean(np.sum(v * v, axis=1)))
        b = float(np.mean(np.sum((onehot - raw) * v, axis=1)))
        raw_class = int(raw.mean(0).argmax())
        for predictor in PREDICTORS:
            prob = p[predictor + "_probs"][indices]
            lp = p[predictor + "_log_probs"][indices]
            prediction = int(prob.mean(0).argmax())
            rows.append(dict(domain=str(domain), unit_id=str(group), acquisition_id=str(acquisition),
                             label=int(y[0]), predictor=predictor, prediction=prediction, windows=len(y),
                             ce=float(-lp[np.arange(len(y)), y].mean()),
                             brier=float(np.sum((prob - onehot) ** 2, axis=1).mean()),
                             A=a, b=b, repair=float(raw_class != y[0] and prediction == y[0]),
                             damage=float(raw_class == y[0] and prediction != y[0]),
                             change=float(raw_class != prediction),
                             mean_probabilities=prob.mean(0).tolist()))
    return rows


def group_estimates(rows: Sequence[Mapping[str, Any]], classes: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["domain"], row["predictor"], row["unit_id"])].append(row)
    result = []
    for (domain, predictor, group), part in sorted(buckets.items()):
        cm = np.zeros((classes, classes))
        for row in part:
            cm[row["label"], row["prediction"]] += 1 / len(part)
        result.append(dict(domain=domain, predictor=predictor, unit_id=group,
                           acquisitions=len(part), windows=sum(row["windows"] for row in part),
                           **{key: float(np.mean([row[key] for row in part]))
                              for key in ("ce", "brier", "A", "b", "repair", "damage", "change")},
                           confusion_matrix=cm))
    return result


def bootstrap_counts(incidence: Mapping[str, Sequence[str]], *, repeats: int = BOOTSTRAPS,
                     seed: int = ANALYSIS_SEED) -> tuple[list[str], np.ndarray]:
    """One global-group multiplicity shared across conditions, predictors and seeds."""
    groups = sorted(incidence)
    strata: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for i, group in enumerate(groups):
        strata[tuple(sorted(incidence[group]))].append(i)
    rng = np.random.default_rng(seed)
    counts = np.zeros((repeats, len(groups)), dtype=np.int64)
    for mask in sorted(strata):
        columns = np.asarray(strata[mask])
        draws = rng.integers(len(columns), size=(repeats, len(columns)))
        for j, column in enumerate(columns):
            counts[:, column] = (draws == j).sum(1)
    return groups, counts


def condition_metrics(rows: Sequence[Mapping[str, Any]], groups: Sequence[str], counts: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized group bootstrap; point estimates are checked against the evaluator."""
    lookup = {group: i for i, group in enumerate(groups)}
    weights = counts[:, [lookup[row["unit_id"]] for row in rows]]
    total = weights.sum(1)
    if np.any(total == 0):
        raise ValueError("A bootstrap sample lost an entire condition.")
    result = {key: (weights @ np.asarray([row[key] for row in rows])) / total
              for key in ("ce", "brier", "A", "b", "repair", "damage", "change")}
    cm = np.einsum("bg,gij->bij", weights, np.stack([row["confusion_matrix"] for row in rows]))
    diagonal = np.diagonal(cm, axis1=1, axis2=2)
    denominator = cm.sum(1) + cm.sum(2)
    result["accuracy"] = diagonal.sum(1) / cm.sum((1, 2))
    result["macro_f1"] = np.divide(2 * diagonal, denominator, out=np.zeros_like(denominator),
                                    where=denominator != 0).mean(1)
    return result


def _aligned(reference: Artifact, other: Artifact) -> None:
    for key in ("class_names", "labels", "domains", "group_ids", "acquisition_ids", "window_ids"):
        if not np.array_equal(reference.arrays[key], other.arrays[key]):
            raise ValueError(f"Cannot pair artifacts: different {key}.")
    for key in ("raw_probs", "raw_log_probs"):
        if not np.allclose(reference.arrays[key], other.arrays[key], rtol=2e-6, atol=2e-7):
            raise ValueError("Compared artifacts do not share the same frozen reference.")


def _interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [.025, .975])
    return float(low), float(high)


def _metadata(artifact: Artifact) -> dict[str, Any]:
    return {key: artifact.spec[key] for key in ("name", "arm", "seed", "split", "role", "alpha")}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value.tolist() if isinstance(value, np.ndarray) else value)
                             if isinstance(value, (list, dict, np.ndarray)) else value for key, value in row.items()})


def run(predictions: Sequence[Mapping[str, Any]], output: str | Path, *,
        condition_sets: Mapping[str, Mapping[str, Sequence[str]]],
        adoption_mode: str | None = None) -> dict[str, Any]:
    """Analyze an explicit export list without discovering/selecting successful runs."""
    if not predictions:
        raise ValueError("The frozen prediction list is empty.")
    identities = [(item["split"], item["name"]) for item in predictions]
    if len(set(identities)) != len(identities):
        raise ValueError("Repeated frozen predictor name within a split.")
    missing = [dict(item) for item in predictions if not Path(item["path"]).is_file()]
    artifacts = [load_artifact(item) for item in predictions if Path(item["path"]).is_file()]
    if not artifacts:
        raise ValueError("No raw prediction artifact is available.")
    adopted = [a for a in artifacts if a.spec["role"] == "adopted"]
    if adopted and adoption_mode not in {"independent", "empirical"}:
        raise ValueError("Adoption analysis requires the mode frozen before assessment.")
    by_split: dict[str, list[Artifact]] = defaultdict(list)
    partitions: dict[str, str] = {}
    for artifact in artifacts:
        by_split[artifact.spec["split"]].append(artifact)
        for group in set(artifact.arrays["group_ids"]):
            if partitions.setdefault(str(group), artifact.spec["split"]) != artifact.spec["split"]:
                raise ValueError("A global physical group crosses analyzed partitions.")
    acquisition_rows, group_rows, metric_rows, mechanism_rows = [], [], [], []
    seed_rows, contrast_rows, contrast_seed_rows, coverage = [], [], [], []
    for split, entries in sorted(by_split.items()):
        reference = entries[0]
        for entry in entries[1:]:
            _aligned(reference, entry)
        domains = sorted(set(reference.arrays["domains"].tolist()))
        sets = {name: list(map(str, ds)) for name, ds in condition_sets[split].items()}
        if not sets or set(domains) != {d for ds in sets.values() for d in ds}:
            raise ValueError("Explicit condition sets must cover exactly the exported population.")
        if any(not ds or len(set(ds)) != len(ds) for ds in sets.values()):
            raise ValueError("Condition sets must be nonempty without repeated conditions.")
        scopes = {**{"condition:" + d: [d] for d in domains}, **sets}
        incidence: dict[str, set[str]] = defaultdict(set)
        for group, domain in zip(reference.arrays["group_ids"], reference.arrays["domains"]):
            incidence[str(group)].add(str(domain))
        global_groups, boot = bootstrap_counts(incidence)
        counts = np.vstack((np.ones(len(global_groups), dtype=np.int64), boot))
        estimates: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
        direct: dict[tuple[str, int], Artifact] = {}
        for entry in entries:
            meta = _metadata(entry)
            if entry.spec["role"] == "direct":
                slot = (entry.spec["arm"], int(entry.spec["seed"]))
                if slot in direct:
                    raise ValueError("Duplicate direct arm/seed slot.")
                direct[slot] = entry
            acquisition_rows.extend(dict(meta, **row) for row in entry.acquisitions)
            group_rows.extend(dict(meta, **row) for row in entry.groups)
            canonical = {(row["domain"], row["predictor"]): row
                         for row in summarize_rows(entry.acquisitions, len(entry.classes))}
            for predictor in PREDICTORS:
                by_condition = {}
                for domain in domains:
                    part = [row for row in entry.groups if row["domain"] == domain and row["predictor"] == predictor]
                    values = condition_metrics(part, global_groups, counts)
                    for metric in METRICS:
                        np.testing.assert_allclose(values[metric][0], canonical[(domain, predictor)][metric],
                                                   rtol=1e-12, atol=1e-12)
                    by_condition[domain] = values
                for scope, ds in scopes.items():
                    values = {metric: np.mean([by_condition[d][metric] for d in ds], axis=0)
                              for metric in by_condition[ds[0]]}
                    estimates[(entry.spec["name"], predictor, scope)] = values
                    represented = sorted({int(row["label"]) for row in entry.acquisitions if row["domain"] in ds})
                    support = {d: sorted({int(row["label"]) for row in entry.acquisitions if row["domain"] == d}) for d in ds}
                    for metric in METRICS:
                        lower, upper = _interval(values[metric][1:])
                        metric_rows.append(dict(meta, predictor=predictor, scope=scope, metric=metric,
                                                value=float(values[metric][0]), lower=lower, upper=upper,
                                                represented_classes=represented, condition_class_support=support,
                                                classes=len(entry.classes), global_groups=len({g for g in incidence if incidence[g] & set(ds)})))
                    a, b = float(values["A"][0]), float(values["b"][0])
                    mechanism_rows.append(dict(meta, predictor=predictor, scope=scope, A=a, b=b,
                                               sqrt_A=float(np.sqrt(a)), b_over_sqrt_A=b / np.sqrt(a) if a > 0 else None,
                                               alpha_sqrt_A=float(entry.spec["alpha"] * np.sqrt(a)),
                                               repair=float(values["repair"][0]), damage=float(values["damage"][0]),
                                               prediction_change_rate=float(values["change"][0])))
        available = sorted(direct)
        missing_slots = [{"arm": arm, "seed": seed} for arm in CORE for seed in SEEDS if (arm, seed) not in direct]
        coverage.append(dict(split=split, complete_core=not missing_slots, missing_core_slots=missing_slots,
                             available_direct_slots=[dict(arm=arm, seed=seed) for arm, seed in available]))
        for arm in CORE:
            present = [seed for seed in SEEDS if (arm, seed) in direct]
            for scope in scopes:
                for metric in METRICS:
                    values = [estimates[(direct[(arm, seed)].spec["name"], "candidate", scope)][metric] for seed in present]
                    complete = len(present) == len(SEEDS)
                    average = np.mean(values, axis=0) if complete else None
                    lower, upper = _interval(average[1:]) if complete else (None, None)
                    seed_rows.append(dict(split=split, arm=arm, scope=scope, metric=metric, seeds=present,
                                          status="complete" if complete else "partial", finite_seed_mean=float(average[0]) if complete else None,
                                          seed_sd=float(np.std([v[0] for v in values], ddof=1)) if complete else None,
                                          lower=lower, upper=upper))
        for contrast, (treatment, control) in CONTRASTS.items():
            present = [seed for seed in SEEDS if (treatment, seed) in direct and (control, seed) in direct]
            for scope in scopes:
                for metric in METRICS:
                    differences = []
                    for seed in present:
                        left = estimates[(direct[(treatment, seed)].spec["name"], "candidate", scope)][metric]
                        right = estimates[(direct[(control, seed)].spec["name"], "candidate", scope)][metric]
                        delta = left - right
                        differences.append(delta)
                        lower, upper = _interval(delta[1:])
                        contrast_rows.append(dict(split=split, contrast=contrast, treatment=treatment, control=control,
                                                  seed=seed, scope=scope, metric=metric, value=float(delta[0]), lower=lower, upper=upper))
                    complete = len(present) == len(SEEDS)
                    average = np.mean(differences, axis=0) if complete else None
                    lower, upper = _interval(average[1:]) if complete else (None, None)
                    contrast_seed_rows.append(dict(split=split, contrast=contrast, scope=scope, metric=metric, seeds=present,
                                                   status="complete" if complete else "partial",
                                                   finite_seed_mean=float(average[0]) if complete else None,
                                                   seed_sd=float(np.std([v[0] for v in differences], ddof=1)) if complete else None,
                                                   lower=lower, upper=upper))
        if split == "validation" and ("MLP16_all", 42) in direct and ("MLP16", 42) in direct:
            for scope in scopes:
                for metric in METRICS:
                    left = estimates[(direct[("MLP16_all", 42)].spec["name"], "candidate", scope)][metric]
                    right = estimates[(direct[("MLP16", 42)].spec["name"], "candidate", scope)][metric]
                    delta = left - right
                    lower, upper = _interval(delta[1:])
                    contrast_rows.append(dict(split=split, contrast="Delta_same_readout", treatment="MLP16_all",
                                              control="MLP16", seed=42, scope=scope, metric=metric,
                                              value=float(delta[0]), lower=lower, upper=upper,
                                              interpretation="source-validation diagnostic only; not a permanent-test claim"))
        for entry in (entry for entry in entries if entry.spec["role"] == "adopted"):
            for scope in scopes:
                for control, contrast in (("candidate", "Delta_adoption"), ("raw", "Delta_adoption_vs_raw")):
                    for metric in METRICS:
                        deployed = estimates[(entry.spec["name"], "deployed", scope)][metric]
                        comparison = estimates[(entry.spec["name"], control, scope)][metric]
                        delta = deployed - comparison
                        lower, upper = _interval(delta[1:])
                        contrast_rows.append(dict(split=split, contrast=contrast, treatment=entry.spec["arm"] + ":adopted",
                                                  control=entry.spec["arm"] + ":direct" if control == "candidate" else "raw",
                                                  seed=entry.spec["seed"], scope=scope, metric=metric,
                                                  value=float(delta[0]), lower=lower, upper=upper, alpha=entry.spec["alpha"],
                                                  mode=adoption_mode, independent_assessment="applicable" if adoption_mode == "independent" else "not_applicable"))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    for filename, rows in (("acquisitions.csv", acquisition_rows), ("groups.csv", group_rows),
                           ("metrics.csv", metric_rows), ("seed_summary.csv", seed_rows),
                           ("paired_contrasts.csv", contrast_rows), ("contrast_seed_summary.csv", contrast_seed_rows),
                           ("mechanism.csv", mechanism_rows)):
        _write_csv(root / filename, rows)
    report = dict(status="complete" if not missing and all(row["complete_core"] for row in coverage) else "partial",
                  missing_artifacts=missing, coverage=coverage, bootstrap_repeats=BOOTSTRAPS, analysis_seed=ANALYSIS_SEED,
                  bootstrap_unit="global physical group, stratified by condition-incidence mask; paired across all arms and seeds",
                  intervals="descriptive 95% percentile intervals conditional on frozen predictors; small group counts can be unstable",
                  classification="acquisition mean probability argmax, group-balanced confusion, fixed full class space",
                  condition_aggregation="equal-weight mean of condition-specific metrics, not a pooled F1",
                  seed_aggregation="finite three-seed mean and sample SD; seeds are not independent physical specimens",
                  Delta_pipe_alias="Delta_repr has the same numeric definition; processing-pipeline contrast, not pure representation causality",
                  adoption_intervals="conditional on the already selected candidate and coefficient",
                  class_names=artifacts[0].classes)
    (root / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    recorded = {(entry.spec["split"], entry.spec["name"]): entry.spec for entry in artifacts}
    input_predictions = [recorded.get((item["split"], item["name"]), dict(item)) for item in predictions]
    (root / "inputs.json").write_text(json.dumps(dict(predictions=input_predictions, condition_sets=condition_sets,
                                                      adoption_mode=adoption_mode), indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--inputs", help="Existing exports.json list from frozen export")
    inputs.add_argument("--source-run", action="append", metavar="ARM:SEED:RUNDIR",
                        help="Explicit source-only run; repeat for every core and diagnostic slot")
    parser.add_argument("--condition-sets", required=True, help="JSON mapping split -> named condition set -> condition IDs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--adoption-mode", choices=("independent", "empirical"))
    args = parser.parse_args()
    predictions = json.loads(Path(args.inputs).read_text()) if args.inputs else source_run_descriptors(args.source_run)
    report = run(predictions, args.output,
                 condition_sets=json.loads(Path(args.condition_sets).read_text()), adoption_mode=args.adoption_mode)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
