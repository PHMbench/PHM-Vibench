"""Bounded source-input timing and frozen, outcome-stratified explanation cases.

The caller owns source-only access and the final predictor freeze. Neither helper
opens waveform storage, trains a model, or selects a checkpoint or configuration.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import platform
import time
from typing import Any

import numpy as np
import torch


def profile_predictor(model: torch.nn.Module, x: torch.Tensor, device: str | torch.device,
                      *, warmup: int = 5, repeats: int = 20) -> dict[str, Any]:
    """Time the complete deployed forward on one already cached source batch.

Fusion must have its final direct-candidate coefficient (alpha=1). A classifier
wrapper's forward executes only its candidate: its reference, retained for paired
scientific evaluation, is not an inference requirement of that baseline.
"""
    from src.model_factory.X_model.TSPN_fusion import TSPNFusion

    dev = torch.device(device)
    if dev.type not in {"cpu", "cuda"}:
        raise ValueError("Profiling supports explicitly requested CPU or CUDA only.")
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    if x.device != dev or x.ndim != 3 or not x.is_floating_point() or not len(x):
        raise ValueError("Provide a cached floating B,L,C source batch on the requested device.")
    if not torch.isfinite(x).all():
        raise ValueError("Cannot profile a nonfinite source batch.")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (warmup, repeats)) or warmup < 0 or repeats < 1:
        raise ValueError("warmup must be a nonnegative integer and repeats a positive integer.")
    if any(value.device != dev for value in (*model.parameters(), *model.buffers())):
        raise ValueError("Move the frozen model to the explicitly requested device before profiling.")
    is_fusion = isinstance(model, TSPNFusion)
    if is_fusion and float(model.alpha) != 1.:
        raise ValueError("Profile the restored alpha=1 fusion deployment; alpha=0 omits all views.")

    modes = [(module, module.training) for module in model.modules()]
    buffers = [(name, value, value.detach().clone()) for name, value in model.named_buffers()]

    def sync() -> None:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    def measure(forward) -> dict[str, Any]:
        for _ in range(warmup):
            forward(x)
        sync()
        baseline_bytes = None
        if dev.type == "cuda":
            baseline_bytes = torch.cuda.memory_allocated(dev)
            torch.cuda.reset_peak_memory_stats(dev)
        durations = []
        for _ in range(repeats):
            sync()
            start = time.perf_counter()
            forward(x)
            sync()
            durations.append(1000. * (time.perf_counter() - start))
        peak = torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None
        return dict(median_ms=float(np.median(durations)), p95_ms=float(np.percentile(durations, 95)),
                    peak_allocated_bytes=peak, allocated_before_forward_bytes=baseline_bytes,
                    forward_peak_increment_bytes=None if peak is None else peak - baseline_bytes)

    changed = []
    try:
        model.eval()
        with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
            full = measure(model if is_fusion else lambda batch: torch.log_softmax(model(batch), -1))
            # Match the reference's inference probability computation without
            # recomputing or summing an auxiliary reference for ordinary baselines.
            reference = measure(lambda batch: torch.log_softmax(
                model.reference(batch) / float(model.reference_temperature), -1)) if is_fusion else None
    finally:
        with torch.no_grad():
            for name, value, before in buffers:
                if not torch.equal(value, before):
                    changed.append(name)
                    value.copy_(before)
        for module, training in modes:
            module.training = training
    if changed:
        raise RuntimeError(f"Inference changed frozen buffers: {changed}; original buffers restored.")
    return dict(scope="cached_source_input_full_prediction", input_shape=list(x.shape), dtype=str(x.dtype),
                device=str(dev), hardware=torch.cuda.get_device_name(dev) if dev.type == "cuda" else platform.machine(),
                torch_version=str(torch.__version__), torch_threads=torch.get_num_threads(),
                cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
                autocast=False, warmup=warmup, repeats=repeats, full=full, reference_only=reference,
                alpha=float(model.alpha) if is_fusion else None,
                incremental_median_ms=None if reference is None else full["median_ms"] - reference["median_ms"],
                incremental_definition="full median minus independently measured reference median; may be negative from timing noise",
                timing_boundary="complete prediction to log probabilities including all configured operators and readout; excludes I/O and host-to-device transfer",
                memory_boundary="CUDA process allocated peak, including already resident models/input; forward increment excludes preexisting allocation; CPU unavailable")


def select_explanation_cases(arrays: Mapping[str, np.ndarray], *, conditions: Sequence[str],
                             class_names: Sequence[str], seed: int = 42) -> dict[str, Any]:
    """Select one frozen window per class × condition × four correctness outcomes.

Selections are descriptive examples, not population performance estimates. All
declared strata, including empty strata, remain visible. Row permutation does not
change selected identities or the true-versus-highest-wrong-class contrast.
"""
    conditions = list(map(str, conditions))
    names = list(map(str, class_names))
    if not conditions or len(set(conditions)) != len(conditions) or len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("Provide unique declared conditions and at least two ordered class names.")
    labels = np.asarray(arrays["labels"])
    n, classes = len(labels), len(names)
    if labels.shape != (n,) or labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels >= classes)):
        raise ValueError("Labels must index the declared class order.")
    fields = ("domains", "group_ids", "acquisition_ids", "window_ids")
    ids = {key: np.asarray(arrays[key]) for key in fields}
    if any(value.shape != (n,) or value.dtype.kind not in "US" or np.any(value == "") for value in ids.values()):
        raise ValueError("Each window needs literal condition, specimen, acquisition and window identifiers.")
    if set(ids["domains"]) - set(conditions):
        raise ValueError("An exported condition is absent from the frozen declared condition list.")
    identities = list(zip(*(ids[key].tolist() for key in fields)))
    if len(set(identities)) != n:
        raise ValueError("Duplicate window identities cannot define reproducible cases.")
    values = {}
    branch_keys = sorted(key for key in arrays if key.startswith("logit_contribution__"))
    if not branch_keys:
        raise ValueError("Additive branch contributions are required for explanation cases.")
    for key in ("raw_probs", "candidate_probs", "raw_log_probs", "candidate_log_probs", *branch_keys):
        value = np.asarray(arrays[key], dtype=float)
        if value.shape != (n, classes) or not np.isfinite(value).all():
            raise ValueError(f"Invalid finite window-by-class array: {key}")
        values[key] = value
    for prefix in ("raw", "candidate"):
        probability, lp = values[prefix + "_probs"], values[prefix + "_log_probs"]
        if np.any(probability < 0) or not np.allclose(probability.sum(1), 1., atol=1e-6, rtol=0.) or not np.allclose(
                np.exp(lp), probability, atol=2e-7, rtol=2e-6):
            raise ValueError("Case selection requires normalized frozen probabilities and matching log probabilities.")
    descriptors = {}
    for key in sorted(key for key in arrays if key.startswith("descriptor__")):
        value = np.asarray(arrays[key], dtype=float)
        if value.ndim != 2 or len(value) != n or not np.isfinite(value).all():
            raise ValueError(f"Invalid window-by-feature descriptor: {key}")
        descriptors[key.split("__", 1)[1]] = value

    raw_prediction = values["raw_probs"].argmax(1)
    prediction = values["candidate_probs"].argmax(1)
    outcomes = ((True, True, "both_correct"), (True, False, "p0_correct_I_wrong"),
                (False, True, "p0_wrong_I_correct"), (False, False, "both_wrong"))
    rng = np.random.default_rng(seed)
    cases, strata, coverage = [], [], []
    for condition in sorted(conditions):
        for label, name in enumerate(names):
            population = set(ids["group_ids"][labels == label].tolist())
            observed = set(ids["group_ids"][(labels == label) & (ids["domains"] == condition)].tolist())
            coverage.append(dict(condition_id=condition, fault_class_index=label, fault_class=name,
                                 observed_specimens=sorted(observed),
                                 missing_specimens=sorted(population - observed),
                                 population_basis="specimens of this class present anywhere in this frozen artifact"))
            for raw_correct, candidate_correct, outcome in outcomes:
                eligible = np.flatnonzero((ids["domains"] == condition) & (labels == label) &
                                         ((raw_prediction == labels) == raw_correct) &
                                         ((prediction == labels) == candidate_correct)).tolist()
                eligible.sort(key=identities.__getitem__)
                stratum = dict(condition_id=condition, fault_class_index=label, fault_class=name, outcome=outcome,
                               windows=len(eligible), specimens=len({ids["group_ids"][i] for i in eligible}),
                               acquisitions=len({(ids["group_ids"][i], ids["acquisition_ids"][i]) for i in eligible}),
                               status="present" if eligible else "missing")
                strata.append(stratum)
                if not eligible:
                    continue
                index = eligible[int(rng.integers(len(eligible)))]
                wrong = values["candidate_probs"][index].copy()
                wrong[label] = -np.inf
                other = int(wrong.argmax())
                terms = {key.split("__", 1)[1]: float(values[key][index, label] - values[key][index, other])
                         for key in branch_keys}
                direct = float((values["candidate_log_probs"][index, label] - values["candidate_log_probs"][index, other]) -
                               (values["raw_log_probs"][index, label] - values["raw_log_probs"][index, other]))
                reconstructed = sum(terms.values())
                cases.append(dict(condition_id=condition, fault_class_index=label, fault_class=name, outcome=outcome,
                                  physical_specimen_id=str(ids["group_ids"][index]),
                                  acquisition_id=str(ids["acquisition_ids"][index]), window_id=str(ids["window_ids"][index]),
                                  raw_prediction=int(raw_prediction[index]), candidate_prediction=int(prediction[index]),
                                  contrast_class_a=label, contrast_class_b=other,
                                  signed_branch_terms=terms, direct_log_odds_correction=direct,
                                  reconstructed_log_odds_correction=reconstructed,
                                  reconstruction_absolute_error=abs(direct - reconstructed),
                                  descriptors={key: value[index].tolist() for key, value in descriptors.items()}))
    return dict(seed=seed, selection_rule="one uniform window per nonempty class/condition/correctness stratum after specimen/acquisition/window ID sort",
                contrast_rule="true class versus highest candidate-probability wrong class; smallest class index breaks ties",
                case_interpretation="descriptive examples only; strata counts are window counts, not population-balanced effects",
                raw_signal_rendering="not exported; descriptors and signed contributions do not supply waveform or view curves",
                cases=cases, strata=strata, specimen_condition_coverage=coverage)
