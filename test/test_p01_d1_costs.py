"""Artifact-only cost accounting: timing boundaries and shared label populations."""
from __future__ import annotations

import csv
import json

import pytest
import yaml

from experiments.p01.summarize_d1_costs import summarize


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture
def artifacts(tmp_path):
    root = tmp_path/"run"
    configs = root/"configs"
    diagnostics = configs/"diagnostics"
    diagnostics.mkdir(parents=True)
    for name in ["linear", "MLP8", "MLP32", "MLP16_all", *[f"only_view_{i}" for i in range(7)]]:
        (diagnostics/f"{name}.yaml").write_text("model: {}\n")
    data = dict(data=dict(windows_per_unit=2), datasets=[dict(name="D1", source_domains=[0, 2], domain_sequence=[1])])
    plan = dict(dataset="D1", mode="independent")
    (configs/"data.yaml").write_text(yaml.safe_dump(data))
    (configs/"d1_plan.yaml").write_text(yaml.safe_dump(plan))
    fit = dict(groups=2, acquisitions=4, windows=8, labelled_acquisitions=4)
    selection = dict(groups=1, acquisitions=2, windows=4, labelled_acquisitions=2)
    p0 = dict(status="classifier_fitted", fit=fit, selection=selection, trainable_parameters=5,
              training_seconds=2., selection_seconds=3., export_seconds=.1, optimizer_steps=1000)
    _json(root/"reference"/"result_scope.json", p0)
    _json(root/"reference"/"execution_status.json", dict(exit_status=0, filesystem_observed_interval_seconds=9.,
                                                       interval_boundary="filesystem mtimes, not process monotonic timer"))
    for seed in (42, 123):
        _json(root/"core"/"O"/f"seed_{seed}"/"result_scope.json", dict(status="candidate_fitted", fit_access=fit,
              selection_access=selection, trainable_parameters=3, total_parameters=8, reference_parameters=5,
              training_seconds=4., source_selection_seconds=5., source_materialization_seconds=.2, optimizer_steps=1000))
        _json(root/"core"/"O"/f"seed_{seed}"/"execution_status.json", dict(exit_status=0, wall_seconds=12., benchmark_commit="fixture"))
    _json(root/"baselines"/"ResNet1D"/"seed_42"/"result_scope.json", dict(status="classifier_fitted", fit=fit, selection=selection,
          trainable_parameters=100, training_seconds=7., selection_seconds=8., export_seconds=.2))
    _json(root/"baselines"/"ResNet1D"/"seed_42"/"execution_status.json", dict(exit_status=0, monotonic_wall_seconds=20.))
    firnet = dict(arm="FIRNet", seeds=[42, 123, 456], status="blocked", reason="No legal original implementation.", training_performed=False)
    _json(root/"firnet_status.json", firnet)
    frozen = root/"frozen"
    frozen.mkdir()
    (frozen/"data_config.yaml").write_text(yaml.safe_dump(data))
    records = []
    for split, groups in (("update", ["u0", "u1"]), ("validation", ["v"]), ("assessment", ["a"]), ("test", ["t"])):
        for domain in ("0", "2", "1"):
            for group in groups:
                records.append(dict(split=split, domain=domain, unit_id=group, acquisition_id=f"{domain}-{group}"))
    _json(frozen/"records.json", records)
    specs = [dict(arm="ResNet1D", seed=42, role="direct", bundle=str(frozen/"bundles"/"resnet.pt")),
             dict(arm="O", seed=42, role="adopted", bundle=str(frozen/"bundles"/"adopted.pt"))]
    _json(frozen/"F1.json", dict(plan=plan, predictors=specs, source_selection_seconds=30.))
    _json(frozen/"F2.json", dict(mode="independent", predictors=specs, assessment_seconds=40.))
    _json(frozen/"test"/"runtime.json", dict(inference_seconds=50., note="current invocation only; prior attempts retain logs"))
    latencies = []
    for index, (spec, alpha, total) in enumerate(zip(specs, [1., 0.], [105, 8])):
        directory = root/"latency"/str(index)
        paths = [dict(path=name, median_ms=1., q1_ms=.8, q3_ms=1.2, iqr_ms=.4,
                      peak_allocated_bytes=1000, repeats_ms=[1.]*100) for name in ("p0", "direct_q", "deployed")]
        _json(directory/"latency.json", dict(bundle=spec["bundle"], deployment=dict(kind="model", alpha=alpha),
              paths=paths, total_parameters=total, device="physical GPU0 / cuda:0", gpu="fixture", dtype="torch.float32",
              input_shape=[1, 128, 1], warmup=20, repeats=100, group_id="v", acquisition_id="0-v", domain="0",
              split="validation", window_id="0", io_boundary="input already on GPU; excludes checkpoint/data I/O"))
        latencies.append(directory)
    return root, frozen, latencies, firnet


def test_distinct_pools_and_timing_boundaries(artifacts, tmp_path):
    root, frozen, latencies, firnet = artifacts
    output = tmp_path/"summary"
    notes = summarize(root, frozen, latencies, output)
    assert notes["expected_source_runs"] == 30  # p0 + 15 core + 11 diagnostic + 3 ResNet.
    assert notes["completed_source_runs"] == 4
    assert notes["firnet_status"] == firnet
    labels = _rows(output/"label_budget.csv")
    global_rows = {row["pool"]: row for row in labels if row["domain"] == "all"}
    assert global_rows["source_fit"]["groups"] == "2"
    assert global_rows["source_fit"]["acquisitions"] == "4"  # Repeated source runs do not multiply labels.
    assert global_rows["assessment"]["groups"] == "1"
    assert global_rows["test"]["groups"] == "1"  # Same physical group in all three conditions.
    assert global_rows["test"]["acquisitions"] == "3"
    assert global_rows["distinct_accessed_union"]["groups"] == "5"
    assert global_rows["distinct_accessed_union"]["acquisitions"] == "11"
    costs = _rows(output/"cost.csv")
    p0 = {row["stage"]: row for row in costs if row["arm"] == "p0"}
    assert p0["process"]["seconds"] == "" and p0["process"]["measurement_status"] == "missing"
    assert p0["filesystem_observed_interval"]["seconds"] == "9.0"
    assert "not process" in p0["filesystem_observed_interval"]["time_boundary"]
    assert p0["fit"]["seconds"] == "2.0" and p0["source_selection"]["seconds"] == "3.0"
    phases = {row["stage"]: row for row in costs if row["arm"] == "shared"}
    assert "bundle" in phases["assessment_and_adoption"]["time_boundary"]
    assert "adopted-source" in phases["assessment_and_adoption"]["time_boundary"]
    assert phases["assessment_and_adoption"]["labelled_acquisitions"] == "2"
    assert "current invocation" in phases["test_export_invocation"]["time_boundary"]
    missing = next(row for row in costs if row["arm"] == "UO" and row["seed"] == "456" and row["stage"] == "fit")
    assert missing["seconds"] == missing["direct_parameters"] == missing["reference_parameters"] == ""
    assert missing["status"] == "missing"


def test_latency_keeps_loaded_and_executed_parameters_separate(artifacts, tmp_path):
    root, frozen, latencies, _ = artifacts
    missing = root/"latency"/"not_yet_measured"
    summarize(root, frozen, [*latencies, missing], tmp_path/"summary")
    rows = _rows(tmp_path/"summary"/"latency.csv")
    baseline = next(row for row in rows if row["arm"] == "ResNet1D" and row["path"] == "direct_q")
    assert baseline["total_loaded_parameters"] == "105" and baseline["executed_parameters"] == "100"
    deployed = next(row for row in rows if row["role"] == "adopted" and row["path"] == "deployed")
    assert deployed["executed_parameters"] == "5" and deployed["total_loaded_parameters"] == "8"
    assert deployed["warmup"] == "20" and deployed["repeats"] == "100"
    assert rows[-1]["measurement_status"] == "missing" and rows[-1]["median_ms"] == ""


def test_before_freezing_does_not_invent_per_condition_or_protected_access(artifacts, tmp_path):
    root, frozen, _, _ = artifacts
    missing_frozen = root/"not_prepared"
    summarize(root, missing_frozen, [], tmp_path/"summary")
    labels = _rows(tmp_path/"summary"/"label_budget.csv")
    global_fit = next(row for row in labels if row["pool"] == "source_fit" and row["domain"] == "all")
    assert global_fit["groups"] == "2" and global_fit["counts_basis"].startswith("observed")
    condition_fit = next(row for row in labels if row["pool"] == "source_fit" and row["domain"] == "0")
    assert condition_fit["groups"] == condition_fit["acquisitions"] == ""
    assessment = next(row for row in labels if row["pool"] == "assessment" and row["domain"] == "all")
    assert assessment["access_status"] == "reserved_not_accessed" and assessment["groups"] == ""
    assert _rows(tmp_path/"summary"/"latency.csv") == []


def test_empirical_route_does_not_charge_reserved_assessment_labels(artifacts, tmp_path):
    root, frozen, _, _ = artifacts
    f2 = json.loads((frozen/"F2.json").read_text())
    f2["mode"] = "empirical"
    _json(frozen/"F2.json", f2)
    summarize(root, frozen, [], tmp_path/"summary")
    labels = _rows(tmp_path/"summary"/"label_budget.csv")
    union = next(row for row in labels if row["pool"] == "distinct_accessed_union" and row["domain"] == "all")
    assert union["groups"] == "4" and union["acquisitions"] == "9"
    assessment = next(row for row in labels if row["pool"] == "assessment" and row["domain"] == "all")
    assert assessment["access_status"] == "not_applicable_empirical"


def test_failed_run_is_not_success_and_source_access_is_preserved(artifacts, tmp_path):
    root, frozen, _, _ = artifacts
    _json(root/"core"/"O"/"seed_42"/"execution_status.json", dict(exit_status=1, wall_seconds=12.))
    notes = summarize(root, frozen, [], tmp_path/"summary")
    assert notes["completed_source_runs"] == 3
    costs = _rows(tmp_path/"summary"/"cost.csv")
    row = next(row for row in costs if row["arm"] == "O" and row["seed"] == "42" and row["stage"] == "fit")
    assert row["status"] == "failed" and row["acquisitions"] == "4"


def test_training_summaries_preserve_scale_direction_and_missing_values(artifacts, tmp_path):
    root, frozen, _, _ = artifacts
    run = root/"core"/"O"/"seed_42"
    def write(name, rows):
        with (run/name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write("training_features.csv", [dict(epoch=0, step=step, branch="view", feature_dim=4,
          sampled_windows=12, mean_scaled_squared_norm=energy, median_scaled_norm=norm,
          input_weight_parameters=12, input_weight_gradient_norm=5+2*step, raw_input_weight_norm=2,
          effective_input_weight_norm=1, effective_output_weight_norm=3)
          for step, energy, norm in [(0, 2, 1), (1, 6, 2)]])
    write("training_domains.csv", [dict(epoch=0, step=step, domain="0", candidate_risk=1+step,
          reference_risk=.5, excess=.5+step, weight=.25+step*.5) for step in (0, 1)])
    write("training_reference_prior.csv", [dict(epoch=0, step=step, domain_i="0", domain_j="2",
          reference_risk_difference_over_rho=value) for step, value in [(0, .4), (1, -.8)]])
    write("training_responses.csv", [dict(epoch=0, step=step, domain="0", A=4*step, b=-2*step,
          sqrt_A=2*step, b_over_sqrt_A=None if step == 0 else -1,
          delta_p0_squared_norm=step, delta_q_squared_norm=2+step, delta_v_squared_norm=3+step,
          delta_p0_mean_vector=json.dumps([step, -step]),
          delta_q_mean_vector=json.dumps([1+step, -1-step]),
          delta_v_mean_vector=json.dumps([1, -1])) for step in (0, 1)])
    write("training_batches.csv", [dict(epoch=0, step=step, loss=1, risk_objective=1, diagnostic_excess=1,
          max_source_excess=1, source_envelope=1, correction_consistency=2+4*step,
          candidate_output_consistency=5+2*step, reference_output_consistency=step, pair_penalty=2+4*step) for step in (0, 1)])
    output = tmp_path/"summary"
    summarize(root, frozen, [], output)
    features = [row for row in _rows(output/"training_features_summary.csv") if row["arm"] == "O" and row["seed"] == "42"]
    metrics = {row["metric"]: row for row in features}
    assert float(metrics["mean_unscaled_squared_norm"]["mean"]) == 16
    assert float(metrics["median_unscaled_norm"]["median"]) == 3
    assert float(metrics["input_weight_gradient_norm"]["mean"]) == 6
    assert metrics["mean_scaled_squared_norm"]["n"] == "2"
    assert metrics["mean_scaled_squared_norm"]["sampled_window_visits"] == "24"
    assert "not pooled-window median" in metrics["median_scaled_norm"]["derivation"]
    mechanisms = [row for row in _rows(output/"training_mechanism_summary.csv") if row["arm"] == "O" and row["seed"] == "42"]
    ratio = next(row for row in mechanisms if row["metric"] == "b_over_sqrt_A")
    assert ratio["n"] == "1" and ratio["n_missing"] == "1" and float(ratio["mean"]) == -1
    prior = next(row for row in mechanisms if row["metric"] == "reference_risk_difference_over_rho")
    assert float(prior["mean"]) == pytest.approx(-.2)
    vector = next(row for row in mechanisms if row["metric"] == "delta_q_mean_vector" and row["component"] == "1")
    assert float(vector["mean"]) == -1.5
    penalties = {row["metric"]: float(row["mean"]) for row in mechanisms if row["family"] == "objective"}
    assert penalties["correction_consistency"] == 4 and penalties["candidate_output_consistency"] == 6
    assert all("not frozen population estimates" in row["summary_scope"] for row in features+mechanisms)
    assert not ({"group_id", "acquisition_id", "window_id"} & set(features[0]))
    assert not ({"group_id", "acquisition_id", "window_id"} & set(mechanisms[0]))


def test_missing_training_files_are_missing_and_failed_runs_are_not_summarized(artifacts, tmp_path):
    root, frozen, _, _ = artifacts
    _json(root/"core"/"O"/"seed_42"/"execution_status.json", dict(exit_status=1))
    output = tmp_path/"summary"
    notes = summarize(root, frozen, [], output)
    features = _rows(output/"training_features_summary.csv")
    assert not any(row["arm"] == "O" and row["seed"] == "42" for row in features)
    missing = next(row for row in features if row["arm"] == "O" and row["seed"] == "123")
    assert missing["measurement_status"] == "missing" and missing["mean"] == missing["median"] == ""
    assert str(root/"core"/"O"/"seed_123"/"training_features.csv") in notes["missing_artifacts"]
