"""Constructed software checks; none is an independent scientific repeat."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.tii_motivation_simulation import (
    RunSpec, aggregate_results, generate_fixture, query_metrics, readout_diagnostic, run,
)


def small_spec() -> RunSpec:
    return RunSpec(steps=2, validation_interval=1, batch_size=4, train_records=8,
                   validation_records=8, target_records=32, repeats=((101, 0),),
                   tasks=("common_informative",))


def test_generator_uses_independent_balanced_roles_and_exact_missing_increment():
    spec = small_spec()
    fixture = generate_fixture("common_informative", 101, spec)
    repeated = generate_fixture("common_informative", 101, spec)
    changed = generate_fixture("common_informative", 202, spec)
    rows = fixture["rows"]
    assert len({r["recording_id"] for r in rows}) == len(rows)
    assert all(r["group"] == r["recording_id"] == r["file_id"] for r in rows)
    for source in (1, 2):
        for role in ("source_train", "source_val"):
            batch = fixture["inventories"][role][source]
            selection = [r for r in rows if r["dataset"] == source and r["role"] == role]
            assert {(a, b): sum(r["latent_common"] == a and r["latent_increment"] == b for r in selection)
                    for a in (0, 1) for b in (0, 1)} == {(0, 0): 2, (0, 1): 2, (1, 0): 2, (1, 1): 2}
            torch.testing.assert_close(batch["x"], repeated["inventories"][role][source]["x"], rtol=0, atol=0)
            assert not torch.equal(batch["x"], changed["inventories"][role][source]["x"])
            if source == 2:
                assert torch.count_nonzero(batch["incremental"]) == 0
                assert torch.count_nonzero(batch["availability"]) == 0
                assert all(r["label"] == 1 - r["latent_common"] for r in selection)
    target = fixture["inventories"]["target"][3]
    support_groups = {g for g, role in zip(target["group"], target["role"]) if role == "support"}
    query_groups = {g for g, role in zip(target["group"], target["role"]) if role == "query"}
    assert len(support_groups) == 10
    assert len(query_groups) == 22
    assert not support_groups & query_groups


def test_opposing_task_changes_only_full_source_and_target_label_mechanism():
    spec = small_spec()
    c = generate_fixture("common_informative", 101, spec)
    p = generate_fixture("increment_informative", 101, spec)
    for key in c["raw_blocks"]:
        torch.testing.assert_close(c["raw_blocks"][key], p["raw_blocks"][key], rtol=0, atol=0)
    for row in p["rows"]:
        assert row["label"] == (1 - row["latent_common"] if row["dataset"] == 2 else row["latent_increment"])


def test_scale_compensation_preserves_predictions_and_refit_is_solved(tmp_path: Path):
    fixture = generate_fixture("common_informative", 101, small_spec())
    target = fixture["inventories"]["target"][3]
    rng = np.random.default_rng(8)
    features = rng.normal(size=(32, 8)) + target["y"].numpy()[:, None]
    result = readout_diagnostic(features, target, tmp_path)
    assert result["logits_max_absolute_difference"] == 0
    assert result["inverse_weight_decisions_equal"]
    assert result["metrics"]["inverse_weight"] == result["metrics"]["original"]
    assert result["original"]["gradient_inf"] <= 1e-6
    assert result["scaled_refit"]["gradient_inf"] <= 1e-6
    artifact = np.load(tmp_path / "heads_and_query_logits.npz")
    assert artifact["original_logits"].shape == (22, 2)
    np.testing.assert_array_equal(artifact["original_logits"], artifact["inverse_weight_logits"])


def test_fixed_episode_and_nonfinite_features_fail_without_repair(tmp_path: Path):
    fixture = generate_fixture("common_informative", 101, small_spec())
    target = fixture["inventories"]["target"][3]
    features = np.zeros((32, 8))
    target["role"][0] = "source_train"
    with pytest.raises(ValueError, match="complete, disjoint"):
        readout_diagnostic(features, target, tmp_path)
    with pytest.raises(ValueError, match="finite binary logits"):
        query_metrics(np.array([[np.nan, 0.]]), np.array([0]))
    with pytest.raises(ValueError, match="incomplete"):
        aggregate_results([], small_spec())


def test_cpu_only_and_existing_output_are_explicit_failures(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="explicitly"):
        run(tmp_path / "not_created", small_spec(), ["software-test"])
    assert not (tmp_path / "not_created").exists()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(FileExistsError):
        run(tmp_path, small_spec(), ["software-test"])


def test_complete_native_software_smoke_retains_actual_outputs(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output = tmp_path / "software_smoke"
    run(output, small_spec(), ["software-smoke: two native encoder updates, not a scientific repeat"])
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "completed"
    assert status["encoder_fits"] == 2 and status["head_fits"] == 4
    repeat = output / "common_informative_data101_opt0"
    initialization = json.loads((repeat / "initialization.json").read_text())
    assert initialization["all_state_tensors_exactly_equal"]
    assert initialization["ordinary_first_affine"]["rank"] == 16
    assert (repeat / "matched_initial_state.pt").is_file()
    for arm in ("support", "ordinary"):
        directory = repeat / arm
        fit_status = json.loads((directory / "status.json").read_text())
        checkpoint = torch.load(directory / "selected.pt", weights_only=False)
        assert fit_status["selected_step"] in (0, 1, 2)
        assert checkpoint["source_val_nll"] == fit_status["source_val_nll"]
        assert checkpoint["source_rms"] > 0
        assert set(k.split(".")[2] for k in checkpoint["model_state"] if k.startswith("task_head.mutiple_fc.")) == {"1", "2"}
        curves = pd.read_csv(directory / "source_curve.csv")
        assert set(curves.step) == {0, 1, 2}
        assert len(curves) == 3 * 2 * 3
        assert len(pd.read_csv(directory / "source_selected_predictions.csv")) == 32
        assert len(pd.read_csv(directory / "query_predictions.csv")) == 22
        assert np.load(directory / "frozen_features.npz")["features"].shape == (32, 8)
    assert len(pd.read_csv(output / "fit_results.csv")) == 2
    assert len(pd.read_csv(output / "per_seed_differences.csv")) == 1


def test_failed_head_is_retained_and_not_replaced(tmp_path: Path, monkeypatch):
    import scripts.tii_motivation_simulation as simulation

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    def failed_head(*args, **kwargs):
        raise RuntimeError("deliberate solver failure")
    monkeypatch.setattr(simulation, "fit_head", failed_head)
    output = tmp_path / "failed_software_fixture"
    with pytest.raises(RuntimeError, match="deliberate solver failure"):
        run(output, small_spec(), ["failure-contract-software-test"])
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed" and status["completed_encoder_fits"] == 0
    fit = output / "common_informative_data101_opt0/support"
    assert json.loads((fit / "status.json").read_text())["error"] == "deliberate solver failure"
    assert (fit / "selected.pt").is_file() and (fit / "source_curve.csv").is_file()
    assert not (output / "aggregate.csv").exists()


def test_checkpoint_tie_keeps_earliest_deep_copied_state(tmp_path: Path, monkeypatch):
    import scripts.tii_motivation_simulation as simulation
    from src.data_factory.data_utils import MetadataAccessor
    from src.data_factory.tii_sampling import fit_source_rms

    torch.set_num_threads(1)
    spec = small_spec()
    fixture = generate_fixture("common_informative", 101, spec)
    source_rows = [r for r in fixture["rows"] if r["dataset"] in (1, 2)]
    metadata = MetadataAccessor(pd.DataFrame([
        dict(Id=r["file_id"], Dataset_id=r["dataset"], Label=r["label"],
             Sample_rate=r["native_rate_hz"], Name="software_fixture") for r in source_rows]))
    torch.manual_seed(0)
    joint = simulation.make_task("support", metadata, fit_source_rms(fixture["inventories"]["source_train"]), spec)
    original = {k: v.clone() for k, v in joint.network.state_dict().items()}
    native_diagnostic = simulation.source_diagnostics
    def fixed_tied_score(*args, **kwargs):
        summaries, predictions, _ = native_diagnostic(*args, **kwargs)
        return summaries, predictions, 1.0
    monkeypatch.setattr(simulation, "source_diagnostics", fixed_tied_score)
    result = simulation.run_fit(joint, fixture, 0, spec, tmp_path / "tie_software_fixture")
    assert result["selected_step"] == 0
    checkpoint = torch.load(tmp_path / "tie_software_fixture/selected.pt", weights_only=False)
    for key, value in original.items():
        torch.testing.assert_close(checkpoint["model_state"][key], value, rtol=0, atol=0)
