"""Synthetic source-only checks using the original TSPN and fixed ResNet1D."""
from __future__ import annotations

import json
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
import yaml

from experiments.p01 import train_source_classifier as trainer
from experiments.p01.fusion_deployment import load_model
from src.model_factory.model_factory import model_factory


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    reference = dict(type="X_model", name="TSPN", device="cpu", num_classes=2,
                     in_channels=1, out_channels=12, scale=1, in_dim=128, out_dim=128,
                     skip_connection=False, internal_instance_normalization=False,
                     signal_processing_configs={"layer1": ["I", "HT", "WF"]},
                     feature_extractor_configs=["RMS", "Std", "AbsMean"],
                     f_c_mu=.2, f_c_sigma=.01, f_b_mu=.05, f_b_sigma=.002)
    baseline = dict(type="CNN", name="ResNet1D", input_dim=1, num_classes=2,
                    block_type="basic", layers=[2, 2, 2, 2], initial_channels=64)
    data = dict(model=dict(num_classes=2, class_names=["healthy", "fault"]),
                data=dict(window_size=128, windows_per_unit=2),
                datasets=[dict(name="fixture", source_domains=["0", "2"], domain_sequence=["1"])])
    for name, contents in (("reference", {"model": reference}), ("baseline", {"model": baseline}), ("data", data)):
        (tmp_path/f"{name}.yaml").write_text(yaml.safe_dump(contents))
    records = []
    for split in ("update", "validation", "assessment", "test"):
        for domain in ("0", "2", "1"):
            for label in (0, 1):
                for acquisition in range(2):
                    records.append(dict(unit_id=f"{split}-{label}", acquisition_id=f"{split}-{domain}-{label}-{acquisition}",
                                        split=split, domain=domain, label=label, sample_rate_hz=64000., rotation_speed_rpm=1500.))
    accesses = []

    def materialize(selected, dataset, cfg):
        result = []
        for record in selected:
            assert record["split"] in {"update", "validation"}
            assert record["domain"] in {"0", "2"}
            accesses.append((record["split"], record["acquisition_id"]))
            time = torch.arange(128, dtype=torch.float32)
            signal = torch.sin(time*(.07+.06*record["label"]))+.1*torch.cos(time*.33)
            windows = torch.stack((signal, signal.roll(3)), dim=0).unsqueeze(-1)
            result.append(dict(record, x=windows))
        return result

    monkeypatch.setattr(trainer, "read_records", lambda dataset, config: records)
    monkeypatch.setattr(trainer, "materialize", materialize)

    def arguments(role, output, extra=()):
        return trainer.parser().parse_args([
            "--role", role, "--model-config", str(tmp_path/f"{role}.yaml"),
            "--data-config", str(tmp_path/"data.yaml"), "--dataset", "fixture",
            "--output", str(tmp_path/output), "--epochs", "1", "--steps-per-epoch", "2", *extra,
        ])

    return arguments, accesses


def test_reference_and_baseline_source_training_checkpoint_and_access(source_fixture, tmp_path):
    arguments, accesses = source_fixture
    reference_dir = trainer.run(arguments("reference", "p0"))
    reference = torch.load(reference_dir/"selected_candidate.pt", weights_only=True)
    assert reference["kind"] == "classifier" and reference["epoch"] == 0
    # The produced p0 is consumed by the actual strict fusion checkpoint path.
    fusion = model_factory(SimpleNamespace(type="X_model", name="TSPN_fusion", device="cpu",
                           num_classes=2, reference_config=reference["model"], checkpoint_kind="reference",
                           checkpoint_path=str(reference_dir/"selected_candidate.pt"), branches=[],
                           use_reference_features=True, reference_temperature=1.), metadata=None)
    assert all(torch.equal(value, fusion.reference.state_dict()[key]) for key, value in reference["state_dict"].items())
    baseline_dir = trainer.run(arguments("baseline", "resnet", [
        "--reference-config", str(reference_dir/"model_config.yaml"),
        "--reference-checkpoint", str(reference_dir/"selected_candidate.pt"), "--pair-shift", "16",
    ]))
    saved = torch.load(baseline_dir/"selected_candidate.pt", weights_only=True)
    assert saved["model"]["layers"] == [2, 2, 2, 2]
    assert all(torch.equal(value, saved["reference_state_dict"][key]) for key, value in reference["state_dict"].items())
    restored, _ = load_model(baseline_dir/"selected_candidate.pt", "cpu")
    assert not restored.training and all(not p.requires_grad for p in restored.parameters())
    with torch.no_grad():
        out = restored.forward_details(torch.ones(2, 128, 1))
    assert out["candidate_logits"].shape == out["raw_logits"].shape == (2, 2)
    assert torch.isfinite(out["candidate_logits"]).all()
    source_sampling = pd.read_csv(reference_dir/"sampling.csv")
    baseline_sampling = pd.read_csv(baseline_dir/"sampling.csv")
    pd.testing.assert_frame_equal(source_sampling.drop(columns="pair_shift"), baseline_sampling.drop(columns="pair_shift"))
    rng = random.Random(10042)
    expected_shifts = [rng.randint(1, 16) for _ in range(2)]
    assert baseline_sampling.groupby("step")["pair_shift"].first().tolist() == expected_shifts
    for output in (reference_dir, baseline_dir):
        scope = json.loads((output/"result_scope.json").read_text())
        assert scope["status"] == "classifier_fitted"
        assert scope["optimizer_steps"] == 2
        assert scope["supervised_endpoints"] == (1 if output == reference_dir else 2)
        assert scope["sampled_windows"] == 16
        assert scope["supervised_window_endpoints"] == scope["sampled_windows"]*scope["supervised_endpoints"]
        assert scope["total_parameters"] == scope["trainable_parameters"]
        assert scope["reference_parameters"] == (0 if output == reference_dir else sum(p.numel() for p in fusion.reference.parameters()))
        assert not scope["permanent_test_predicted"] and not scope["independent_assessment_performed"]
        assert scope["fit"]["groups"] == scope["selection"]["groups"] == 2
        qualification = json.loads((output/"source_qualification.json").read_text())
        assert qualification["expected_source_conditions"] == ["0", "2"]
        assert len(qualification["source_conditions"]) == 2
        assert all(row["class_support"] == [2, 2] for row in qualification["source_conditions"])
        with np.load(output/"selected_source_validation_windows.npz", allow_pickle=False) as p:
            assert np.isfinite(p["candidate_log_probs"]).all()
            np.testing.assert_allclose(np.exp(p["candidate_log_probs"]), p["candidate_probs"], rtol=1e-6)
            assert all(group.startswith("validation-") for group in p["group_ids"])
            assert set(p["domains"]) == {"0", "2"}
    assert {split for split, _ in accesses} == {"update", "validation"}


def test_paired_composite_supervises_each_endpoint_with_group_weights():
    a = torch.tensor([[2., -.3], [.2, .8], [.7, -.1], [-1., 1.]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[-.1, 1.2], [.8, .3], [.2, 1.], [.9, .2]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0, 0, 1, 0])
    groups, domains = torch.tensor([0, 0, 1, 2]), torch.tensor([0, 0, 0, 1])
    actual = trainer.supervised_loss(a, y, groups, domains, brier_weight=.25, paired_logits=b)
    weights = torch.tensor([.125, .125, .25, .5], dtype=torch.float64)
    onehot = F.one_hot(y, 2)
    def score(logits):
        return F.cross_entropy(logits, y, reduction="none")+.25*(logits.softmax(-1)-onehot).square().sum(-1)
    expected = (.5*(score(a)+score(b))*weights).sum()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    ga, gb = torch.autograd.grad(actual, (a, b))
    assert ga.abs().sum() > 0 and gb.abs().sum() > 0


def test_reference_uses_ce_and_earliest_validation_tie(source_fixture, monkeypatch):
    arguments, _ = source_fixture
    original = trainer.evaluate
    def same_score(*args, **kwargs):
        score, rows, summary = original(*args, **kwargs)
        for row in summary:
            if row["predictor"] == "candidate":
                row["ce"] = 1.
        return score, rows, summary
    monkeypatch.setattr(trainer, "evaluate", same_score)
    args = arguments("reference", "tied", ["--epochs", "2"])
    output = trainer.run(args)
    saved = torch.load(output/"selected_candidate.pt", weights_only=True)
    assert saved["epoch"] == 0
    scope = json.loads((output/"result_scope.json").read_text())
    assert scope["objective"] == "ordinary_group_balanced_CE"
    assert scope["selector"] == "worst_domain_CE"


def test_condition_dg_reference_matches_baseline_paired_sampling(source_fixture, tmp_path, monkeypatch):
    arguments, _ = source_fixture
    data = yaml.safe_load((tmp_path/"data.yaml").read_text())
    data["datasets"][0]["access_scope"] = "source"
    (tmp_path/"data.yaml").write_text(yaml.safe_dump(data))
    seen = []
    original_loss = trainer.supervised_loss
    def capture_loss(logits, labels, groups, domains, **kwargs):
        seen.append((kwargs["brier_weight"], kwargs["paired_logits"] is not None))
        return original_loss(logits, labels, groups, domains, **kwargs)
    monkeypatch.setattr(trainer, "supervised_loss", capture_loss)
    reference = trainer.run(arguments("reference", "paired-reference", ["--dg", "--pair-shift", "4"]))
    baseline = trainer.run(arguments("baseline", "paired-baseline", [
        "--dg", "--pair-shift", "4", "--reference-config", str(reference/"model_config.yaml"),
        "--reference-checkpoint", str(reference/"selected_candidate.pt")]))
    pd.testing.assert_frame_equal(pd.read_csv(reference/"sampling.csv"), pd.read_csv(baseline/"sampling.csv"))
    rng = random.Random(10042)
    assert pd.read_csv(reference/"training_batches.csv").pair_shift.tolist() == [rng.randint(1, 4) for _ in range(2)]
    assert seen == [(0., True), (0., True), (.25, True), (.25, True)]
    scope = json.loads((reference/"result_scope.json").read_text())
    assert scope["objective"] == "paired_group_balanced_CE" and scope["selector"] == "worst_domain_CE"
    assert scope["supervised_endpoints"] == 2 and scope["supervised_window_endpoints"] == 32


def test_condition_dg_reference_cannot_omit_the_paired_shift(source_fixture, tmp_path):
    arguments, accesses = source_fixture
    data = yaml.safe_load((tmp_path/"data.yaml").read_text())
    data["datasets"][0]["access_scope"] = "source"
    (tmp_path/"data.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="same declared paired shift"):
        trainer.run(arguments("reference", "unpaired-dg", ["--dg"]))
    assert not accesses


def test_nonfinite_training_preserves_failure_without_recipe_change(source_fixture, monkeypatch, tmp_path):
    arguments, _ = source_fixture
    monkeypatch.setattr(trainer, "supervised_loss", lambda *args, **kwargs: torch.tensor(float("nan")))
    with pytest.raises(FloatingPointError, match="recipe is unchanged"):
        trainer.run(arguments("reference", "failed"))
    scope = json.loads((tmp_path/"failed"/"result_scope.json").read_text())
    assert scope["status"] == "failed" and scope["error_type"] == "FloatingPointError"
    assert not (tmp_path/"failed"/"selected_candidate.pt").exists()
    assert json.loads((tmp_path/"failed"/"failure.json").read_text())["error_type"] == "FloatingPointError"


def test_invalid_pretraining_request_keeps_failure_and_existing_output(source_fixture, tmp_path):
    arguments, accesses = source_fixture
    with pytest.raises(ValueError, match="positive training budget"):
        trainer.run(arguments("reference", "invalid", ["--lr", "-1"]))
    scope = json.loads((tmp_path/"invalid"/"result_scope.json").read_text())
    assert scope["status"] == "failed"
    assert not accesses
    sentinel = tmp_path/"invalid"/"user.txt"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        trainer.run(arguments("reference", "invalid"))
    assert sentinel.read_text() == "preserve"


def test_formal_dataset_cannot_bypass_source_isolation_by_omitting_dg(source_fixture, tmp_path):
    arguments, accesses = source_fixture
    data = yaml.safe_load((tmp_path/"data.yaml").read_text())
    data["datasets"][0]["protocol"] = "specimen_disjoint_condition_dg"
    (tmp_path/"data.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="source-only metadata"):
        trainer.run(arguments("reference", "unisolated"))
    assert not accesses


def test_mwa_six_level_identity_cannot_silently_use_legacy_four_levels(source_fixture, tmp_path):
    arguments, _ = source_fixture
    (tmp_path/"baseline.yaml").write_text(yaml.safe_dump(dict(model=dict(
        type="X_model", name="MWA_CNN", in_channels=1, num_classes=2, depth=4))))
    with pytest.raises(ValueError, match="explicit depth=6"):
        trainer.run(arguments("baseline", "wrong-mwa", [
            "--dg", "--pair-shift", "4", "--reference-config", str(tmp_path/"reference.yaml"),
            "--reference-checkpoint", str(tmp_path/"not-read.pt")]))


def test_mwa_six_level_baseline_runs_and_restores_through_shared_trainer(source_fixture, tmp_path):
    arguments, _ = source_fixture
    reference_dir = trainer.run(arguments("reference", "mwa-reference"))
    (tmp_path/"baseline.yaml").write_text(yaml.safe_dump(dict(model=dict(
        type="X_model", name="MWA_CNN", in_channels=1, num_classes=2, depth=6))))
    data = yaml.safe_load((tmp_path/"data.yaml").read_text())
    data["datasets"][0]["access_scope"] = "source"
    (tmp_path/"data.yaml").write_text(yaml.safe_dump(data))
    output = trainer.run(arguments("baseline", "mwa-six", [
        "--dg", "--pair-shift", "4", "--reference-config", str(reference_dir/"model_config.yaml"),
        "--reference-checkpoint", str(reference_dir/"selected_candidate.pt")]))
    saved = torch.load(output/"selected_candidate.pt", weights_only=True)
    assert saved["model"]["name"] == "MWA_CNN" and saved["model"]["depth"] == 6
    restored, _ = load_model(output/"selected_candidate.pt", "cpu")
    with torch.no_grad():
        details = restored.forward_details(torch.ones(2, 128, 1))
    assert details["candidate_logits"].shape == (2, 2)
    assert torch.isfinite(details["candidate_logits"]).all()
    assert (output/"source_qualification.json").is_file()


def test_ton_tspn_configuration_fits_independently_with_ce_and_common_selector(source_fixture, tmp_path, monkeypatch):
    arguments, accesses = source_fixture
    reference_dir = trainer.run(arguments("reference", "ton-reference"))
    recipe = dict(model=dict(type="X_model", name="TSPN", in_channels=1, num_classes=2,
        in_dim=128, out_dim=128, out_channels=1, scale=4, skip_connection=True,
        internal_instance_normalization=False,
        signal_processing_configs={f"layer{i}": ["WF"] for i in (1, 2, 3)},
        feature_extractor_configs=["Mean", "Entropy", "Kurtosis"],
        feature_definitions={"Entropy": "absolute_mean_xlogx", "Kurtosis": "population_moment"},feature_epsilon=1e-12,
        gate_parameterization="raw", gate_bias=False, skip_bias=False,
        feature_mixing="per_feature", feature_mixing_bias=False,
        classifier_hidden_dims=[4], classifier_activation="identity", classifier_bias=False,
        f_c_mu=.18, f_c_sigma=.01, f_b_mu=.04, f_b_sigma=.001),
        arm="TSPN_TON", training=dict(objective="ce"),
        provenance=dict(status="documented_adaptation", differences=["Absolute mean xlogx entropy; regularized population kurtosis; width-4 two-affine readout."]))
    (tmp_path/"baseline.yaml").write_text(yaml.safe_dump(recipe))
    data = yaml.safe_load((tmp_path/"data.yaml").read_text())
    data["datasets"][0]["access_scope"] = "source"
    (tmp_path/"data.yaml").write_text(yaml.safe_dump(data))
    objectives, selectors = [], []
    original_loss, original_evaluate = trainer.supervised_loss, trainer.evaluate
    def capture_loss(*args, **kwargs):
        objectives.append(kwargs["brier_weight"])
        return original_loss(*args, **kwargs)
    def capture_selection(*args, **kwargs):
        selectors.append(args[4])
        return original_evaluate(*args, **kwargs)
    monkeypatch.setattr(trainer, "supervised_loss", capture_loss)
    monkeypatch.setattr(trainer, "evaluate", capture_selection)
    output = trainer.run(arguments("baseline", "ton", ["--dg", "--pair-shift", "4",
        "--reference-config", str(reference_dir/"model_config.yaml"),
        "--reference-checkpoint", str(reference_dir/"selected_candidate.pt")]))
    assert objectives == [0., 0.] and selectors == [.25]
    scope = json.loads((output/"result_scope.json").read_text())
    assert scope["role"] == "baseline" and scope["arm"] == "TSPN_TON"
    assert scope["objective"] == "paired_group_balanced_CE"
    assert scope["selector"] == "worst_domain_CE_plus_0.25_Brier_excess"
    assert scope["provenance"] == recipe["provenance"] and scope["reference_state_unchanged"]
    saved = torch.load(output/"selected_candidate.pt", weights_only=True)
    reference = torch.load(reference_dir/"selected_candidate.pt", weights_only=True)
    assert saved["model"]["name"] == "TSPN" and saved["objective"] == "ce"
    assert saved["model"]["signal_processing_configs"] != reference["model"]["signal_processing_configs"]
    assert all(torch.equal(value, saved["reference_state_dict"][key]) for key, value in reference["state_dict"].items())
    restored, _ = load_model(output/"selected_candidate.pt", "cpu")
    time = torch.arange(128, dtype=torch.float32)
    x = torch.stack([torch.sin(time*frequency)+.1*torch.cos(time*.33) for frequency in (.07,.13)]).unsqueeze(-1)
    with torch.no_grad():
        details = restored.forward_details(x)
    assert details["candidate_logits"].shape == details["raw_logits"].shape == (2, 2)
    assert torch.isfinite(details["candidate_logits"]).all()
    with torch.no_grad():
        assert torch.isfinite(restored.forward_details(torch.ones(2, 128, 1))["candidate_logits"]).all()
    with np.load(output/"selected_source_validation_windows.npz", allow_pickle=False) as arrays:
        assert arrays["arm"].item() == "TSPN_TON"
    assert {split for split, _ in accesses} == {"update", "validation"}


def test_ton_configuration_cannot_claim_faithful_implementation(source_fixture, tmp_path):
    arguments, accesses = source_fixture
    model = yaml.safe_load((tmp_path/"reference.yaml").read_text())["model"]
    (tmp_path/"baseline.yaml").write_text(yaml.safe_dump(dict(model=model,arm="TSPN_TON",
        training=dict(objective="ce"),provenance=dict(status="faithful_reproduction"))))
    with pytest.raises(ValueError, match="adaptation provenance"):
        trainer.run(arguments("baseline", "ton-false-claim", ["--dg", "--pair-shift", "4",
            "--reference-config", str(tmp_path/"reference.yaml"),
            "--reference-checkpoint", str(tmp_path/"not-read.pt")]))
    assert not accesses
