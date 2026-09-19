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
        assert not scope["permanent_test_predicted"] and not scope["independent_assessment_performed"]
        assert scope["fit"]["groups"] == scope["selection"]["groups"] == 2
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


def test_nonfinite_training_preserves_failure_without_recipe_change(source_fixture, monkeypatch, tmp_path):
    arguments, _ = source_fixture
    monkeypatch.setattr(trainer, "supervised_loss", lambda *args, **kwargs: torch.tensor(float("nan")))
    with pytest.raises(FloatingPointError, match="recipe is unchanged"):
        trainer.run(arguments("reference", "failed"))
    scope = json.loads((tmp_path/"failed"/"result_scope.json").read_text())
    assert scope["status"] == "failed" and scope["error_type"] == "FloatingPointError"
    assert not (tmp_path/"failed"/"selected_candidate.pt").exists()
