"""Scientific timing and case-selection contracts on constructed data only."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.p01 import diagnostic_analysis as diagnostics
from experiments.p01.fusion_deployment import FrozenClassifier, log_predictions, verify_vectors
from src.model_factory.X_model.TSPN_fusion import TSPNFusion


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(1)
        self.calls = []
        self.autocast_flags = []

    def forward(self, x):
        self.calls.append((self.training, torch.is_grad_enabled(), self.bn.training))
        self.autocast_flags.append(torch.is_autocast_cpu_enabled())
        return self.bn(x.transpose(1, 2)).mean(-1)


class Features(nn.Module):
    def forward(self, x):
        return torch.cat([x.mean(1), x.square().mean(1)], -1)


class Reference(nn.Module):
    def __init__(self):
        super().__init__()
        self.args = SimpleNamespace(num_classes=2, in_channels=1)
        self.channel_for_classifier = 2
        self.signal_processing_layers = nn.ModuleList([nn.Identity()])
        self.feature_extractor_layers = Features()
        self.clf = nn.Linear(2, 2)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.clf(self.feature_extractor_layers(x))


def fusion():
    model = TSPNFusion(Reference(), in_channels=1, num_classes=2,
                       branches=[], head_type="operator_residual")
    model.set_alpha(1.)
    return model.eval()


def test_profile_executes_full_forward_preserves_modes_and_buffers(monkeypatch):
    model = Probe().train()
    model.bn.eval()  # Mixed module modes must survive, not just the root flag.
    before = {key: value.clone() for key, value in model.named_buffers()}
    times = iter([0., .01, 1., 1.02])
    monkeypatch.setattr(diagnostics.time, "perf_counter", lambda: next(times))
    with torch.enable_grad(), torch.autocast(device_type="cpu", enabled=True):
        report = diagnostics.profile_predictor(model, torch.ones(2, 8, 1), "cpu", warmup=1, repeats=2)
        assert torch.is_grad_enabled()
        assert torch.is_autocast_cpu_enabled()
    assert model.calls == [(False, False, False)] * 3
    assert model.autocast_flags == [False] * 3
    assert model.training and not model.bn.training
    assert all(torch.equal(before[key], value) for key, value in model.named_buffers())
    assert report["full"]["median_ms"] == pytest.approx(15.)
    assert report["full"]["p95_ms"] == pytest.approx(19.5)
    assert report["full"]["peak_allocated_bytes"] is None
    assert report["reference_only"] is None
    assert report["input_shape"] == [2, 8, 1]


def test_classifier_profile_never_adds_auxiliary_reference_cost():
    wrapper = FrozenClassifier.__new__(FrozenClassifier)
    nn.Module.__init__(wrapper)
    wrapper.candidate = Probe()
    wrapper.reference = Probe()
    report = diagnostics.profile_predictor(wrapper, torch.ones(2, 8, 1), "cpu", warmup=1, repeats=2)
    assert len(wrapper.candidate.calls) == 3
    assert wrapper.reference.calls == []
    assert report["reference_only"] is None


def test_fusion_profile_requires_full_deployment_and_separately_times_reference():
    model = fusion()
    with pytest.raises(ValueError, match="alpha=1"):
        model.set_alpha(0.)
        diagnostics.profile_predictor(model, torch.ones(2, 8, 1), "cpu", warmup=0, repeats=1)
    model.set_alpha(1.)
    report = diagnostics.profile_predictor(model, torch.ones(2, 8, 1), "cpu", warmup=1, repeats=2)
    assert report["reference_only"] is not None
    assert model.reference.calls == 3
    assert report["alpha"] == 1.
    assert report["incremental_median_ms"] == report["full"]["median_ms"] - report["reference_only"]["median_ms"]


def test_inference_buffer_mutation_fails_and_restores():
    class BadInference(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("counter", torch.tensor(0))

        def forward(self, x):
            self.counter.add_(1)
            return x.mean(1)

    model = BadInference().train()
    with pytest.raises(RuntimeError, match="changed frozen buffers"):
        diagnostics.profile_predictor(model, torch.ones(2, 8, 1), "cpu", warmup=0, repeats=1)
    assert model.counter.item() == 0 and model.training


def predictions():
    raw = torch.tensor([[3., 0., 0.], [3., 0., 0.], [0., 3., 0.], [0., 3., 0.]]).repeat_interleave(2, 0)
    candidate = torch.tensor([[3., 0., 0.], [0., 3., 3.], [3., 0., 0.], [0., 3., 3.]]).repeat_interleave(2, 0)
    return dict(labels=np.zeros(8, dtype=int), domains=np.repeat("C0", 8),
                group_ids=np.array(["g1", "g0"] * 4), acquisition_ids=np.array([f"a{i}" for i in range(8)]),
                window_ids=np.repeat("0", 8), raw_probs=raw.softmax(-1).numpy(), candidate_probs=candidate.softmax(-1).numpy(),
                raw_log_probs=raw.log_softmax(-1).numpy(), candidate_log_probs=candidate.log_softmax(-1).numpy(),
                logit_contribution__first=((candidate - raw) * .25).numpy(),
                logit_contribution__second=((candidate - raw) * .75).numpy(),
                descriptor__first=np.arange(16).reshape(8, 2))


def select(arrays):
    return diagnostics.select_explanation_cases(arrays, conditions=["C0", "C1"], class_names=["healthy", "inner", "outer"])


def test_frozen_cases_cover_good_bad_and_missing_strata_without_ranking_gain():
    report = select(predictions())
    assert len(report["cases"]) == 4
    assert len(report["strata"]) == 24
    assert {case["outcome"] for case in report["cases"]} == {
        "both_correct", "p0_correct_I_wrong", "p0_wrong_I_correct", "both_wrong"}
    assert sum(row["status"] == "missing" for row in report["strata"]) == 20
    assert all(row["windows"] == 2 and row["specimens"] == 2 for row in report["strata"] if row["status"] == "present")
    assert all(case["contrast_class_a"] == 0 and case["contrast_class_b"] == 1 for case in report["cases"])
    assert max(case["reconstruction_absolute_error"] for case in report["cases"]) < 1e-6
    missing = next(row for row in report["specimen_condition_coverage"]
                   if row["condition_id"] == "C1" and row["fault_class_index"] == 0)
    assert missing["observed_specimens"] == [] and missing["missing_specimens"] == ["g0", "g1"]


def test_cases_and_descriptors_are_invariant_to_export_row_order():
    arrays = predictions()
    order = np.random.default_rng(917).permutation(8)
    assert select(arrays) == select({key: value[order] for key, value in arrays.items()})


def test_duplicate_window_and_undeclared_condition_fail():
    arrays = predictions()
    arrays["domains"][0] = "C9"
    with pytest.raises(ValueError, match="frozen declared"):
        select(arrays)
    arrays = predictions()
    arrays["group_ids"][0] = arrays["group_ids"][1]
    arrays["acquisition_ids"][0] = arrays["acquisition_ids"][1]
    with pytest.raises(ValueError, match="Duplicate"):
        select(arrays)


def test_descriptors_export_uses_same_forward_and_preserves_old_call_contract(monkeypatch):
    model = fusion()
    old_forward = model.forward_details
    calls = []

    def counted(x):
        calls.append(1)
        return old_forward(x)

    monkeypatch.setattr(model, "forward_details", counted)
    x = torch.arange(16, dtype=torch.float32).reshape(2, 8, 1)
    contributions, descriptors = {}, {}
    with torch.no_grad():
        old = log_predictions(model, x, "model", 1., 1.)
        new = log_predictions(model, x, "model", 1., 1., contributions, descriptors)
    assert len(calls) == 2
    assert all(torch.equal(a, b) for a, b in zip(old, new))
    assert list(descriptors) == ["reference"] and descriptors["reference"].shape == (2, 2)
    assert "reference" in contributions


def test_artifact_reorders_descriptors_with_window_identity(tmp_path):
    from experiments.p01.analyze_d1 import load_artifact

    arrays = predictions()
    arrays["deployed_probs"] = arrays["candidate_probs"]
    arrays["deployed_log_probs"] = arrays["candidate_log_probs"]
    arrays["raw_class_names"] = arrays["candidate_class_names"] = np.array(["healthy", "inner", "outer"])
    original = select(arrays)
    np.savez(tmp_path / "predictions.npz", **arrays)
    spec = dict(name="I_seed_42", arm="I", seed=42, split="test", role="direct", alpha=1.,
                path=str(tmp_path / "predictions.npz"))
    artifact = load_artifact(spec)
    assert select(artifact.arrays) == original
    verify_vectors(arrays, arrays)
    no_descriptors = {key: value for key, value in arrays.items() if not key.startswith("descriptor__")}
    verify_vectors(no_descriptors, arrays)  # Pre-descriptor exports still compare.
    with pytest.raises(AssertionError, match="descriptor is missing"):
        verify_vectors(arrays, no_descriptors)
