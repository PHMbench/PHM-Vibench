import torch

from src.Pipeline_06_generative import _expand_condition, _load_sample_payload


def test_sample_payload_roundtrip_preserves_conditions(tmp_path, monkeypatch):
    path = tmp_path / "samples.pt"
    monkeypatch.setenv("PHM_TRUSTED_CHECKPOINT_ROOTS", str(tmp_path))
    payload = {
        "samples": torch.randn(2, 2, 16),
        "fault_label": torch.tensor([0, 1]),
        "domain_id": torch.tensor([1, 2]),
    }
    torch.save(payload, path)

    samples, labels, domains = _load_sample_payload(path)

    assert samples.shape == (2, 2, 16)
    assert labels.tolist() == [0, 1]
    assert domains.tolist() == [1, 2]


def test_expand_condition_repeats_singleton():
    condition = {"fault_label": torch.tensor([1]), "domain_id": torch.tensor([2])}

    out = _expand_condition(condition, num_samples=3, device="cpu")

    assert out["fault_label"].tolist() == [1, 1, 1]
    assert out["domain_id"].tolist() == [2, 2, 2]
