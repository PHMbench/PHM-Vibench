from pathlib import Path
import copy
from types import SimpleNamespace
import sys
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model_factory.X_model.P01OperatorBias import Model, PhysicalBands, envelope
from src.model_factory.X_model.P01Reference import Model as ReferenceModel
from src.task_factory.Components.p01_bias_losses import (
    covariance_loss, retention_loss, unit_mean, path_margins, replay_der_losses
)


def model(**kwargs):
    return Model(SimpleNamespace(num_classes=3, in_channels=1, **kwargs))


def data(n=257, b=4):
    return torch.randn(b, n, 1), {
        "sample_rate_hz": torch.full((b,), 2048.0),
        "rotation_speed_rpm": torch.tensor([480.0, 600.0, 840.0, 1080.0])[:b],
    }


def test_same_forward_reconstruction_and_gradients():
    torch.manual_seed(7); m = model(); x, meta = data()
    output = m.forward_details(x, physical_metadata=meta)
    assert output["path_names"] == ("bias", "condition", "raw", "periodic", "envelope", "stft")
    torch.testing.assert_close(output["logits"], output["contributions"].sum(1), rtol=0, atol=0)
    torch.testing.assert_close(output["contributions"].mean(-1), torch.zeros(4, 6), atol=1e-6, rtol=0)
    F.cross_entropy(output["logits"], torch.tensor([0, 1, 2, 1])).backward()
    for name, parameter in m.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    for readout in m.readouts.values(): assert readout.weight.grad.abs().sum() > 0
    assert m.context_readout.weight.grad.abs().sum() > 0


def test_metadata_is_separate_from_raw_operator_contribution():
    torch.manual_seed(3); m = model(paths=["raw"]); x = torch.randn(2, 256, 1); x[1] = x[0]
    meta = {"sample_rate_hz": torch.tensor([2048.0, 4096.0]), "rotation_speed_rpm": torch.tensor([480.0, 960.0])}
    out = m.forward_details(x, physical_metadata=meta)
    ri = out["path_names"].index("raw"); ci = out["path_names"].index("condition")
    torch.testing.assert_close(out["contributions"][0, ri], out["contributions"][1, ri])
    assert not torch.allclose(out["contributions"][0, ci], out["contributions"][1, ci])


def test_physical_laws_per_sample():
    m = model(); fs = torch.tensor([2048.0, 2048.0]); rho = torch.tensor([8.0, 16.0])
    fc, bw = m.bands["periodic"].physical(fs, rho)
    torch.testing.assert_close(fc[1], 2 * fc[0]); torch.testing.assert_close(bw[1], 2 * bw[0])
    fc, bw = m.carrier.physical(fs, rho)
    torch.testing.assert_close(fc[0], fc[1]); torch.testing.assert_close(bw[0], bw[1])


def test_fixed_binding_uses_explicit_source_reference():
    m = model(binding="fixed", reference_speed_hz=12.0, fixed_reference_speed_hz=11.0)
    torch.testing.assert_close(m._periodic_speed(torch.tensor([8.0, 16.0])), torch.tensor([11.0, 11.0]))


def test_wrong_resonance_is_incorrect_but_can_remain_observable():
    m = model(binding="wrong_resonance", reference_speed_hz=10.0, fixed_reference_speed_hz=10.0, wrong_resonance_slope_hz_per_hz=5.0)
    result = m._wrong_carrier(torch.randn(2, 512, 1), torch.tensor([2048.0, 2048.0]), torch.tensor([8.0, 18.0]))
    assert result.shape[0] == 2 and torch.isfinite(result).all()


def test_no_metadata_broadcast_or_guessed_speed():
    m = model(); x, meta = data()
    with pytest.raises(ValueError): m(x)
    meta["rotation_speed_rpm"] = torch.tensor([600.0])
    with pytest.raises(ValueError): m(x, physical_metadata=meta)


def test_metadata_rows_preserve_file_identity():
    rows = {11: {"sample_rate_hz": 2048.0, "rotation_speed_rpm": 480.0}, 27: {"sample_rate_hz": 2048.0, "rotation_speed_rpm": 960.0}}
    m = Model(SimpleNamespace(num_classes=3, in_channels=1), rows); x = torch.randn(2, 256, 1)
    fs, rho = m._physical_metadata(x, torch.tensor([27, 11]), None)
    torch.testing.assert_close(rho, torch.tensor([16.0, 8.0]))


@pytest.mark.parametrize("n", [255, 256, 257])
def test_odd_even_signal_shape(n):
    x, meta = data(n=n)
    assert envelope(x).shape == x.shape
    assert model()(x, physical_metadata=meta).shape == (4, 3)


def test_band_support_fails_without_clipping():
    m = model(); x, meta = data(); meta["sample_rate_hz"] = torch.full((4,), 100.0)
    with pytest.raises(ValueError, match="observable support"): m(x, physical_metadata=meta)


def test_stft_resolution_admission_is_explicit():
    fs = torch.tensor([2048.0]); rpm = torch.tensor([480.0]); speed = rpm / 60.0
    under = model(paths=["stft"], n_fft=128, min_stft_bins_per_band=2.0)
    assert under.stft_resolution(fs, speed)["bins_per_band"].min() < 2
    with pytest.raises(ValueError, match="under-resolved"):
        under(torch.randn(1, 1024, 1), physical_metadata={"sample_rate_hz": fs, "rotation_speed_rpm": rpm})
    resolved = model(paths=["stft"], n_fft=1024, hop_length=256, min_stft_bins_per_band=2.0)
    assert resolved.stft_resolution(fs, speed)["bins_per_band"].min() >= 2
    assert torch.isfinite(resolved(torch.randn(1, 2048, 1), physical_metadata={"sample_rate_hz": fs, "rotation_speed_rpm": rpm})).all()


def test_stft_does_not_erase_frequency_distribution():
    m = model(paths=["stft"]); t = torch.arange(1024) / 2048
    x = torch.stack([torch.sin(2 * torch.pi * 30 * t), torch.sin(2 * torch.pi * 140 * t)])[:, :, None]
    meta = {"sample_rate_hz": torch.full((2,), 2048.0), "rotation_speed_rpm": torch.full((2,), 600.0)}
    z = m.forward_details(x, physical_metadata=meta)["features"]["stft"]
    assert not torch.allclose(z[0], z[1], atol=1e-3)


def test_metadata_only_reference_uses_only_condition_context():
    m = ReferenceModel(SimpleNamespace(num_classes=3, in_channels=1, reference_kind="metadata_only", reference_speed_hz=10.0))
    x = torch.randn(2, 256, 1); x[1] = x[0]
    meta = {"sample_rate_hz": torch.tensor([2048.0, 4096.0]), "rotation_speed_rpm": torch.tensor([480.0, 960.0])}
    out = m(x, physical_metadata=meta)
    assert out.shape == (2, 3) and not torch.allclose(out[0], out[1])


def test_state_unchanged_during_eval_and_checkpoint_roundtrip(tmp_path):
    m = model().eval(); x, meta = data(); state = copy.deepcopy(m.state_dict()); expected = m(x, physical_metadata=meta)
    for key, value in state.items(): torch.testing.assert_close(value, m.state_dict()[key], rtol=0, atol=0)
    torch.save(m.state_dict(), tmp_path / "m.pt"); n = model().eval(); n.load_state_dict(torch.load(tmp_path / "m.pt", weights_only=True))
    torch.testing.assert_close(expected, n(x, physical_metadata=meta))


def test_teacher_stops_gradients_bias_is_constrained_context_is_not():
    current = torch.randn(3, 6, 3, requires_grad=True); teacher = torch.randn(3, 6, 3, requires_grad=True)
    y = torch.tensor([0, 1, 2]); ids = torch.arange(3)
    retention_loss(current, teacher, y, ids).backward()
    assert teacher.grad is None and current.grad is not None
    assert current.grad[:, 0].abs().sum() > 0 and current.grad[:, 1].abs().sum() == 0
    assert current.grad[:, 2:].abs().sum() > 0


def test_one_sided_allows_improvement_not_deterioration():
    old = torch.zeros(2, 4, 3); new = old.clone(); y = torch.tensor([0, 1]); ids = torch.arange(2)
    new[0, 2:, 0] = 1; new[1, 2:, 1] = 1
    assert retention_loss(new, old, y, ids).item() == 0
    assert retention_loss(-new, old, y, ids).item() > 0
    assert retention_loss(new, old, y, ids, "path_symmetric").item() > 0


def test_path_cancellation_is_visible_but_total_margin_is_not():
    old = torch.zeros(1, 5, 3); new = old.clone(); new[0, 2, 0] = 1; new[0, 3, 0] = -1
    y = torch.tensor([0]); ids = torch.tensor([0])
    assert retention_loss(new, old, y, ids, "total_one_sided").item() == 0
    assert retention_loss(new, old, y, ids).item() > 0


def test_unit_weights_not_window_weights():
    assert unit_mean(torch.tensor([1.0, 1.0, 1.0, 5.0]), torch.tensor([0, 0, 0, 1])).item() == 3


def test_covariance_uses_only_signal_paths():
    a = torch.randn(2, 5, 3, requires_grad=True); b = torch.randn(2, 5, 3, requires_grad=True); ids = torch.arange(2)
    assert covariance_loss(a, a, ids).item() == 0
    covariance_loss(a, b, ids).backward()
    assert a.grad[:, 2:].abs().sum() > 0 and b.grad[:, 2:].abs().sum() > 0
    assert a.grad[:, :2].abs().sum() == 0 and b.grad[:, :2].abs().sum() == 0


def test_retention_ce_upper_bound_for_bias_and_signal_paths():
    torch.manual_seed(10); old = torch.randn(8, 6, 4); new = torch.randn(8, 6, 4); old[:, 1] = 0; new[:, 1] = 0; y = torch.arange(8) % 4
    old_paths = torch.cat((old[:, :1], old[:, 2:]), dim=1); new_paths = torch.cat((new[:, :1], new[:, 2:]), dim=1)
    decline = (path_margins(old_paths, y) - path_margins(new_paths, y)).relu()
    bound = (old_paths.shape[1] * decline.square().sum((1, 2))).sqrt()
    increase = F.cross_entropy(new.sum(1), y, reduction="none") - F.cross_entropy(old.sum(1), y, reduction="none")
    assert (increase <= bound + 1e-6).all()


def test_der_uses_given_frozen_logits():
    x = torch.randn(2, 3, requires_grad=True); old = torch.randn(2, 3, requires_grad=True); y = torch.tensor([0, 1]); ids = torch.arange(2)
    ce, mse = replay_der_losses(x, old, y, ids); (ce + mse).backward(); assert old.grad is None
