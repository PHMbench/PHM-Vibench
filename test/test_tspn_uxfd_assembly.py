from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from src.model_factory.X_model.TSPN_UXFD import Model as TSPNUXFD


def _ns(**kwargs):  # type: ignore[no-untyped-def]
    return SimpleNamespace(**kwargs)


def _make_args(
    *,
    enable_sp2d: bool = False,
    fusion_type: str = "concat",
    enable_fuzzy: bool = False,
    fuzzy_logit_scale: float = 1.0,
    enable_operator_attention: bool = False,
    operator_list: Optional[list[str]] = None,
    enable_logic: bool = False,
    logic_logit_scale: float = 1.0,
    num_classes: object = 3,
) -> SimpleNamespace:
    uxfd = _ns(
        enable_sp2d=enable_sp2d,
        sp2d=_ns(
            n_fft=128,
            hop_length=64,
            win_length=128,
            center=True,
            normalized=False,
            onesided=True,
            magnitude=True,
        ),
        fusion=_ns(type=fusion_type),
        fuzzy=_ns(enable=enable_fuzzy, logit_scale=fuzzy_logit_scale),
        operator_attention=_ns(
            enable=enable_operator_attention,
            operators=operator_list or ["I", "FFT"],
            hidden_dim=32,
            temperature=1.0,
        ),
        logic=_ns(enable=enable_logic, logit_scale=logic_logit_scale, hidden_dim=32),
    )

    return _ns(
        device="cpu",
        num_classes=num_classes,
        in_channels=2,
        out_channels=4,
        scale=1,
        skip_connection=True,
        signal_processing_configs={"layer1": ["I"]},
        feature_extractor_configs=["Mean", "Std"],
        in_dim=128,
        out_dim=128,
        uxfd=uxfd,
    )


def _forward_once(model: TSPNUXFD, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(x)


def test_tspn_uxfd_rejects_dict_num_classes() -> None:
    args = _make_args(num_classes={"source": 3, "target": 4})

    with pytest.raises(ValueError, match="dict-valued multi-dataset heads are out of scope"):
        TSPNUXFD(args)


def test_tspn_uxfd_rejects_complex_sp2d_output() -> None:
    args = _make_args(enable_sp2d=True)
    args.uxfd.sp2d.magnitude = False

    with pytest.raises(ValueError, match="requires uxfd.sp2d.magnitude=true"):
        TSPNUXFD(args)


def test_tspn_uxfd_requires_complete_sp2d_config() -> None:
    args = _make_args(enable_sp2d=True)
    del args.uxfd.sp2d.center

    with pytest.raises(ValueError, match="Missing explicit uxfd.sp2d fields"):
        TSPNUXFD(args)


def test_tspn_uxfd_rejects_unknown_sp2d_fields() -> None:
    args = _make_args(enable_sp2d=True)
    args.uxfd.sp2d.window = "hann"

    with pytest.raises(ValueError, match="Unknown uxfd.sp2d fields"):
        TSPNUXFD(args)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("n_fft", 256, "n_fft must be in"),
        ("hop_length", 129, "hop_length must be in"),
        ("win_length", 129, "win_length must be in"),
    ],
)
def test_tspn_uxfd_rejects_out_of_range_sp2d_values(
    field: str, value: int, match: str
) -> None:
    args = _make_args(enable_sp2d=True)
    setattr(args.uxfd.sp2d, field, value)

    with pytest.raises(ValueError, match=match):
        TSPNUXFD(args)


def test_tspn_uxfd_sp2d_modules_follow_configured_device() -> None:
    args = _make_args(enable_sp2d=True, fusion_type="concat")
    args.device = "meta"

    model = TSPNUXFD(args)

    assert model._uxfd_2d_proj is not None
    assert model._uxfd_2d_proj.weight.device.type == "meta"
    assert model._uxfd_fusion is not None
    fusion_params = list(model._uxfd_fusion.parameters())
    assert fusion_params
    assert all(param.device.type == "meta" for param in fusion_params)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_tspn_uxfd_sp2d_forward_on_cuda() -> None:
    torch.manual_seed(0)
    args = _make_args(enable_sp2d=True, fusion_type="gated")
    args.device = "cuda"
    model = TSPNUXFD(args)
    x = torch.randn(2, 128, 2, device="cuda")

    logits = _forward_once(model, x)

    assert logits.shape == (2, args.num_classes)
    assert logits.device.type == "cuda"
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("fusion_type", ["concat", "sum", "gated"])
def test_tspn_uxfd_sp2d_fusion_forward_shape(fusion_type: str) -> None:
    torch.manual_seed(0)
    args = _make_args(enable_sp2d=True, fusion_type=fusion_type)
    model = TSPNUXFD(args)
    x = torch.randn(2, 128, 2)

    logits = _forward_once(model, x)
    assert logits.shape == (2, args.num_classes)
    assert torch.isfinite(logits).all()


def test_tspn_uxfd_fuzzy_and_logic_residuals_forward_shape() -> None:
    torch.manual_seed(0)
    args = _make_args(enable_fuzzy=True, enable_logic=True, fuzzy_logit_scale=0.5, logic_logit_scale=0.5)
    model = TSPNUXFD(args)
    x = torch.randn(2, 128, 2)

    logits = _forward_once(model, x)
    assert logits.shape == (2, args.num_classes)
    assert torch.isfinite(logits).all()


def test_tspn_uxfd_operator_attention_debug_state() -> None:
    torch.manual_seed(0)
    args = _make_args(enable_operator_attention=True, operator_list=["I", "FFT"])
    model = TSPNUXFD(args)
    x = torch.randn(2, 128, 2)

    _ = _forward_once(model, x)
    state = model.get_uxfd_debug_state()
    assert state["enable_operator_attention"] is True


def test_tspn_uxfd_forward_is_repeatable_given_same_state() -> None:
    torch.manual_seed(0)
    args = _make_args(enable_sp2d=True, fusion_type="gated", enable_operator_attention=True)
    model = TSPNUXFD(args)
    x = torch.randn(2, 128, 2)

    state_before = copy.deepcopy(model.state_dict())
    out1 = _forward_once(model, x)
    model.load_state_dict(state_before)
    out2 = _forward_once(model, x)
    assert torch.allclose(out1, out2, atol=0.0, rtol=0.0)
