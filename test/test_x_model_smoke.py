from __future__ import annotations

import csv
import importlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import pytest
import torch


@dataclass(frozen=True)
class XModelRegistryRow:
    model_type: str
    model_name: str
    module_path: str


def _iter_x_model_registry_rows() -> List[XModelRegistryRow]:
    registry_path = Path("src/model_factory/model_registry.csv")
    rows: List[XModelRegistryRow] = []
    with registry_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("model.type") or "").strip() != "X_model":
                continue
            module_path = (row.get("module_path") or "").strip().replace("/", ".")
            if module_path.endswith(".py"):
                module_path = module_path[:-3]
            rows.append(
                XModelRegistryRow(
                    model_type="X_model",
                    model_name=(row.get("model.name") or "").strip(),
                    module_path=module_path,
                )
            )
    return rows


def _base_args(model_name: str) -> Dict[str, object]:
    return {
        "type": "X_model",
        "name": model_name,
        "device": "cpu",
        "in_dim": 128,
        "out_dim": 128,
        "in_channels": 2,
        "out_channels": 4,
        "num_classes": 4,
        "output_dim": 4,
        "scale": 1,
        "skip_connection": True,
        "signal_processing_configs": {"layer1": ["I"]},
        "feature_extractor_configs": ["Mean", "Std"],
        "uxfd": {
            "enable_sp2d": False,
            "fuzzy": {"enable": False},
            "operator_attention": {"enable": False},
            "logic": {"enable": False},
        },
        "decision_configs": {"type": "linear"},
        "width": 64,
        "dropout": 0.1,
        "hidden_dim": 64,
        "seq_len": 128,
        "input_dim": 256,
    }


def _build_case(model_name: str) -> Tuple[SimpleNamespace, torch.Tensor]:
    args = _base_args(model_name)
    input_shape = (2, 128, 2)

    if model_name == "BASE_ExplainableCNN":
        args.update(in_channels=2, width=64, dropout=0.1, num_classes=4)
    elif model_name == "MWA_CNN":
        args.update(in_channels=2, num_classes=4)
    elif model_name == "CI_GNN":
        args.update(in_channels=4, num_sensors=4, hidden_dim=32, num_layers=3, num_classes=4)
        input_shape = (2, 128, 4)
    elif model_name == "GradCAM_XFD":
        args.update(in_channels=2, input_channels=2, seq_length=128, num_classes=4)
    elif model_name == "Physics_informed_PDN":
        args.update(input_dim=256, hidden_dim=64, num_samples=4, num_classes=4)
    elif model_name == "Resnet":
        args.update(in_channels=2, in_channel=2, num_classes=4, num_class=4, first_kernel="conv")
    elif model_name == "Sincnet":
        args.update(in_channels=2, in_channel=2, num_classes=4, num_class=4, variant="sinc_m")
    elif model_name == "WKN":
        args.update(in_channels=2, in_channel=2, num_classes=4, num_class=4, variant="wkn_m")
    elif model_name == "EELM":
        args.update(in_channels=2, num_classes=4, num_class=4)
    elif model_name == "MCN":
        args.update(in_channels=2, in_dim=128, seq_len=128, num_mfks=4, num_classes=4, mode="gfk")
    elif model_name == "TFN":
        args.update(in_channels=2, variant="morlet", mid_channel=8, num_classes=4)
    elif model_name == "F_EQL":
        args.update(in_channels=2, hidden_dim=32, num_classes=4)
    elif model_name in {"TSPN", "TSPN_UXFD", "NSN"}:
        pass

    ns = SimpleNamespace(**args)
    x = torch.randn(*input_shape)
    return ns, x


@pytest.mark.parametrize(
    "row",
    _iter_x_model_registry_rows(),
    ids=lambda row: f"{row.model_type}.{row.model_name}",
)
def test_x_model_import_init_forward_smoke(row: XModelRegistryRow) -> None:
    torch.manual_seed(0)

    try:
        module = importlib.import_module(row.module_path)
    except ModuleNotFoundError as exc:
        if row.model_name == "CI_GNN" and "torch_geometric" in str(exc):
            pytest.skip(f"dependency-blocked: {row.model_name} requires torch_geometric")
        raise
    assert hasattr(module, "Model"), f"{row.module_path} missing Model class"

    args, x = _build_case(row.model_name)
    model = module.Model(args, metadata={})
    model.eval()

    with torch.no_grad():
        out = model(x)

    assert isinstance(out, torch.Tensor), f"{row.model_name} must return torch.Tensor"
    assert out.ndim == 2, f"{row.model_name} output should be 2D logits, got {tuple(out.shape)}"
    assert out.shape[0] == x.shape[0], f"{row.model_name} batch mismatch: {tuple(out.shape)} vs {tuple(x.shape)}"
    assert out.shape[1] == int(args.num_classes), (
        f"{row.model_name} class dim mismatch: expected {args.num_classes}, got {out.shape[1]}"
    )


def test_mwa_huan_net_compatibility_alias() -> None:
    from src.model_factory.X_model.MWA_CNN import Huan_net

    model = Huan_net(input_size=2, num_class=4)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 128, 2))
    assert out.shape == (2, 4)


def test_mcn_default_mode_is_gfk() -> None:
    from src.model_factory.X_model.MCN import Model

    args = SimpleNamespace(type="X_model", name="MCN", in_channels=2, in_dim=128, num_classes=4)
    model = Model(args, metadata={})
    assert getattr(model, "use_wfk", True) is False
    with torch.no_grad():
        out = model(torch.randn(2, 128, 2))
    assert out.shape == (2, 4)


def test_tfn_default_variant_is_morlet() -> None:
    from src.model_factory.X_model.TFN import Model

    args = SimpleNamespace(type="X_model", name="TFN", in_channels=2, num_classes=4)
    model = Model(args, metadata={})
    assert type(model.network).__name__ == "TFN_Morlet"
    with torch.no_grad():
        out = model(torch.randn(2, 128, 2))
    assert out.shape == (2, 4)


def test_feql_compatibility_forward() -> None:
    from src.model_factory.X_model.F_EQL import Model

    args = SimpleNamespace(type="X_model", name="F_EQL", in_channels=2, hidden_dim=32, num_classes=4)
    model = Model(args, metadata={})
    with torch.no_grad():
        out = model(torch.randn(2, 128, 2))
    assert out.shape == (2, 4)
