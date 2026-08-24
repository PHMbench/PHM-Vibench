from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from phmfactory.config import analyze_config
from phmfactory.device import resolve_device_request


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_maintained_smoke_uses_one_visible_device_count() -> None:
    trainer = analyze_config("smoke").effective_config["trainer"]

    assert trainer["device"] == "cpu"
    assert trainer["devices"] == 1
    assert "gpus" not in trainer


def test_device_count_is_required_without_hidden_default() -> None:
    with pytest.raises(ValueError, match="trainer.devices is required"):
        resolve_device_request(Namespace(device="cpu"))


def test_dual_device_count_authorities_are_rejected() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_device_request(
            Namespace(device="cpu", devices=1, gpus=1)
        )


def test_device_value_is_not_normalized_from_user_text() -> None:
    with pytest.raises(ValueError, match="unsupported trainer.device"):
        resolve_device_request(Namespace(device="CPU", devices=1))


def test_device_count_is_not_coerced_from_string() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        resolve_device_request(Namespace(device="cpu", devices="1"))


def test_legacy_gpus_alias_is_direct_python_compatibility_only() -> None:
    with pytest.warns(DeprecationWarning, match="trainer.gpus is deprecated"):
        assert resolve_device_request(
            Namespace(device="cpu", gpus=1)
        ) == ("cpu", 1)


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/base/trainer/default_single_gpu.yaml",
        "configs/base/trainer/default_multi_gpu.yaml",
        "configs/base/trainer/fast_debug.yaml",
        "configs/demo/00_smoke/dummy_dg.yaml",
        "configs/demo/00_smoke/dummy_global_average_linear.yaml",
        "configs/demo/10_generative/dummy_generative_cfm.yaml",
        "configs/demo/uxfd/20_smoke_tspn_uxfd_full_cpu.yaml",
        "configs/experiments/p07_xoan_operator_attention/g030_executable_operator_path_smoke.yaml",
        "configs/baselines/01_mfpt/mfpt_global_average_linear.yaml",
    ],
)
def test_maintained_device_configs_do_not_use_legacy_gpus(
    relative_path: str,
) -> None:
    payload = yaml.safe_load(
        (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
    )
    trainer = payload.get("trainer", payload)

    assert "gpus" not in trainer, relative_path
    assert trainer["devices"] >= 1, relative_path
