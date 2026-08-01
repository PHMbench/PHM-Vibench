from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest

from src.trainer_factory.p05_runtime import (
    NvidiaGpuIdentity,
    P05RuntimeContract,
    P05RuntimeContractError,
    prepare_p05_runtime,
    query_nvidia_smi_gpu,
)


default_trainer_module = importlib.import_module(
    "src.trainer_factory.Default_trainer"
)
trainer_factory_module = importlib.import_module(
    "src.trainer_factory.trainer_factory"
)


def _evidence_args(**overrides):  # type: ignore[no-untyped-def]
    values = {
        "p05_evidence_mode": True,
        "device": "cuda",
        "expected_gpu_uuid": "GPU-EXPECTED",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _valid_preflight(args, *, visible="0", environment=None):  # type: ignore[no-untyped-def]
    runtime_environment = (
        {"CUDA_VISIBLE_DEVICES": visible}
        if environment is None
        else environment
    )
    return prepare_p05_runtime(
        args,
        environment=runtime_environment,
        cuda_is_available=lambda: True,
        gpu_query=lambda index: NvidiaGpuIdentity(index, "GPU-EXPECTED"),
    )


def test_legacy_mode_returns_without_cuda_or_nvidia_smi_checks() -> None:
    args = SimpleNamespace(
        p05_evidence_mode=False,
        device="cpu",
        gpus=2,
        precision=16,
    )

    contract = prepare_p05_runtime(
        args,
        environment={},
        cuda_is_available=lambda: pytest.fail("CUDA must not be queried"),
        gpu_query=lambda _index: pytest.fail("nvidia-smi must not be queried"),
    )

    assert contract is None
    assert args.device == "cpu"
    assert args.gpus == 2
    assert not hasattr(args, "p05_runtime_identity")


@pytest.mark.parametrize("value", [1, "true", None])
def test_evidence_mode_flag_must_be_a_literal_boolean(value) -> None:  # type: ignore[no-untyped-def]
    args = _evidence_args(p05_evidence_mode=value)

    with pytest.raises(P05RuntimeContractError, match="literal boolean"):
        prepare_p05_runtime(args, environment={})


@pytest.mark.parametrize(
    "visible_devices",
    [None, "", "2", "0,1", "1,0", " 0", "0 "],
)
def test_visible_device_must_be_exactly_one_allowed_physical_index(
    visible_devices,
) -> None:  # type: ignore[no-untyped-def]
    environment = {}
    if visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = visible_devices

    with pytest.raises(P05RuntimeContractError, match="CUDA_VISIBLE_DEVICES"):
        prepare_p05_runtime(
            _evidence_args(),
            environment=environment,
            cuda_is_available=lambda: True,
            gpu_query=lambda _index: pytest.fail("query must not run"),
        )


def test_cuda_unavailable_cannot_fall_back_to_cpu() -> None:
    with pytest.raises(P05RuntimeContractError, match="CPU fallback is forbidden"):
        prepare_p05_runtime(
            _evidence_args(),
            environment={"CUDA_VISIBLE_DEVICES": "0"},
            cuda_is_available=lambda: False,
            gpu_query=lambda _index: pytest.fail("query must not run"),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device", "cpu"),
        ("accelerator", "auto"),
        ("devices", 2),
        ("gpus", 2),
        ("num_nodes", 2),
        ("num_processes", 2),
        ("precision", 16),
        ("deterministic", False),
        ("strategy", "ddp"),
    ],
)
def test_conflicting_trainer_runtime_values_fail_closed(field: str, value) -> None:  # type: ignore[no-untyped-def]
    args = _evidence_args(**{field: value})

    with pytest.raises(P05RuntimeContractError, match=f"trainer.{field}|DDP"):
        _valid_preflight(args)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("WORLD_SIZE", "2"),
        ("LOCAL_WORLD_SIZE", "2"),
        ("SLURM_NTASKS", "4"),
        ("OMPI_COMM_WORLD_SIZE", "2"),
        ("RANK", "1"),
        ("LOCAL_RANK", "1"),
    ],
)
def test_distributed_launcher_environment_is_rejected(name: str, value: str) -> None:
    environment = {"CUDA_VISIBLE_DEVICES": "0", name: value}

    with pytest.raises(P05RuntimeContractError, match=name):
        _valid_preflight(_evidence_args(), environment=environment)


def test_expected_gpu_uuid_is_required_and_must_match() -> None:
    with pytest.raises(P05RuntimeContractError, match="expected_gpu_uuid is required"):
        _valid_preflight(_evidence_args(expected_gpu_uuid=None))

    with pytest.raises(P05RuntimeContractError, match="does not match"):
        prepare_p05_runtime(
            _evidence_args(expected_gpu_uuid="GPU-DIFFERENT"),
            environment={"CUDA_VISIBLE_DEVICES": "0"},
            cuda_is_available=lambda: True,
            gpu_query=lambda index: NvidiaGpuIdentity(index, "GPU-OBSERVED"),
        )


@pytest.mark.parametrize("physical_index", [0, 1])
def test_valid_preflight_fixes_single_gpu_determinism_and_serializes_identity(
    physical_index: int,
) -> None:
    args = _evidence_args()
    queried = []

    contract = prepare_p05_runtime(
        args,
        environment={"CUDA_VISIBLE_DEVICES": str(physical_index)},
        cuda_is_available=lambda: True,
        gpu_query=lambda index: (
            queried.append(index)
            or NvidiaGpuIdentity(index, "GPU-EXPECTED")
        ),
    )

    assert contract is not None
    assert queried == [physical_index]
    assert contract.trainer_kwargs == {
        "accelerator": "gpu",
        "devices": 1,
        "strategy": "auto",
        "precision": 32,
        "deterministic": True,
    }
    assert args.gpus == 1
    assert args.devices == 1
    assert args.precision == 32
    assert args.deterministic is True
    assert contract.runtime_identity["physical_gpu_index"] == physical_index
    assert contract.runtime_identity["gpu_uuid"] == "GPU-EXPECTED"
    assert json.loads(json.dumps(contract.runtime_identity)) == contract.runtime_identity
    assert args.p05_runtime_identity == contract.runtime_identity


def test_nvidia_smi_query_is_mockable_and_uses_physical_index_uuid_output() -> None:
    calls = []

    def fake_runner(command, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="0, GPU-ZERO\n1, GPU-ONE\n")

    identity = query_nvidia_smi_gpu(1, runner=fake_runner)

    assert identity == NvidiaGpuIdentity(physical_index=1, uuid="GPU-ONE")
    assert calls[0][0] == [
        "nvidia-smi",
        "--query-gpu=index,uuid",
        "--format=csv,noheader,nounits",
    ]
    assert calls[0][1]["check"] is True
    assert calls[0][1]["timeout"] == 10


def test_nvidia_smi_query_rejects_missing_requested_physical_index() -> None:
    with pytest.raises(P05RuntimeContractError, match="did not report"):
        query_nvidia_smi_gpu(
            1,
            runner=lambda *_args, **_kwargs: SimpleNamespace(
                stdout="0, GPU-ZERO\n"
            ),
        )


def test_default_trainer_binds_preflight_kwargs_and_runtime_identity(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    contract = P05RuntimeContract(
        trainer_kwargs={
            "accelerator": "gpu",
            "devices": 1,
            "strategy": "auto",
            "precision": 32,
            "deterministic": True,
        },
        runtime_identity={"paper_id": "P05", "gpu_uuid": "GPU-EXPECTED"},
    )
    captured = {}

    class FakeTrainer:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)

    monkeypatch.setattr(
        default_trainer_module,
        "prepare_p05_runtime",
        lambda _args: contract,
    )
    monkeypatch.setattr(default_trainer_module, "call_backs", lambda *_args: [])
    monkeypatch.setattr(default_trainer_module, "CSVLogger", lambda *_args, **_kwargs: "csv")
    monkeypatch.setattr(default_trainer_module.pl, "Trainer", FakeTrainer)

    trainer = default_trainer_module.trainer(
        args_e=SimpleNamespace(wandb=False, swanlab=False),
        args_t=SimpleNamespace(
            p05_evidence_mode=True,
            device="cuda",
            gpus=1,
            num_epochs=1,
            pruning=0.0,
            log_every_n_steps=1,
        ),
        args_d=SimpleNamespace(),
        path="/tmp/p05-runtime-test",
    )

    assert captured["accelerator"] == "gpu"
    assert captured["devices"] == 1
    assert captured["strategy"] == "auto"
    assert captured["precision"] == 32
    assert captured["deterministic"] is True
    assert trainer.p05_runtime_identity == contract.runtime_identity


def test_default_trainer_preserves_legacy_runtime_selection(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured = {}

    class FakeTrainer:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)

    monkeypatch.setattr(
        default_trainer_module,
        "prepare_p05_runtime",
        lambda _args: None,
    )
    monkeypatch.setattr(default_trainer_module, "call_backs", lambda *_args: [])
    monkeypatch.setattr(default_trainer_module, "CSVLogger", lambda *_args, **_kwargs: "csv")
    monkeypatch.setattr(default_trainer_module.pl, "Trainer", FakeTrainer)

    trainer = default_trainer_module.trainer(
        args_e=SimpleNamespace(wandb=False, swanlab=False),
        args_t=SimpleNamespace(
            device="cuda",
            gpus=2,
            num_epochs=1,
            pruning=0.0,
            log_every_n_steps=1,
        ),
        args_d=SimpleNamespace(),
        path="/tmp/p05-runtime-test",
    )

    assert captured["accelerator"] == "auto"
    assert captured["devices"] == 2
    assert captured["strategy"] == "ddp_find_unused_parameters_true"
    assert "precision" not in captured
    assert "deterministic" not in captured
    assert not hasattr(trainer, "p05_runtime_identity")


def test_default_trainer_runs_preflight_before_callbacks(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def reject_preflight(_args):  # type: ignore[no-untyped-def]
        raise P05RuntimeContractError("preflight rejected")

    monkeypatch.setattr(
        default_trainer_module,
        "prepare_p05_runtime",
        reject_preflight,
    )
    monkeypatch.setattr(
        default_trainer_module,
        "call_backs",
        lambda *_args: pytest.fail("callbacks must not be constructed"),
    )

    with pytest.raises(P05RuntimeContractError, match="preflight rejected"):
        default_trainer_module.trainer(
            args_e=SimpleNamespace(wandb=False, swanlab=False),
            args_t=SimpleNamespace(),
            args_d=SimpleNamespace(),
            path="/tmp/p05-runtime-test",
        )


@pytest.mark.parametrize("evidence_mode", [False, True])
def test_trainer_factory_preserves_legacy_none_but_raises_in_evidence_mode(
    monkeypatch,
    evidence_mode: bool,
) -> None:  # type: ignore[no-untyped-def]
    def failing_trainer(**_kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("construction failed")

    monkeypatch.setattr(
        trainer_factory_module.TRAINER_REGISTRY,
        "get",
        lambda _name: failing_trainer,
    )
    args_trainer = SimpleNamespace(
        name="failing",
        p05_evidence_mode=evidence_mode,
    )

    if evidence_mode:
        with pytest.raises(RuntimeError, match="failed to create trainer"):
            trainer_factory_module.trainer_factory(
                SimpleNamespace(),
                args_trainer,
                SimpleNamespace(),
                "/tmp/p05-runtime-test",
            )
    else:
        assert (
            trainer_factory_module.trainer_factory(
                SimpleNamespace(),
                args_trainer,
                SimpleNamespace(),
                "/tmp/p05-runtime-test",
            )
            is None
        )


def test_trainer_factory_import_error_is_not_swallowed_in_evidence_mode(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        trainer_factory_module.TRAINER_REGISTRY,
        "get",
        lambda _name: (_ for _ in ()).throw(KeyError("missing")),
    )
    monkeypatch.setattr(
        trainer_factory_module.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("broken module")),
    )

    with pytest.raises(RuntimeError, match="failed to import trainer"):
        trainer_factory_module.trainer_factory(
            SimpleNamespace(),
            SimpleNamespace(
                name="missing",
                trainer_name="missing",
                p05_evidence_mode=True,
            ),
            SimpleNamespace(),
            "/tmp/p05-runtime-test",
        )
