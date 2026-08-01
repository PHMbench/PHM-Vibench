import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.Pipeline_05_Explainable_Fault_Diagnosis as pipeline_module
from src.data_factory.data_factory import data_factory
from src.data_factory.data_utils import MetadataAccessor
from src.data_factory.splitting import SplitResult


data_factory_module = importlib.import_module("src.data_factory.data_factory")


def _metadata():
    return MetadataAccessor(
        pd.DataFrame(
            [
                {"Id": 1, "Label": 0, "Dataset_id": 2},
                {"Id": 2, "Label": 1, "Dataset_id": 2},
                {"Id": 3, "Label": 1, "Dataset_id": 2},
            ]
        ),
        key_column="Id",
    )


class _OneWindowDataset:
    accessed_ids = []

    def __init__(self, data, metadata, args_data, args_task, mode):
        del metadata, args_data, args_task, mode
        self.accessed_ids.extend(data)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        del index
        return {"x": np.zeros((2, 4)), "y": 0}


class _NoTestData(dict):
    def __getitem__(self, key):
        if key == 3:
            raise AssertionError("test signal was accessed during fit_validate_only")
        return super().__getitem__(key)


def test_fit_validate_only_constructs_no_test_dataset(monkeypatch):
    factory = data_factory.__new__(data_factory)
    factory.args_data = SimpleNamespace()
    factory.args_task = SimpleNamespace(name="classification", type="Default_task")
    factory.execution_stage = "fit_validate_only"
    factory.target_metadata = _metadata()
    factory.split_result = SplitResult(
        train_ids=(1,),
        val_ids=(2,),
        test_ids=(3,),
        strategy="preassigned_metadata",
    )
    factory.data = _NoTestData(
        {
            1: np.zeros((8, 2, 1)),
            2: np.zeros((8, 2, 1)),
        }
    )
    _OneWindowDataset.accessed_ids = []
    monkeypatch.setattr(
        data_factory_module.importlib,
        "import_module",
        lambda _: SimpleNamespace(set_dataset=_OneWindowDataset),
    )

    train, val, test = factory._init_dataset()

    assert tuple(train.dataset_dict) == (1,)
    assert tuple(val.dataset_dict) == (2,)
    assert test is None
    assert _OneWindowDataset.accessed_ids == [1, 2]


def test_fit_validate_only_limits_cache_preparation_to_train_and_val():
    factory = data_factory.__new__(data_factory)
    factory.active_metadata = data_factory._metadata_for_ids(_metadata(), [1, 2])
    observed = []
    factory._determine_missing_ids = (
        lambda task_meta, args_data, use_cache: observed.append(tuple(task_meta.keys())) or {}
    )
    factory._build_final_cache = (
        lambda task_meta, args_data, use_cache: observed.append(tuple(task_meta.keys()))
        or "/tmp/nonexistent-p05-stage-cache.h5"
    )

    lazy_data = factory._init_data(SimpleNamespace())

    assert observed == [(1, 2), (1, 2)]
    assert lazy_data.h5_file == "/tmp/nonexistent-p05-stage-cache.h5"


def test_test_loader_access_fails_closed_when_stage_omits_test():
    factory = data_factory.__new__(data_factory)
    factory.execution_stage = "fit_validate_only"
    factory.train_loader = object()
    factory.val_loader = object()
    factory.test_loader = None

    with pytest.raises(RuntimeError, match="fit_validate_only"):
        factory.get_dataloader("test")


def test_pipeline_fit_validate_only_never_requests_test(monkeypatch, tmp_path):
    configs = SimpleNamespace(
        data=SimpleNamespace(),
        model=SimpleNamespace(),
        task=SimpleNamespace(name="classification", type="Default_task"),
        trainer=SimpleNamespace(),
        environment=SimpleNamespace(
            iterations=1,
            seed=11,
            stage="fit_validate_only",
            output_dir=str(tmp_path),
        ),
    )
    loader_requests = []

    class FakeDataFactory:
        def __init__(self):
            self.data = SimpleNamespace(close=lambda: None)

        def get_metadata(self):
            return {}

        def get_dataloader(self, mode):
            loader_requests.append(mode)
            if mode == "test":
                raise AssertionError("pipeline requested the test loader")
            return []

    class FakeTrainer:
        def fit(self, task, train_loader, val_loader):
            del task, train_loader, val_loader

        def test(self, task, test_loader):
            del task, test_loader
            raise AssertionError("trainer.test was called")

    fake_data = FakeDataFactory()
    monkeypatch.setattr(pipeline_module, "load_config", lambda *_: configs)
    monkeypatch.setattr(pipeline_module, "merge_with_local_override", lambda *_: configs)
    monkeypatch.setattr(pipeline_module, "transfer_namespace", lambda value: value)
    monkeypatch.setattr(
        pipeline_module,
        "path_name",
        lambda *_: (str(tmp_path / "run"), "p05-stage-test"),
    )
    monkeypatch.setattr(pipeline_module, "save_config", lambda *_: None)
    monkeypatch.setattr(pipeline_module, "init_lab", lambda *_: None)
    monkeypatch.setattr(pipeline_module, "close_lab", lambda: None)
    monkeypatch.setattr(pipeline_module, "build_data", lambda *_: fake_data)
    monkeypatch.setattr(pipeline_module, "build_model", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(pipeline_module, "build_task", lambda **_kwargs: object())
    monkeypatch.setattr(pipeline_module, "build_trainer", lambda *_: FakeTrainer())
    monkeypatch.setattr(pipeline_module, "write_metadata_snapshot", lambda *_: None)

    result = pipeline_module.pipeline(
        SimpleNamespace(config_path="unused.yaml", local_config=None, override=[])
    )

    assert result == [
        {"stage": "fit_validate_only", "status": "fit_validate_complete", "seed": 11}
    ]
    assert "test" not in loader_requests
    assert loader_requests == ["val", "train", "val"]


@pytest.mark.parametrize("stage", ["fit_validate_only", "fit_validate_test"])
def test_pipeline_accepts_only_registered_execution_stages(stage):
    assert (
        pipeline_module._resolve_execution_stage(SimpleNamespace(stage=stage))
        == stage
    )


def test_pipeline_rejects_unknown_execution_stage():
    with pytest.raises(ValueError, match="environment.stage"):
        pipeline_module._resolve_execution_stage(SimpleNamespace(stage="train"))


def test_p05_process_contract_requires_one_seed_per_process():
    accepted = SimpleNamespace(iterations=1, seed=42)
    pipeline_module._validate_p05_process_contract(accepted, object())

    with pytest.raises(ValueError, match="iterations=1"):
        pipeline_module._validate_p05_process_contract(
            SimpleNamespace(iterations=5, seed=42), object()
        )
    with pytest.raises(ValueError, match="integer seed"):
        pipeline_module._validate_p05_process_contract(
            SimpleNamespace(iterations=1, seed=True), object()
        )


def test_p05_evaluation_contract_binds_trace_only_to_method_arm():
    runtime = object()
    assert pipeline_module._validate_p05_evaluation_contract(
        SimpleNamespace(
            p05_evidence_mode=True,
            p05_arm_id="P05-M",
            p05_trace_export=True,
        ),
        runtime,
    ) is True
    assert pipeline_module._validate_p05_evaluation_contract(
        SimpleNamespace(
            p05_evidence_mode=True,
            p05_arm_id="P05-B0",
            p05_trace_export=False,
        ),
        runtime,
    ) is False

    with pytest.raises(ValueError, match="P05-M.*trace"):
        pipeline_module._validate_p05_evaluation_contract(
            SimpleNamespace(
                p05_evidence_mode=True,
                p05_arm_id="P05-M",
                p05_trace_export=False,
            ),
            runtime,
        )
    with pytest.raises(ValueError, match="only for P05-M"):
        pipeline_module._validate_p05_evaluation_contract(
            SimpleNamespace(
                p05_evidence_mode=True,
                p05_arm_id="P05-B0",
                p05_trace_export=True,
            ),
            runtime,
        )
