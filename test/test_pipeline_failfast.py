from __future__ import annotations

import argparse

import pytest

import src.Pipeline_03_multitask_pretrain_finetune as p03
import src.Pipeline_04_unified_metric as p04


def test_pipeline_03_unified_adapter_failure_is_not_swallowed(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("bad p03 adapter")

    monkeypatch.setattr(p03, "adapt_p03", boom)

    with pytest.raises(RuntimeError, match="bad p03 adapter"):
        p03.pipeline(argparse.Namespace(config_path="dummy.yaml", local_config=None))


def test_pipeline_04_unified_adapter_failure_is_not_swallowed(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("bad p04 adapter")

    monkeypatch.setattr(p04, "adapt_p04", boom)

    with pytest.raises(RuntimeError, match="bad p04 adapter"):
        p04.pipeline(argparse.Namespace(config_path="dummy.yaml", local_config=None))
