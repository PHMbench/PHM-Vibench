"""
Two-Stage Orchestrator

A thin, reusable orchestrator to run stage 1 (pretraining) and stage 2 (adaptation)
using the PHM‑Vibench factory pattern. Stages expect fully specified sections:
  { data, model, task, trainer, environment }

Usage:
  - Provide a unified config dict with keys 'stage_1' and 'stage_2'.
  - Each stage sub-dict contains the five sections above.
  - Optionally set dry_run=True to skip actual training for unit tests.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import os
from copy import deepcopy

from src.configs.config_utils import (
    load_config,
    transfer_namespace,
    path_name,
    ConfigWrapper,
)
from src.data_factory import build_data
from src.model_factory import build_model
from src.task_factory import build_task
from src.trainer_factory import build_trainer
from src.utils.utils import load_best_model_checkpoint, init_lab, close_lab

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint


class TwoStageOrchestrator:
    def __init__(self, unified_config: Dict[str, Any] | ConfigWrapper, dry_run: bool = False) -> None:
        self.cfg = load_config(unified_config) if not isinstance(unified_config, ConfigWrapper) else unified_config
        self.dry_run = dry_run

        if not hasattr(self.cfg, 'stage_1') or not hasattr(self.cfg, 'stage_2'):
            raise ValueError("Unified config must include 'stage_1' and 'stage_2' sections")

    # ------------------------ helpers ------------------------
    def _stage_to_namespaces(self, stage_cfg: Any):
        # stage_cfg may be ConfigWrapper or plain dict
        if not isinstance(stage_cfg, (dict, ConfigWrapper)):
            raise ValueError("Stage config must be dict/ConfigWrapper")
        # allow dict-like access
        obj = stage_cfg if isinstance(stage_cfg, dict) else stage_cfg.__dict__
        env = transfer_namespace(obj.get('environment', {}))
        data = transfer_namespace(obj.get('data', {}))
        model = transfer_namespace(obj.get('model', {}))
        task = transfer_namespace(obj.get('task', {}))
        trainer = transfer_namespace(obj.get('trainer', {}))
        return env, data, model, task, trainer

    # ------------------------ stage runs ------------------------
    def run_pretrain(self, stage_cfg: Any, iteration: int = 0) -> Dict[str, Any]:
        env, data, model, task, trainer = self._stage_to_namespaces(stage_cfg)

        # seed
        seed = getattr(env, 'seed', 42) + int(iteration)
        seed_everything(seed)

        # path and logging
        path, name = path_name(ConfigWrapper(data=data, model=model, task=task, trainer=trainer))
        trainer.logger_name = name
        init_lab(env, self.cfg, name)

        if self.dry_run:
            close_lab()
            return {'checkpoint_path': None, 'metrics': {'dry_run': True}, 'path': path}

        # build
        data_factory = build_data(data, task)
        net = build_model(model, metadata=data_factory.get_metadata())
        lightning_task = build_task(
            args_task=task,
            network=net,
            args_data=data,
            args_model=model,
            args_trainer=trainer,
            args_environment=env,
            metadata=data_factory.get_metadata(),
        )
        pl_trainer = build_trainer(env, trainer, data, path)

        # train
        pl_trainer.fit(lightning_task, data_factory.get_dataloader('train'), data_factory.get_dataloader('val'))

        # best ckpt
        lightning_task = load_best_model_checkpoint(lightning_task, pl_trainer)
        ckpt_path = None
        for cb in pl_trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                ckpt_path = cb.best_model_path
                break

        # optional test
        test_metrics = {}
        try:
            result = pl_trainer.test(lightning_task, data_factory.get_dataloader('test'))
            if result:
                test_metrics = deepcopy(result[0])
        except Exception:
            pass

        close_lab()
        return {'checkpoint_path': ckpt_path, 'metrics': test_metrics, 'path': path}

    def run_adapt(self, stage_cfg: Any, checkpoint_path: Optional[str] = None, iteration: int = 0) -> Dict[str, Any]:
        env, data, model, task, trainer = self._stage_to_namespaces(stage_cfg)

        # feed ckpt
        if checkpoint_path:
            setattr(model, 'weights_path', checkpoint_path)

        seed = getattr(env, 'seed', 42) + int(iteration)
        seed_everything(seed)

        path, name = path_name(ConfigWrapper(data=data, model=model, task=task, trainer=trainer))
        trainer.logger_name = name
        init_lab(env, self.cfg, name)

        if self.dry_run:
            close_lab()
            return {'metrics': {'dry_run': True}, 'path': path}

        data_factory = build_data(data, task)
        net = build_model(model, metadata=data_factory.get_metadata())
        lightning_task = build_task(
            args_task=task,
            network=net,
            args_data=data,
            args_model=model,
            args_trainer=trainer,
            args_environment=env,
            metadata=data_factory.get_metadata(),
        )
        pl_trainer = build_trainer(env, trainer, data, path)
        pl_trainer.fit(lightning_task, data_factory.get_dataloader('train'), data_factory.get_dataloader('val'))

        # test
        test_metrics = {}
        try:
            result = pl_trainer.test(lightning_task, data_factory.get_dataloader('test'))
            if result:
                test_metrics = deepcopy(result[0])
        except Exception:
            pass

        close_lab()
        return {'metrics': test_metrics, 'path': path}

    def run_complete(self) -> Dict[str, Any]:
        stage1 = getattr(self.cfg, 'stage_1')
        stage2 = getattr(self.cfg, 'stage_2')

        pre = self.run_pretrain(stage1, iteration=0)
        ckpt = pre.get('checkpoint_path')
        ada = self.run_adapt(stage2, checkpoint_path=ckpt, iteration=0)

        return {
            'stage_1': pre,
            'stage_2': ada,
        }


__all__ = ['TwoStageOrchestrator']

