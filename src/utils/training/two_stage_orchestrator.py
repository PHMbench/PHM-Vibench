"""
Multi-Stage / Two-Stage Orchestrator

一个可复用的多阶段训练调度器，基于 PHM‑Vibench 的工厂模式。

- 输入：统一配置（dict 或 ConfigWrapper），包含一组阶段：
    stages = [ {data, model, task, trainer, environment}, ... ]
- 兼容旧格式：{'stage_1': {...}, 'stage_2': {...}}，并通过 TwoStageOrchestrator 提供同名接口。
- 每个阶段都可以单独保存 best checkpoint，方便后续从任意阶段继续训练。
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from types import SimpleNamespace

from copy import deepcopy

from src.configs.config_utils import (
    load_config,
    transfer_namespace,
    path_name,
    ConfigWrapper,
    dict_to_namespace,
    _validate_config_wrapper,
)
from src.data_factory import build_data
from src.model_factory import build_model
from src.task_factory import build_task
from src.trainer_factory import build_trainer
from src.utils.utils import load_best_model_checkpoint, init_lab, close_lab

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint


class MultiStageOrchestrator:
    """通用多阶段 Orchestrator，TwoStageOrchestrator 作为其向后兼容子类。"""

    def __init__(self, unified_config: Any, dry_run: bool = False) -> None:
        self.dry_run = dry_run

        # 1) 已经是 ConfigWrapper：直接使用
        if isinstance(unified_config, ConfigWrapper):
            self.cfg = unified_config

        # 2) dict：支持 {'stages': [...]} 或 {'stage_1': ..., 'stage_2': ...}
        elif isinstance(unified_config, dict):
            if 'stages' in unified_config:
                stages_dict_list = unified_config['stages']
            else:
                stages_dict_list = [
                    unified_config.get('stage_1', {}),
                    unified_config.get('stage_2', {}),
                ]
            stages_ns = [dict_to_namespace(s or {}) for s in stages_dict_list if s]
            attrs: Dict[str, Any] = {'stages': stages_ns}
            # 保留 stage_1/stage_2 映射，兼容旧的 CLI override 和代码访问
            if len(stages_ns) >= 1:
                attrs['stage_1'] = stages_ns[0]
            if len(stages_ns) >= 2:
                attrs['stage_2'] = stages_ns[1]
            self.cfg = ConfigWrapper(**attrs)

        # 3) 其他情况：视为单阶段配置源（路径/预设），做最小兼容
        else:
            base_cfg = load_config(unified_config)
            stage_ns = SimpleNamespace(
                data=getattr(base_cfg, 'data', SimpleNamespace()),
                model=getattr(base_cfg, 'model', SimpleNamespace()),
                task=getattr(base_cfg, 'task', SimpleNamespace()),
                trainer=getattr(base_cfg, 'trainer', SimpleNamespace()),
                environment=getattr(base_cfg, 'environment', SimpleNamespace()),
            )
            self.cfg = ConfigWrapper(stages=[stage_ns], stage_1=stage_ns)

        # 验证每个阶段的 data/model/task 必需字段
        self._validate_stages()

    # ------------------------ validation ------------------------
    def _validate_stages(self) -> None:
        stages = getattr(self.cfg, 'stages', []) or []
        for stage in stages:
            stage_wrapped = ConfigWrapper(
                data=getattr(stage, 'data', SimpleNamespace()),
                model=getattr(stage, 'model', SimpleNamespace()),
                task=getattr(stage, 'task', SimpleNamespace()),
                trainer=getattr(stage, 'trainer', SimpleNamespace()),
            )
            _validate_config_wrapper(stage_wrapped)

    # ------------------------ helpers ------------------------
    def _stage_to_namespaces(self, stage_cfg: Any):
        # stage_cfg may be dict / ConfigWrapper / SimpleNamespace
        if isinstance(stage_cfg, dict):
            obj = stage_cfg
        elif isinstance(stage_cfg, (ConfigWrapper, SimpleNamespace)):
            obj = stage_cfg.__dict__
        else:
            raise ValueError("Stage config must be dict / ConfigWrapper / SimpleNamespace")

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
        test_metrics: Dict[str, Any] = {}
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

        # feed ckpt from previous stage if provided
        if checkpoint_path:
            setattr(model, 'weights_path', checkpoint_path)

        seed = getattr(env, 'seed', 42) + int(iteration)
        seed_everything(seed)

        path, name = path_name(ConfigWrapper(data=data, model=model, task=task, trainer=trainer))
        trainer.logger_name = name
        init_lab(env, self.cfg, name)

        if self.dry_run:
            close_lab()
            return {'checkpoint_path': checkpoint_path, 'metrics': {'dry_run': True}, 'path': path}

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

        # best ckpt for this stage
        lightning_task = load_best_model_checkpoint(lightning_task, pl_trainer)
        ckpt_path = None
        for cb in pl_trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                ckpt_path = cb.best_model_path
                break

        # test
        test_metrics: Dict[str, Any] = {}
        try:
            result = pl_trainer.test(lightning_task, data_factory.get_dataloader('test'))
            if result:
                test_metrics = deepcopy(result[0])
        except Exception:
            pass

        close_lab()
        return {'checkpoint_path': ckpt_path, 'metrics': test_metrics, 'path': path}

    # ------------------------ orchestrator APIs ------------------------
    def run_all_stages(self) -> Dict[str, Any]:
        """按顺序运行所有阶段，并返回每个阶段的结果与 checkpoint 路径。"""
        stages = getattr(self.cfg, 'stages', []) or []
        if not stages:
            raise ValueError("MultiStageOrchestrator 需要至少一个 stage 配置")

        ckpt_path: Optional[str] = None
        results: Dict[str, Any] = {}
        ckpt_registry: Dict[str, Optional[str]] = {}

        for idx, stage_cfg in enumerate(stages):
            env = getattr(stage_cfg, 'environment', None)
            stage_name = getattr(env, 'stage_name', f"stage_{idx+1}") if env is not None else f"stage_{idx+1}"

            if idx == 0:
                out = self.run_pretrain(stage_cfg, iteration=idx)
            else:
                out = self.run_adapt(stage_cfg, checkpoint_path=ckpt_path, iteration=idx)

            # 更新当前累计 checkpoint（如果本阶段训练出了新的 best ckpt）
            stage_ckpt = out.get('checkpoint_path', ckpt_path)
            ckpt_path = stage_ckpt or ckpt_path
            ckpt_registry[stage_name] = ckpt_path

            # 确保返回结果中带有 checkpoint_path 字段
            out['checkpoint_path'] = stage_ckpt
            results[stage_name] = out

        results['_ckpt_registry'] = ckpt_registry
        return results

    def run_complete(self) -> Dict[str, Any]:
        """兼容旧接口的两阶段视图：返回 {'stage_1': ..., 'stage_2': ...}。"""
        # 如果有显式的 stage_1/2，优先使用；否则退化为 run_all_stages 的前两阶段
        stage1 = getattr(self.cfg, 'stage_1', None)
        stage2 = getattr(self.cfg, 'stage_2', None)

        if stage1 is None:
            stages = getattr(self.cfg, 'stages', []) or []
            if not stages:
                raise ValueError("Two-stage view requires at least one stage")
            stage1 = stages[0]
            if len(stages) > 1:
                stage2 = stages[1]

        pre = self.run_pretrain(stage1, iteration=0)
        ckpt = pre.get('checkpoint_path')

        if stage2 is None:
            return {'stage_1': pre}

        ada = self.run_adapt(stage2, checkpoint_path=ckpt, iteration=0)
        return {'stage_1': pre, 'stage_2': ada}


class TwoStageOrchestrator(MultiStageOrchestrator):
    """
    兼容旧接口的两阶段 Orchestrator，实际逻辑全部继承自 MultiStageOrchestrator。

    - 接受 {'stage_1': ..., 'stage_2': ...} 或包含 stages 列表的配置；
    - pipeline 仍然可以调用 `run_complete()` 获取两阶段结果。
    """

    pass


__all__ = ['TwoStageOrchestrator', 'MultiStageOrchestrator']
