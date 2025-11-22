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
import logging

from copy import deepcopy

logger = logging.getLogger(__name__)

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


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并配置，只更新叶子节点，避免破坏嵌套结构

    Args:
        base: 基础配置字典
        override: 覆盖配置字典

    Returns:
        合并后的配置字典
    """
    from copy import deepcopy

    if not isinstance(base, dict) or not isinstance(override, dict):
        return override

    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def nested_set(config: dict, path: str, value: Any) -> None:
    """安全设置嵌套配置值，支持点号路径

    Args:
        config: 配置字典
        path: 点号路径，如 "task.lr" 或 "model.d_model"
        value: 要设置的值
    """
    keys = path.split('.')
    current = config

    # 创建嵌套路径
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    # 设置最终值
    current[keys[-1]] = value


def parse_stage_overrides(override_list):
    """
    解析CLI阶段覆盖参数，支持多种格式

    支持格式：
    1. stages[0].task.lr=0.001     # 新格式：推荐使用
    2. stage_1.task.lr=0.001      # 旧格式：向后兼容
    3. trainer.max_epochs=10       # 全局覆盖：应用到所有阶段
    """
    if not override_list:
        return {}, {}

    import yaml
    import re

    stage_overrides = []
    global_overrides = {}

    # 初始化阶段覆盖列表
    # 注意：这里返回列表格式，用于和现有代码兼容
    processed_stage_overrides = {}

    for override in override_list:
        if '=' not in override:
            continue

        try:
            key, value = override.split('=', 1)
            key = key.strip()
            value = _parse_value(value.strip())

            if override.startswith('stages['):
                # 解析新格式：stages[index].path=value
                parsed = _parse_stages_index_format(override)
                if parsed:
                    stage_idx, path, value = parsed
                    if stage_idx not in processed_stage_overrides:
                        processed_stage_overrides[stage_idx] = {}
                    _set_nested_value(processed_stage_overrides[stage_idx], path, value)
                else:
                    logger.warning(f"Failed to parse stages[index] format: {override}")

            elif key.startswith('stage_'):
                # 解析旧格式：stage_N.path=value
                parsed = _parse_stage_n_format(override)
                if parsed:
                    stage_idx, path, value = parsed
                    if stage_idx not in processed_stage_overrides:
                        processed_stage_overrides[stage_idx] = {}
                    _set_nested_value(processed_stage_overrides[stage_idx], path, value)
                else:
                    logger.warning(f"Failed to parse stage_N format: {override}")

            else:
                # 解析全局覆盖：path=value
                if '.' in key:
                    _set_nested_value(global_overrides, key, value)
                else:
                    global_overrides[key] = value

        except Exception as e:
            logger.error(f"解析覆盖参数失败: {override}, 错误: {e}")
            raise ValueError(f"Invalid override format: {override}")

    # 转换为列表格式以保持向后兼容
    max_stage_idx = max(processed_stage_overrides.keys(), default=-1) + 1
    stage_overrides = [processed_stage_overrides.get(i, {}) for i in range(max_stage_idx)]

    return global_overrides, stage_overrides


def _parse_stages_index_format(override: str):
    """解析stages[index]格式"""
    # 匹配模式：stages[0].task.lr=0.001
    pattern = r'stages\[(\d+)\]\.(.+?)=(.+)'
    match = re.match(pattern, override)

    if match:
        stage_idx = int(match.group(1))
        path = match.group(2)
        value_str = match.group(3)

        # 尝试解析值的类型
        value = _parse_value(value_str)

        return stage_idx, path, value

    return None


def _parse_stage_n_format(override: str):
    """解析stage_N格式"""
    # 分割出stage_N部分
    if '=' not in override:
        return None

    key_part, value_part = override.split('=', 1)
    parts = key_part.split('.', 1)

    if len(parts) < 2:
        return None

    stage_part, path = parts
    # 提取数字
    match = re.match(r'stage_(\d+)', stage_part)

    if match:
        stage_idx = int(match.group(1)) - 1  # stage_1 -> index 0
        value = _parse_value(value_part.strip())
        return stage_idx, path, value

    return None


def _parse_value(value_str: str):
    """智能解析值的类型"""
    import ast

    value_str = value_str.strip()

    # 尝试解析为Python字面量
    try:
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        pass

    # 特殊字符串处理
    if value_str.lower() in ['true', 'false']:
        return value_str.lower() == 'true'
    elif value_str.lower() in ['null', 'none']:
        return None
    elif value_str.startswith(('\"', "'")) and value_str.endswith(('\"', "'")):
        return value_str[1:-1]  # 去掉引号
    else:
        return value_str  # 保持字符串


def _set_nested_value(d: dict, path: str, value: Any) -> None:
    """在嵌套字典中设置值"""
    keys = path.split('.')
    current = d
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def apply_stage_overrides(stage_config, global_config, stage_overrides):
    """应用阶段特定的override配置

    Args:
        stage_config: 阶段基础配置
        global_config: 全局默认配置
        stage_overrides: 阶段特定override配置

    Returns:
        合并后的配置字典
    """
    from copy import deepcopy

    # 1. 从global_config继承默认配置
    merged_config = deepcopy(global_config)

    # 2. 应用stage特定的配置
    if stage_overrides:
        merged_config = deep_merge(merged_config, stage_overrides)

    # 3. 合并stage_config（如果存在）
    if stage_config:
        if isinstance(stage_config, dict):
            merged_config = deep_merge(merged_config, stage_config)
        elif hasattr(stage_config, '__dict__'):
            merged_config = deep_merge(merged_config, stage_config.__dict__)

    return merged_config


class MultiStageOrchestrator:
    """通用多阶段 Orchestrator，支持stages.overrides配置模式。"""

    def __init__(self, unified_config: Any, dry_run: bool = False, cli_overrides: list = None) -> None:
        self.dry_run = dry_run
        self.cli_overrides = cli_overrides or []

        # 1) 已经是 ConfigWrapper：直接使用
        if isinstance(unified_config, ConfigWrapper):
            self.cfg = unified_config

        # 2) dict：支持多种配置格式
        elif isinstance(unified_config, dict):
            # 检查是否为单YAML + stages.overrides格式
            if self._is_unified_yaml_format(unified_config):
                self.cfg = self._load_unified_yaml_config(unified_config)
            else:
                # 兼容旧的双阶段格式
                self.cfg = self._load_legacy_config(unified_config)

        # 3) 其他情况：视为单阶段配置源（路径/预设）
        else:
            self.cfg = self._load_single_stage_config(unified_config)

        # 验证每个阶段的 data/model/task 必需字段
        self._validate_stages()

    def _is_unified_yaml_format(self, config: dict) -> bool:
        """检查是否为单YAML + stages.overrides格式"""
        return (
            'stages' in config and
            isinstance(config['stages'], list) and
            len(config['stages']) > 0 and
            all(isinstance(stage, dict) and 'overrides' in stage for stage in config['stages'])
        )

    def _load_unified_yaml_config(self, config: dict) -> ConfigWrapper:
        """加载单YAML + stages.overrides格式配置"""
        # 提取全局配置作为基础
        global_sections = ['data', 'model', 'task', 'trainer', 'environment']
        global_config = {section: config.get(section, {}) for section in global_sections}

        # 处理CLI overrides
        global_cli_overrides, stage_cli_overrides = parse_stage_overrides(self.cli_overrides)
        if global_cli_overrides:
            global_config = deep_merge(global_config, global_cli_overrides)

        # 处理stages
        processed_stages = []
        for stage_idx, stage_dict in enumerate(config['stages']):
            stage_name = stage_dict.get('name', f"stage_{stage_idx}")

            # 合并配置：全局配置 + stage overrides + CLI overrides
            stage_overrides = stage_dict.get('overrides', {})

            # 应用CLI stage-specific overrides
            if stage_name in stage_cli_overrides:
                stage_overrides = deep_merge(stage_overrides, stage_cli_overrides[stage_name])

            merged_stage_config = apply_stage_overrides(
                stage_dict.get('config', {}),
                global_config,
                stage_overrides
            )

            # 转换为Namespace并添加stage名称
            stage_ns = dict_to_namespace(merged_stage_config)
            if hasattr(stage_ns, 'environment'):
                stage_ns.environment.stage_name = stage_name
            else:
                stage_ns.environment = SimpleNamespace(stage_name=stage_name)

            processed_stages.append(stage_ns)

        # 保留stage_1/stage_2映射以兼容现有代码
        attrs = {'stages': processed_stages}
        if len(processed_stages) >= 1:
            attrs['stage_1'] = processed_stages[0]
        if len(processed_stages) >= 2:
            attrs['stage_2'] = processed_stages[1]

        return ConfigWrapper(**attrs)

    def _load_legacy_config(self, config: dict) -> ConfigWrapper:
        """加载旧的双阶段格式配置"""
        if 'stages' in config:
            stages_dict_list = config['stages']
        else:
            stages_dict_list = [
                config.get('stage_1', {}),
                config.get('stage_2', {}),
            ]

        stages_ns = [dict_to_namespace(s or {}) for s in stages_dict_list if s]

        # 应用CLI overrides
        global_cli_overrides, stage_cli_overrides = parse_stage_overrides(self.cli_overrides)

        # 应用CLI overrides到所有stage
        for idx, stage_ns in enumerate(stages_ns):
            stage_key = f"stage_{idx+1}"
            if stage_key in stage_cli_overrides:
                stage_dict = stage_ns.__dict__
                merged_dict = deep_merge(stage_dict, stage_cli_overrides[stage_key])
                stages_ns[idx] = dict_to_namespace(merged_dict)

        attrs = {'stages': stages_ns}
        if len(stages_ns) >= 1:
            attrs['stage_1'] = stages_ns[0]
        if len(stages_ns) >= 2:
            attrs['stage_2'] = stages_ns[1]

        return ConfigWrapper(**attrs)

    def _load_single_stage_config(self, config_source: Any) -> ConfigWrapper:
        """加载单阶段配置"""
        base_cfg = load_config(config_source)
        stage_ns = SimpleNamespace(
            data=getattr(base_cfg, 'data', SimpleNamespace()),
            model=getattr(base_cfg, 'model', SimpleNamespace()),
            task=getattr(base_cfg, 'task', SimpleNamespace()),
            trainer=getattr(base_cfg, 'trainer', SimpleNamespace()),
            environment=getattr(base_cfg, 'environment', SimpleNamespace()),
        )
        return ConfigWrapper(stages=[stage_ns], stage_1=stage_ns)

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
