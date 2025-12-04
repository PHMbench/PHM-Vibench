"""
Pipeline Adapters

Adapters that convert existing pipeline configs (P02/P03/P04) into
the unified two-stage format expected by the TwoStageOrchestrator.

The unified format requires:
{
  'stage_1': { 'data': ..., 'model': ..., 'task': ..., 'trainer': ..., 'environment': ... },
  'stage_2': { 'data': ..., 'model': ..., 'task': ..., 'trainer': ..., 'environment': ... }
}
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from copy import deepcopy

from src.configs.config_utils import load_config, merge_with_local_override, ConfigWrapper


def _stage_from(cfg: ConfigWrapper) -> Dict[str, Any]:
    """Extract a stage dict from a standard PHM-Vibench config wrapper."""
    return {
        'data': deepcopy(getattr(cfg, 'data', {}).__dict__ if hasattr(cfg, 'data') else {}),
        'model': deepcopy(getattr(cfg, 'model', {}).__dict__ if hasattr(cfg, 'model') else {}),
        'task': deepcopy(getattr(cfg, 'task', {}).__dict__ if hasattr(cfg, 'task') else {}),
        'trainer': deepcopy(getattr(cfg, 'trainer', {}).__dict__ if hasattr(cfg, 'trainer') else {}),
        'environment': deepcopy(getattr(cfg, 'environment', {}).__dict__ if hasattr(cfg, 'environment') else {}),
    }


def adapt_p02(pretrain_cfg_path: str, fewshot_cfg_path: str, local_config: Optional[str] = None) -> Dict[str, Any]:
    """Adapter for Pipeline_02: merge pretrain + fewshot into unified two-stage config."""
    pre = merge_with_local_override(pretrain_cfg_path, local_config)
    fs = merge_with_local_override(fewshot_cfg_path, local_config)

    unified = {
        'stage_1': _stage_from(pre),
        'stage_2': _stage_from(fs),
    }
    # Hint downstream about mode (non-breaking)
    unified['stage_1']['task']['training_stage'] = 'pretrain'
    unified['stage_2']['task']['training_stage'] = 'finetune'
    unified['stage_2']['task']['adapt_mode'] = 'fewshot'
    return unified


def adapt_p03(config_path: str, local_config: Optional[str] = None) -> Dict[str, Any]:
    """Adapter for Pipeline_03: build two-stage config from a single multi-stage YAML.

    Minimal mapping: use top-level sections as defaults for both stages. If the
    YAML includes `training.stage_1_pretraining` or `training.stage_2_finetuning`,
    you can enrich the stage dicts upstream before calling the orchestrator.
    """
    cfg = merge_with_local_override(config_path, local_config)
    stage_base = _stage_from(cfg)
    s1 = deepcopy(stage_base)
    s2 = deepcopy(stage_base)
    s1['task']['training_stage'] = 'pretrain'
    s2['task']['training_stage'] = 'finetune'
    return {'stage_1': s1, 'stage_2': s2}


def adapt_p04(config_path: str, local_config: Optional[str] = None) -> Dict[str, Any]:
    """Adapter for Pipeline_04: unified metric learning → two-stage config.

    The unified metric YAMLs often include nested training sections; this adapter
    does a minimal extraction and sets stages to be identical unless the caller
    modifies stage_2 specifics upstream.
    """
    cfg = merge_with_local_override(config_path, local_config)
    s1 = _stage_from(cfg)
    s2 = deepcopy(s1)
    s1['task']['training_stage'] = 'pretrain'
    s2['task']['training_stage'] = 'finetune'
    return {'stage_1': s1, 'stage_2': s2}


__all__ = ['adapt_p02', 'adapt_p03', 'adapt_p04']

