"""Registry data exposed in the research console."""

from __future__ import annotations

import csv
import importlib
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import yaml


def repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=8)
def _load_csv(path_str: str) -> List[Dict[str, str]]:
    path = Path(path_str)
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_model_registry() -> List[Dict[str, str]]:
    """Load model registry rows."""
    return _load_csv(str(repo_root() / "src" / "model_factory" / "model_registry.csv"))


def load_task_registry() -> List[Dict[str, str]]:
    """Load task registry rows."""
    return _load_csv(str(repo_root() / "src" / "task_factory" / "task_registry.csv"))


@lru_cache(maxsize=1)
def load_data_factories() -> List[Dict[str, str]]:
    """Load available data factories."""
    from src.data_factory import DATA_FACTORY_REGISTRY

    available = DATA_FACTORY_REGISTRY.available()
    rows: List[Dict[str, str]] = []
    for name, factory in sorted(available.items()):
        rows.append(
            {
                "name": name,
                "module": getattr(factory, "__module__", ""),
                "symbol": getattr(factory, "__name__", ""),
            }
        )
    return rows


@lru_cache(maxsize=1)
def load_trainer_registry() -> List[Dict[str, str]]:
    """Load registered trainers and shipped trainer presets."""
    from src.trainer_factory import TRAINER_REGISTRY

    importlib.import_module("src.trainer_factory.Default_trainer")
    rows: List[Dict[str, str]] = []
    for name, trainer in sorted(TRAINER_REGISTRY.available().items()):
        rows.append(
            {
                "name": name,
                "module": getattr(trainer, "__module__", ""),
                "symbol": getattr(trainer, "__name__", ""),
            }
        )
    return rows


@lru_cache(maxsize=1)
def load_trainer_presets() -> List[Dict[str, str]]:
    """Load YAML trainer presets under configs/base/trainer."""
    base_dir = repo_root() / "configs" / "base" / "trainer"
    rows: List[Dict[str, str]] = []
    for path in sorted(base_dir.glob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        trainer_cfg = config.get("trainer", {})
        if not isinstance(trainer_cfg, dict):
            trainer_cfg = {}
        rows.append(
            {
                "path": str(path.relative_to(repo_root())),
                "name": str(trainer_cfg.get("name") or ""),
                "device": str(trainer_cfg.get("device") or ""),
                "gpus": str(trainer_cfg.get("gpus") or ""),
                "num_epochs": str(trainer_cfg.get("num_epochs") or ""),
            }
        )
    return rows
