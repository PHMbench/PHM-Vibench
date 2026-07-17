from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    project: str = Field(..., description="Experiment short name, used in output organization.")
    seed: int = Field(42, description="Global random seed.")
    output_dir: str = Field(..., description="Base output directory (prefer repo-relative).")
    iterations: int = Field(1, ge=1, description="Repeat runs with different seeds.")
    notes: str = Field("", description="Free-form notes.")

    @model_validator(mode="after")
    def _check_uppercase_env_values(self) -> "EnvironmentConfig":
        for k, v in self.__dict__.items():
            if k.isupper() and not isinstance(v, (str, int, float, bool)):
                raise ValueError(f"environment.{k} must be a scalar (got {type(v).__name__})")
        return self


class DataSplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["legacy_windows", "grouped_metadata"] = "legacy_windows"
    group_key: Optional[str] = None
    stratify_key: Optional[str] = None
    seed: int = 42
    test_policy: Literal["partition", "task_defined"] = "partition"
    fractions: Optional[Dict[Literal["train", "val", "test"], float]] = None
    manifest_path: Optional[str] = None

    @model_validator(mode="after")
    def _check_grouped_protocol(self) -> "DataSplitConfig":
        if self.strategy == "legacy_windows":
            return self
        if not self.group_key:
            raise ValueError("data.split.group_key is required for grouped_metadata")
        if not self.manifest_path:
            raise ValueError("data.split.manifest_path is required for grouped_metadata")
        expected = {"train", "val", "test"} if self.test_policy == "partition" else {"train", "val"}
        fractions = self.fractions or {}
        if set(fractions) != expected:
            raise ValueError(
                f"data.split.fractions must contain exactly {sorted(expected)} "
                f"for {self.test_policy}"
            )
        values = list(fractions.values())
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("data.split.fractions values must be finite and positive")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("data.split.fractions must sum to 1.0")
        return self


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    data_dir: str = Field(..., description="Dataset root dir containing metadata and processed files.")
    metadata_file: str = Field(..., description="Metadata filename relative to data_dir (xlsx/csv).")
    batch_size: Optional[int] = Field(None, ge=1)
    num_workers: Optional[int] = Field(None, ge=0)
    split: Optional[DataSplitConfig] = None


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = Field(..., description="Top-level model family key used by model_factory.")
    name: str = Field(..., description="Concrete model implementation name under model.type.")

    embedding: Optional[str] = None
    backbone: Optional[str] = None
    task_head: Optional[str] = None

    @model_validator(mode="after")
    def _check_isfm_components(self) -> "ModelConfig":
        if self.type == "ISFM":
            missing = [k for k in ["embedding", "backbone", "task_head"] if not getattr(self, k)]
            if missing:
                raise ValueError(f"model.type=ISFM requires: {', '.join(missing)}")
        return self


TaskType = Literal["DG", "CDDG", "FS", "GFS", "pretrain", "Default_task"]


class GradientConstraintConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["fic"]
    epsilon: float = Field(2.0, gt=0.0)


class PPTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["ssl", "supervised"] = "ssl"
    order_axes: List[Literal["time", "channel"]] = Field(
        default_factory=lambda: ["time", "channel"]
    )
    weighting: Literal["fixed", "uncertainty"] = "fixed"
    weak_swaps: int = Field(1, ge=1)
    strong_swaps: int = Field(5, ge=1)
    channel_weak_swaps: int = Field(1, ge=1)
    channel_strong_swaps: int = Field(2, ge=1)
    permutation_bank_size: int = Field(256, ge=1)
    permutation_seed: int = 42
    temperature: float = Field(0.1, gt=0.0)
    consistency_weight: float = Field(1.0, ge=0.0)
    contrastive_weight: float = Field(1.0, ge=0.0)
    classification_weight: float = Field(1.0, ge=0.0)

    @model_validator(mode="after")
    def _check_permutations(self) -> "PPTConfig":
        if not self.order_axes or len(self.order_axes) != len(set(self.order_axes)):
            raise ValueError("task.ppt.order_axes must be non-empty and unique")
        if self.weak_swaps >= self.strong_swaps:
            raise ValueError("task.ppt.strong_swaps must exceed weak_swaps")
        if "channel" in self.order_axes and self.channel_weak_swaps >= self.channel_strong_swaps:
            raise ValueError(
                "task.ppt.channel_strong_swaps must exceed channel_weak_swaps"
            )
        if self.consistency_weight == 0.0 and self.contrastive_weight == 0.0:
            raise ValueError("at least one PPT self-supervised loss weight must be positive")
        return self


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: TaskType = Field(..., description="Task type key used by task_factory.")
    name: str = Field(..., description="Task name under task.type.")

    target_system_id: Optional[List[int]] = None
    loss: Optional[str] = None
    gradient_constraint: Optional[GradientConstraintConfig] = None
    ppt: Optional[PPTConfig] = None

    @model_validator(mode="after")
    def _check_target_system_id(self) -> "TaskConfig":
        if self.target_system_id is not None:
            if not self.target_system_id:
                raise ValueError("task.target_system_id must not be empty when provided")
        return self

    @model_validator(mode="after")
    def _check_gradient_constraint(self) -> "TaskConfig":
        if self.gradient_constraint is not None and (self.loss or "").upper() != "CE":
            raise ValueError("task.gradient_constraint.name=fic requires task.loss=CE")
        return self


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="Trainer implementation name under trainer_factory.")
    num_epochs: Optional[int] = Field(None, ge=1)
    extensions: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional orchestration extensions (e.g., explain/report/collect/agent) "
            "hanging under trainer.extensions.*; must be safe to ignore when unsupported."
        ),
    )


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    pipeline: str = Field(..., description="Pipeline module name under src/ (e.g. Pipeline_01_default).")
    environment: EnvironmentConfig
    data: DataConfig
    model: ModelConfig
    task: TaskConfig
    trainer: TrainerConfig

    @model_validator(mode="after")
    def _basic_coupling_checks(self) -> "ExperimentConfig":
        if self.pipeline and not self.pipeline.startswith("Pipeline_"):
            raise ValueError("pipeline should be a src/Pipeline_*.py module name")
        split = self.data.split
        if split and split.strategy == "grouped_metadata":
            if self.task.type in {"FS", "GFS"}:
                raise ValueError(
                    f"grouped_metadata does not define episode-safe splitting for {self.task.type}"
                )
            if self.task.type in {"DG", "CDDG"} and split.test_policy != "task_defined":
                raise ValueError(f"task.type={self.task.type} requires test_policy=task_defined")
            if self.task.type not in {"DG", "CDDG"} and split.test_policy == "task_defined":
                raise ValueError("test_policy=task_defined is only supported for DG and CDDG")
        if self.task.name == "ppt_order":
            if self.task.ppt is None:
                raise ValueError("task.name=ppt_order requires task.ppt")
            if (
                self.model.type != "ISFM"
                or self.model.embedding != "E_03_Patch"
                or self.model.backbone != "B_08_PatchTST"
            ):
                raise ValueError(
                    "ppt_order requires ISFM with E_03_Patch and B_08_PatchTST"
                )
            if not bool(getattr(self.model, "channel_independent", False)):
                raise ValueError("ppt_order requires model.channel_independent=true")
            if "channel" in self.task.ppt.order_axes and int(
                getattr(self.model, "input_dim", 0)
            ) < 3:
                raise ValueError("channel-order PPT requires model.input_dim >= 3")
        return self
