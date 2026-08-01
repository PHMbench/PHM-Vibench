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
    stage: Literal["fit_validate_only", "fit_validate_test"] = Field(
        "fit_validate_test",
        description=(
            "Execution stage. The default preserves the legacy fit/validate/test behavior."
        ),
    )
    notes: str = Field("", description="Free-form notes.")

    @model_validator(mode="after")
    def _check_uppercase_env_values(self) -> "EnvironmentConfig":
        for k, v in self.__dict__.items():
            if k.isupper() and not isinstance(v, (str, int, float, bool)):
                raise ValueError(f"environment.{k} must be a scalar (got {type(v).__name__})")
        return self


class DataSplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal[
        "legacy_windows",
        "grouped_metadata",
        "preassigned_metadata",
    ] = "legacy_windows"
    group_key: Optional[str] = None
    stratify_key: Optional[str] = None
    split_key: Optional[str] = None
    seed: int = 42
    test_policy: Literal["partition", "task_defined"] = "partition"
    fractions: Optional[Dict[Literal["train", "val", "test"], float]] = None
    manifest_path: Optional[str] = None

    @model_validator(mode="after")
    def _check_split_protocol(self) -> "DataSplitConfig":
        if self.strategy == "legacy_windows":
            return self
        if self.strategy == "preassigned_metadata":
            if not self.split_key:
                raise ValueError(
                    "data.split.split_key is required for preassigned_metadata"
                )
            if not self.group_key:
                raise ValueError(
                    "data.split.group_key is required for preassigned_metadata"
                )
            if not self.manifest_path:
                raise ValueError(
                    "data.split.manifest_path is required for preassigned_metadata"
                )
            if self.test_policy != "partition":
                raise ValueError(
                    "preassigned_metadata requires test_policy=partition"
                )
            if self.fractions is not None:
                raise ValueError(
                    "data.split.fractions must be omitted for preassigned_metadata"
                )
            return self
        if not self.group_key:
            raise ValueError("data.split.group_key is required for grouped_metadata")
        if not self.manifest_path:
            raise ValueError("data.split.manifest_path is required for grouped_metadata")
        expected = (
            {"train", "val", "test"}
            if self.test_policy == "partition"
            else {"train", "val"}
        )
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
    metadata_file: str = Field(
        ...,
        description=(
            "Legacy runtime metadata key: a filename relative to data_dir or an "
            "absolute path canonicalized from metadata_path."
        ),
    )
    metadata_path: Optional[str] = Field(
        None,
        description=(
            "Explicit metadata path. The legacy runtime key metadata_file is populated "
            "with the same value."
        ),
    )
    batch_size: Optional[int] = Field(None, ge=1)
    num_workers: Optional[int] = Field(None, ge=0)
    split_strategy: Optional[
        Literal["legacy_windows", "grouped_metadata", "preassigned_metadata"]
    ] = None
    split: Optional[DataSplitConfig] = None

    @model_validator(mode="before")
    @classmethod
    def _canonicalize_metadata_path(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        values = dict(values)
        metadata_path = values.get("metadata_path")
        metadata_file = values.get("metadata_file")
        if metadata_path is None:
            return values
        if not isinstance(metadata_path, str) or not metadata_path.strip():
            raise ValueError("data.metadata_path must be a non-empty string")
        if metadata_file is not None and metadata_file != metadata_path:
            raise ValueError(
                "data.metadata_path and data.metadata_file must agree when both are provided"
            )
        values["metadata_path"] = metadata_path
        values["metadata_file"] = metadata_path
        return values

    @model_validator(mode="after")
    def _check_split_strategy_alias(self) -> "DataConfig":
        if (
            self.split_strategy is not None
            and self.split is not None
            and self.split_strategy != self.split.strategy
        ):
            raise ValueError(
                "data.split_strategy must match data.split.strategy when both are provided"
            )
        return self


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


TaskType = Literal[
    "DG",
    "CDDG",
    "FS",
    "GFS",
    "pretrain",
    "Default_task",
    "generative",
]


class GradientConstraintConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["fic"]
    epsilon: float = Field(2.0, gt=0.0)


class PopulationRegularizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    weight: float = Field(0.1, gt=0.0)
    dependency: Literal["pearson_correlation"] = "pearson_correlation"
    estimator: Literal["biased"] = "biased"
    rbf_bandwidths: List[float] = Field(
        default_factory=lambda: [0.1, 0.5, 1.0, 2.0]
    )
    same_time_per_batch: bool = True

    @model_validator(mode="after")
    def _check_population_contract(self) -> "PopulationRegularizationConfig":
        if not self.rbf_bandwidths or any(
            value <= 0.0 for value in self.rbf_bandwidths
        ):
            raise ValueError(
                "task.population_regularization.rbf_bandwidths must be positive"
            )
        if self.enabled and not self.same_time_per_batch:
            raise ValueError(
                "enabled population regularization requires same_time_per_batch=true"
            )
        return self


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: TaskType = Field(..., description="Task type key used by task_factory.")
    name: str = Field(..., description="Task name under task.type.")

    target_system_id: Optional[List[int]] = None
    loss: Optional[str] = None
    gradient_constraint: Optional[GradientConstraintConfig] = None
    population_regularization: Optional[PopulationRegularizationConfig] = None

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

    pipeline: str = Field(..., description="Pipeline module name under src/ (e.g. Pipeline_01_Fault_Diagnosis).")
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
        population = self.task.population_regularization
        if population is not None and population.enabled:
            if (
                self.task.type != "generative"
                or self.task.name != "conditional_flow_matching"
            ):
                raise ValueError(
                    "population_regularization is supported only by "
                    "conditional_flow_matching"
                )
            if int(getattr(self.model, "in_channels", 0)) < 2:
                raise ValueError(
                    "population_regularization requires model.in_channels >= 2"
                )
            if self.data.batch_size is None or self.data.batch_size < 2:
                raise ValueError(
                    "population_regularization requires data.batch_size >= 2"
                )
        return self
