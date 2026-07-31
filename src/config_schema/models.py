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


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    data_dir: str = Field(..., description="Dataset root dir containing metadata and processed files.")
    metadata_file: str = Field(..., description="Metadata filename relative to data_dir (xlsx/csv).")
    batch_size: Optional[int] = Field(None, ge=1)
    num_workers: Optional[int] = Field(None, ge=0)


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
        if not math.isfinite(self.weight):
            raise ValueError("task.population_regularization.weight must be finite")
        if not self.rbf_bandwidths or any(
            not math.isfinite(value) or value <= 0.0
            for value in self.rbf_bandwidths
        ):
            raise ValueError(
                "task.population_regularization.rbf_bandwidths must be finite and positive"
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
    population_regularization: Optional[PopulationRegularizationConfig] = None

    @model_validator(mode="after")
    def _check_target_system_id(self) -> "TaskConfig":
        if self.target_system_id is not None:
            if not self.target_system_id:
                raise ValueError("task.target_system_id must not be empty when provided")
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
