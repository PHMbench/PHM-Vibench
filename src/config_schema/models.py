from __future__ import annotations

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
    window_size: Optional[int] = Field(None, ge=1)
    stride: Optional[int] = Field(None, ge=1)


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


TaskType = Literal["DG", "CDDG", "FS", "GFS", "pretrain", "Default_task", "generative"]


class GenerativeRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["train", "sample", "eval"] = "train"
    source_split: str = "train"
    eval_split: str = "train"
    run_test_loss_after_train: bool = False
    allow_test_reference_eval: bool = False
    domain_map_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    generated_path: Optional[str] = None
    num_steps: int = Field(8, ge=1)
    num_samples: int = Field(2, ge=1)
    length: Optional[int] = Field(None, ge=1)
    validity_status: Literal["benchmark-valid", "exploratory", "docs-only"] = "exploratory"
    allow_untrained_smoke: bool = False
    leakage_duplicate_threshold: float = Field(1e-6, ge=0.0)
    condition_sampling_policy: str = "first_metadata_repeated"
    synthetic_dataset_id: Optional[str] = None

    @model_validator(mode="after")
    def _check_mode_contract(self) -> "GenerativeRuntimeConfig":
        split = str(self.source_split).lower()
        if self.mode == "train" and split in {"val", "valid", "validation", "test", "target_test"}:
            raise ValueError("task.generative.source_split must be train for generative train mode")
        if self.mode == "sample" and not self.checkpoint_path and not self.allow_untrained_smoke:
            raise ValueError("sample mode requires checkpoint_path unless allow_untrained_smoke=true")
        if self.mode == "eval" and not self.generated_path:
            raise ValueError("eval mode requires generated_path")
        eval_split = str(self.eval_split).lower()
        if eval_split in {"test", "target_test"} and not self.allow_test_reference_eval:
            raise ValueError("test eval requires allow_test_reference_eval=true")
        return self


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: TaskType = Field(..., description="Task type key used by task_factory.")
    name: str = Field(..., description="Task name under task.type.")

    target_system_id: Optional[List[int]] = None
    generative: Optional[GenerativeRuntimeConfig] = None

    @model_validator(mode="after")
    def _check_target_system_id(self) -> "TaskConfig":
        if self.target_system_id is not None:
            if not self.target_system_id:
                raise ValueError("task.target_system_id must not be empty when provided")
        if self.type == "generative" and self.generative is None:
            raise ValueError("task.type=generative requires task.generative")
        return self


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(..., description="Trainer implementation name under trainer_factory.")
    monitor: str = Field("val_loss", description="Metric monitored by checkpoint/early stopping callbacks.")
    device: str = Field("auto", description="Trainer device selector, e.g. cpu/cuda/auto.")
    gpus: int = Field(1, ge=0, description="Device count consumed by Default_trainer.")
    num_epochs: int = Field(1, ge=1)
    log_every_n_steps: Optional[int] = Field(None, ge=1)
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
        return self
