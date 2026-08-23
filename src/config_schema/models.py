from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    project: str = Field(..., description="Experiment short name, used in output organization.")
    seed: int = Field(..., description="Required global random seed.")
    output_dir: str = Field(..., description="Base output directory (prefer repo-relative).")
    iterations: int = Field(..., ge=1, description="Required number of repeated runs.")
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
    metadata_file: str = Field(..., description="Metadata filename relative to data_dir (xlsx/csv).")
    batch_size: Optional[int] = Field(None, ge=1)
    num_workers: Optional[int] = Field(None, ge=0)
    split: Optional[DataSplitConfig] = None


OperatorName = Literal[
    "I",
    "D1",
    "ABS",
    "SQUARE",
    "MA3",
    "MA5",
    "HT",
    "FFT_MAG",
    "F_ID",
]


class XOANOperatorPathConfig(BaseModel):
    """Fail-closed scientific controls for the standalone P07 method."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    dictionary_id: str = Field(..., min_length=1)
    dictionary_version: str = Field(..., min_length=1)
    stage_operators: List[List[OperatorName]] = Field(..., min_length=1, max_length=8)
    addable_stage_operators: List[List[OperatorName]] = Field(
        ..., min_length=1, max_length=8
    )
    hidden_dim: int = Field(64, ge=1)
    temperature: float = Field(1.0, gt=0.0)
    relaxation: Literal["sparsemax"] = "sparsemax"
    relaxation_version: Literal["sparsemax-euclidean-projection-1"] = (
        "sparsemax-euclidean-projection-1"
    )
    support_tolerance: float = Field(1e-8, ge=0.0, le=1e-4)
    execution_mode: Literal["relaxed"] = "relaxed"
    tie_break_rule: Literal["registry_order"] = "registry_order"
    input_kind: Literal["blc_real_series"] = "blc_real_series"
    entropy_weight: float = Field(0.5, ge=0.0)
    export_gap_weight: float = Field(0.5, ge=0.0)
    eps: float = Field(1e-8, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _check_typed_dictionary(self) -> "XOANOperatorPathConfig":
        if len(self.addable_stage_operators) != len(self.stage_operators):
            raise ValueError(
                "model.operator_path.addable_stage_operators must have the same "
                "number of stages as stage_operators"
            )
        current_kind = "blc_real_series"
        signatures = {
            "I": ("blc_real_series", "blc_real_series"),
            "D1": ("blc_real_series", "blc_real_series"),
            "ABS": ("blc_real_series", "blc_real_series"),
            "SQUARE": ("blc_real_series", "blc_real_series"),
            "MA3": ("blc_real_series", "blc_real_series"),
            "MA5": ("blc_real_series", "blc_real_series"),
            "HT": ("blc_real_series", "blc_real_series"),
            "FFT_MAG": ("blc_real_series", "blc_frequency_magnitude"),
            "F_ID": ("blc_frequency_magnitude", "blc_frequency_magnitude"),
        }
        for stage, (operators, addable) in enumerate(
            zip(self.stage_operators, self.addable_stage_operators)
        ):
            if not operators:
                raise ValueError(f"model.operator_path.stage_operators[{stage}] is empty")
            if len(set(operators)) != len(operators):
                raise ValueError(
                    f"model.operator_path.stage_operators[{stage}] contains duplicates"
                )
            if len(set(addable)) != len(addable):
                raise ValueError(
                    f"model.operator_path.addable_stage_operators[{stage}] contains duplicates"
                )
            overlap = set(operators).intersection(addable)
            if overlap:
                raise ValueError(
                    f"model.operator_path stage {stage} active/addable dictionaries overlap"
                )
            candidates = operators + addable
            inputs = {signatures[name][0] for name in candidates}
            outputs = {signatures[name][1] for name in candidates}
            if inputs != {current_kind} or len(outputs) != 1:
                raise ValueError(
                    f"model.operator_path stage {stage} has incompatible type signatures"
                )
            current_kind = next(iter(outputs))
        if self.entropy_weight + self.export_gap_weight <= 0:
            raise ValueError("at least one insufficiency-score weight must be positive")
        return self


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = Field(..., description="Top-level model family key used by model_factory.")
    name: str = Field(..., description="Concrete model implementation name under model.type.")

    embedding: Optional[str] = None
    backbone: Optional[str] = None
    task_head: Optional[str] = None
    operator_path: Optional[XOANOperatorPathConfig] = None

    @model_validator(mode="after")
    def _check_isfm_components(self) -> "ModelConfig":
        if self.type == "ISFM":
            missing = [k for k in ["embedding", "backbone", "task_head"] if not getattr(self, k)]
            if missing:
                raise ValueError(f"model.type=ISFM requires: {', '.join(missing)}")
        return self

    @model_validator(mode="after")
    def _check_xoan_operator_path(self) -> "ModelConfig":
        if self.type != "X_model" or self.name != "XOANOperatorPath":
            return self
        if self.operator_path is None:
            raise ValueError("X_model/XOANOperatorPath requires model.operator_path")
        allowed_extra = {
            "device",
            "in_channels",
            "num_classes",
            "classifier_hidden_dim",
            "dropout",
            "inference_mode",
        }
        unexpected = sorted(set(self.model_extra or {}).difference(allowed_extra))
        if unexpected:
            raise ValueError(
                f"X_model/XOANOperatorPath has unsupported model fields: {unexpected}"
            )
        for name in ("in_channels", "num_classes", "classifier_hidden_dim"):
            value = getattr(self, name, None)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"X_model/XOANOperatorPath requires positive model.{name}")
        dropout = getattr(self, "dropout", None)
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise ValueError("X_model/XOANOperatorPath requires numeric model.dropout")
        if not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("model.dropout must be finite and in [0, 1)")
        if getattr(self, "inference_mode", None) not in {"relaxed", "discrete"}:
            raise ValueError("model.inference_mode must be relaxed or discrete")
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
    num_epochs: int = Field(..., ge=1)
    device: Optional[Literal["cpu", "cuda", "auto"]] = None
    gpus: Optional[int] = Field(None, ge=1)
    devices: Optional[int] = Field(None, ge=1)
    test_after_fit: Optional[bool] = None
    monitor: Optional[str] = None
    monitor_mode: Optional[Literal["min", "max"]] = None
    extensions: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional orchestration extensions (e.g., explain/report/collect/agent) "
            "hanging under trainer.extensions.*; must be safe to ignore when unsupported."
        ),
    )

    @model_validator(mode="after")
    def _reject_legacy_epoch_alias(self) -> "TrainerConfig":
        if "max_epochs" in (self.model_extra or {}):
            raise ValueError(
                "trainer.max_epochs is unsupported; use the single public field "
                "trainer.num_epochs"
            )
        return self


_CLASSIFICATION_LIFECYCLE_PIPELINES = frozenset(
    {
        "Pipeline_01_Fault_Diagnosis",
        "Pipeline_02_Pretraining_Few_Shot",
        "Pipeline_05_Explainable_Fault_Diagnosis",
    }
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
        if (
            self.pipeline in _CLASSIFICATION_LIFECYCLE_PIPELINES
            and self.trainer.test_after_fit is None
        ):
            raise ValueError(
                f"pipeline={self.pipeline} requires explicit trainer.test_after_fit"
            )
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
