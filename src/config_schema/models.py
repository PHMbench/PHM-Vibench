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
        "grouped_kfold",
        "preassigned_metadata",
    ] = "legacy_windows"
    group_key: Optional[str] = None
    stratify_key: Optional[str] = None
    split_key: Optional[str] = None
    seed: int = 42
    outer_folds: Optional[int] = Field(None, ge=3)
    outer_fold: Optional[int] = Field(None, ge=0)
    validation_offset: Optional[int] = Field(None, ge=1)
    expected_manifest_payload_sha256: Optional[str] = None
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
        if self.strategy == "grouped_kfold":
            if not self.group_key:
                raise ValueError("data.split.group_key is required for grouped_kfold")
            if not self.stratify_key:
                raise ValueError(
                    "data.split.stratify_key is required for grouped_kfold"
                )
            if not self.manifest_path:
                raise ValueError("data.split.manifest_path is required for grouped_kfold")
            if self.outer_folds is None:
                raise ValueError("data.split.outer_folds is required for grouped_kfold")
            if self.outer_fold is None or self.outer_fold >= self.outer_folds:
                raise ValueError(
                    "data.split.outer_fold must be in [0, outer_folds) for grouped_kfold"
                )
            if self.validation_offset is None or self.validation_offset >= self.outer_folds:
                raise ValueError(
                    "data.split.validation_offset must be in [1, outer_folds) "
                    "for grouped_kfold"
                )
            if not self.expected_manifest_payload_sha256:
                raise ValueError(
                    "data.split.expected_manifest_payload_sha256 is required for "
                    "grouped_kfold"
                )
            if self.test_policy != "partition":
                raise ValueError("grouped_kfold requires test_policy=partition")
            if self.fractions is not None:
                raise ValueError(
                    "data.split.fractions must be omitted for grouped_kfold"
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

    factory_name: str = Field("default", description="Registered data-factory name.")
    data_dir: Optional[str] = Field(
        None, description="Dataset root dir containing metadata and processed files."
    )
    metadata_file: Optional[str] = Field(
        None, description="Metadata filename relative to data_dir (xlsx/csv)."
    )
    metadata_path: Optional[str] = Field(
        None,
        description=(
            "Explicit metadata path. It is canonicalized to metadata_file for "
            "the legacy data factory."
        ),
    )
    phm_data_config: Optional[str] = Field(
        None,
        description="Configuration path for the optional phm-data-factory backend.",
    )
    dataset_name: Optional[str] = None
    batch_size: Optional[int] = Field(None, ge=1)
    num_workers: Optional[int] = Field(None, ge=0)
    split_strategy: Optional[
        Literal[
            "legacy_windows",
            "grouped_metadata",
            "grouped_kfold",
            "preassigned_metadata",
        ]
    ] = None
    split: Optional[DataSplitConfig] = None

    @model_validator(mode="before")
    @classmethod
    def _canonicalize_legacy_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        metadata_path = data.get("metadata_path")
        metadata_file = data.get("metadata_file")
        if metadata_path is not None:
            if not isinstance(metadata_path, str) or not metadata_path.strip():
                raise ValueError("data.metadata_path must be a non-empty path")
            if metadata_file is not None and str(metadata_file) != metadata_path:
                raise ValueError(
                    "data.metadata_path and data.metadata_file must agree when both "
                    "are provided"
                )
            data["metadata_file"] = metadata_path
        return data

    @model_validator(mode="after")
    def _check_factory_fields(self) -> "DataConfig":
        if self.factory_name == "phm_data":
            if not self.phm_data_config:
                raise ValueError(
                    "data.factory_name=phm_data requires data.phm_data_config"
                )
        elif not self.data_dir or not self.metadata_file:
            raise ValueError(
                "data.data_dir and data.metadata_file are required for legacy factories"
            )
        if (
            self.split_strategy is not None
            and self.split is not None
            and self.split_strategy != self.split.strategy
        ):
            raise ValueError(
                "data.split_strategy must match data.split.strategy when both are provided"
            )
        return self


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
