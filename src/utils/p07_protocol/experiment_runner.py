"""In-memory execution primitives for the preregistered P07 experiments.

The utilities in this module stop at training and atomic measurement.  They do
not write datasets or artifacts, inspect test data during tuning, select claim
thresholds, or make claim decisions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from types import SimpleNamespace
from typing import Any, Literal, Optional

import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.model_factory.CNN.AttentionCNN import Model as AttentionCNN
from src.model_factory.X_model.XOANOperatorPath import Model as XOANOperatorPath
from src.model_factory.X_model.UXFD.operator_attention import (
    ExecutableOperatorPath1D,
    ExecutableOperatorPathConfig,
)
from src.model_factory.X_model.baselines.ExplainableCNN import (
    Model as ExplainableCNN,
)

from . import path_universe as _path_universe
from .comparators import DenseOperatorMixture1D, RandomDictionaryOperatorPath1D
from .cwru_preprocessing import materialize_manifest_windows, standardize_window
from .cwru_manifest import (
    CWRUFold,
    CWRUManifest,
    ManifestSpecimen,
    WindowCoordinate,
)


PRIMARY_EXHAUSTIVE_EVALUATION_BUDGET = 216
PARAMETER_MATCH_TOLERANCE = 0.05
FROZEN_RIDGE_SKLEARN_VERSION = "1.2.2"

# Frozen DIRG held-severity profile.  These values are deliberately separate
# from the CWRU defaults: changing dataset dimensions must never trigger an
# implicit architecture search after the protocol is approved.
DIRG_IN_CHANNELS = 6
DIRG_NUM_CLASSES = 2
DIRG_OPERATOR_CLASSIFIER_HIDDEN_DIM = 67
DIRG_ATTENTION_CHANNELS = (24,)
DIRG_EXPLAINABLE_CNN_WIDTH = 9
DIRG_DROPOUT = 0.1
DIRG_EXPECTED_TRAINABLE_PARAMETER_COUNTS = (
    ("proposed", 4892),
    ("dense_operator_mixture", 4892),
    ("random_dictionary", 4892),
    ("attention_cnn", 4720),
    ("explainable_cnn", 5123),
)


@dataclass(frozen=True, slots=True)
class TrainingBudget:
    """Frozen optimizer, batching, stopping, and update budget."""

    optimizer: str = "AdamW"
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    batch_size: int = 64
    max_epochs: int = 200
    max_updates: int = 1600
    patience: int = 20
    min_delta: float = 1.0e-5

    def __post_init__(self) -> None:
        if self.optimizer != "AdamW":
            raise ValueError("P07 TrainingBudget requires optimizer='AdamW'.")
        for name in ("learning_rate", "weight_decay", "min_delta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number, not boolean.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if float(self.min_delta) < 0.0:
            raise ValueError("min_delta must be non-negative.")
        for name in ("batch_size", "max_epochs", "max_updates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if (
            isinstance(self.patience, bool)
            or not isinstance(self.patience, int)
            or self.patience < 0
        ):
            raise ValueError("patience must be a non-negative integer.")


@dataclass(frozen=True, slots=True)
class TrainingTrace:
    """Atomic in-memory training history after restoring the best epoch."""

    optimization_seed: int
    epochs_completed: int
    updates_completed: int
    best_epoch: int
    best_validation_loss: float
    training_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    stopped_early: bool
    stop_reason: Literal["early_stopping", "max_epochs", "max_updates"]
    best_state_restored: bool = True


@dataclass(frozen=True, slots=True)
class ReconstructionEvaluation:
    loss: float
    sample_count: int
    target_source: str = "path_universe.oracle_execute_path"


@dataclass(frozen=True, slots=True)
class PathLossAtom:
    raw_path: _path_universe.RawPath
    raw_path_id: str
    class_id: str
    validation_loss: float


@dataclass(frozen=True, slots=True)
class ExhaustiveSelectionResult:
    selected_path: _path_universe.RawPath
    selected_raw_path_id: str
    selected_class_id: str
    selected_validation_loss: float
    evaluated_paths: int
    evaluation_budget: int
    primary: bool
    evaluations: tuple[PathLossAtom, ...]
    tie_rule: str = "path_universe_registry_order"
    target_role: str = "validation_only"


@dataclass(frozen=True, slots=True)
class RecoveryAtoms:
    target_path: _path_universe.RawPath
    predicted_path: _path_universe.RawPath
    target_raw_path_id: str
    predicted_raw_path_id: str
    target_class_id: str
    predicted_class_id: str
    exact_match: bool
    semantic_match: bool
    raw_edit_distance: int
    canonical_edit_distance: int


@dataclass(frozen=True, slots=True)
class StabilityAtom:
    left_seed: int
    right_seed: int
    left_raw_path_id: str
    right_raw_path_id: str
    left_class_id: str
    right_class_id: str
    exact_path_agreement: bool
    semantic_path_agreement: bool
    raw_edit_distance: int
    canonical_edit_distance: int


@dataclass(frozen=True, slots=True)
class CWRUFileBatch:
    """All standardized windows belonging to one immutable file unit."""

    file_key: str
    file_name: str
    split: Literal["train", "validation", "test"]
    label: int
    class_index: int
    file_weight: float
    windows: torch.Tensor
    coordinates: tuple[WindowCoordinate, ...]

    @property
    def window_count(self) -> int:
        return int(self.windows.shape[0])


@dataclass(frozen=True, slots=True)
class CWRUFoldData:
    fold_id: str
    manifest_root_sha256: str
    train_files: tuple[CWRUFileBatch, ...]
    validation_files: tuple[CWRUFileBatch, ...]
    test_files: tuple[CWRUFileBatch, ...]
    evaluation_unit: str = "file"
    weighting: str = "equal_file"


@dataclass(frozen=True, slots=True)
class FilePrediction:
    file_key: str
    label: int
    class_index: int
    predicted_class_index: int
    correct: bool
    window_count: int
    mean_window_loss: float


@dataclass(frozen=True, slots=True)
class FileMacroEvaluation:
    split: str
    macro_accuracy: float
    macro_loss: float
    independent_unit_count: int
    total_window_count: int
    predictions: tuple[FilePrediction, ...]
    evaluation_unit: str = "file"
    weighting: str = "equal_file"


@dataclass(frozen=True, slots=True)
class MultinomialRidgeSpec:
    """Frozen train-only sklearn rule for every discrete CWRU path."""

    sklearn_version: str = FROZEN_RIDGE_SKLEARN_VERSION
    feature_scaler: str = "StandardScaler_train_only_zero_variance_scale_one"
    penalty: str = "l2"
    c: float = 1.0
    solver: str = "lbfgs"
    multi_class: str = "multinomial"
    fit_intercept: bool = True
    tolerance: float = 1.0e-8
    max_iterations: int = 1000
    evaluation_batch_size: int = 64
    stochastic_fit: bool = False
    class_order: str = "ascending_integer_class_index"
    class_tie_rule: str = "lowest_sorted_class_index"


@dataclass(frozen=True, slots=True)
class DiscreteClassifierCandidate:
    """Validation-only selection atom for one registry-order path fit."""

    raw_path: _path_universe.RawPath
    raw_path_id: str
    class_id: str
    validation_macro_accuracy: float
    validation_macro_loss: float
    solver_iterations: int


@dataclass(frozen=True, slots=True)
class DiscreteClassifierCompute:
    """Compute charged to one exhaustive discrete classifier search."""

    candidate_fits: int
    candidate_validation_evaluations: int
    total_solver_iterations: int
    wall_time_seconds: float = field(compare=False)


@dataclass(frozen=True, slots=True)
class DiscreteClassifierSelectionResult:
    """Selected frozen classifier plus the complete validation search ledger."""

    selected_path: _path_universe.RawPath
    selected_raw_path_id: str
    selected_class_id: str
    selected_validation_macro_accuracy: float
    selected_validation_macro_loss: float
    selected_model: "FrozenPathClassifier" = field(compare=False, repr=False)
    bookkeeping_seed: int
    evaluated_paths: int
    evaluation_budget: int
    primary: bool
    evaluations: tuple[DiscreteClassifierCandidate, ...]
    compute: DiscreteClassifierCompute
    classifier_spec: MultinomialRidgeSpec = MultinomialRidgeSpec()
    tie_rule: str = (
        "validation_macro_accuracy_desc_then_macro_loss_asc_then_registry_order"
    )
    fit_role: str = "train_only"
    selection_role: str = "validation_only"
    seed_role: str = "bookkeeping_only_seed_invariant_fit"
    fit_reuse_scope: str = "once_per_fold_not_per_optimization_seed"


@dataclass(frozen=True, slots=True)
class ExperimentArm:
    arm_id: str
    role: str
    model: Optional[nn.Module]
    trainable_parameter_count: Optional[int]
    parameter_match_required: bool


class OperatorCoreClassifier(nn.Module):
    """Shared file classifier head for dense and random operator-path cores."""

    def __init__(
        self,
        core: ExecutableOperatorPath1D,
        *,
        in_channels: int,
        num_classes: int,
        classifier_hidden_dim: int = 66,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.operator_path = core
        pooled_dim = 2 * _positive_int(in_channels, "in_channels")
        class_count = _positive_int(num_classes, "num_classes")
        hidden = _positive_int(classifier_hidden_dim, "classifier_hidden_dim")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise TypeError("dropout must be a real number.")
        if not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be finite and in [0, 1).")
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signal, _trace = self.operator_path(x)
        pooled = torch.cat(
            (signal.mean(dim=1), signal.var(dim=1, unbiased=False)), dim=1
        )
        return self.classifier(pooled)


class FrozenPathClassifier(nn.Module):
    """Fixed public-oracle path plus train-only multinomial ridge estimator."""

    def __init__(
        self,
        path: Sequence[str],
        *,
        in_channels: int = 2,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        record = _path_record(_path_universe.validate_raw_path(path))
        self.raw_path = record.raw_path
        self.raw_path_id = record.raw_path_id
        self.class_id = record.class_id
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.num_classes = _positive_int(num_classes, "num_classes")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least two.")
        self.spec = MultinomialRidgeSpec()
        self.scaler: Optional[StandardScaler] = None
        self.estimator: Optional[LogisticRegression] = None
        self.solver_iterations: Optional[int] = None
        self.register_buffer("_device_dtype_anchor", torch.zeros(()))

    @property
    def is_fitted(self) -> bool:
        return self.scaler is not None and self.estimator is not None

    def fit_train_files(
        self,
        train_files: Sequence[CWRUFileBatch],
    ) -> int:
        """Fit exactly once from training windows; no validation input is accepted."""

        _require_frozen_sklearn_version()
        if self.is_fitted:
            raise RuntimeError("A frozen path classifier may be fitted exactly once.")
        train_data = _validate_file_batches(train_files, required_split="train")
        if any(int(item.windows.shape[2]) != self.in_channels for item in train_data):
            raise ValueError("Training files do not match the classifier channel count.")
        observed_classes = {item.class_index for item in train_data}
        expected_classes = set(range(self.num_classes))
        if observed_classes != expected_classes:
            raise ValueError(
                "Ridge fitting requires every deterministic class index in training."
            )

        feature_blocks: list[np.ndarray] = []
        label_blocks: list[np.ndarray] = []
        with torch.no_grad():
            for item in train_data:
                pooled = self._pooled_features(item.windows)
                feature_blocks.append(
                    pooled.detach().to(device="cpu", dtype=torch.float64).numpy()
                )
                label_blocks.append(
                    np.full(item.window_count, item.class_index, dtype=np.int64)
                )
        features = np.concatenate(feature_blocks, axis=0)
        labels = np.concatenate(label_blocks, axis=0)
        if not np.isfinite(features).all():
            raise FloatingPointError("Discrete path produced non-finite ridge features.")

        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        standardized = scaler.fit_transform(features)
        if not np.isfinite(standardized).all():
            raise FloatingPointError("StandardScaler produced non-finite features.")
        if scaler.scale_ is None or np.any(~np.isfinite(scaler.scale_)):
            raise FloatingPointError("StandardScaler produced an invalid scale.")
        if np.any(scaler.scale_ <= 0.0):
            raise ValueError("StandardScaler scale must be positive.")
        if scaler.var_ is None or np.any(scaler.scale_[scaler.var_ == 0.0] != 1.0):
            raise RuntimeError("StandardScaler zero-variance safety rule drifted.")

        estimator = LogisticRegression(
            penalty=self.spec.penalty,
            C=self.spec.c,
            solver=self.spec.solver,
            multi_class=self.spec.multi_class,
            fit_intercept=self.spec.fit_intercept,
            tol=self.spec.tolerance,
            max_iter=self.spec.max_iterations,
            random_state=None,
            warm_start=False,
            n_jobs=None,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(standardized, labels)
        expected_order = np.arange(self.num_classes, dtype=np.int64)
        if not np.array_equal(estimator.classes_, expected_order):
            raise RuntimeError("LogisticRegression class ordering drifted.")
        if not np.isfinite(estimator.coef_).all() or not np.isfinite(
            estimator.intercept_
        ).all():
            raise FloatingPointError("LogisticRegression produced non-finite coefficients.")
        iterations = int(np.max(estimator.n_iter_))
        if not 0 <= iterations <= self.spec.max_iterations:
            raise RuntimeError("LogisticRegression reported invalid solver iterations.")
        self.scaler = scaler
        self.estimator = estimator
        self.solver_iterations = iterations
        self.eval()
        return iterations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scaler is None or self.estimator is None:
            raise RuntimeError("FrozenPathClassifier must be fitted before evaluation.")
        with torch.no_grad():
            pooled = self._pooled_features(x)
            features = pooled.detach().to(device="cpu", dtype=torch.float64).numpy()
            standardized = self.scaler.transform(features)
            decision = self.estimator.decision_function(standardized)
            if decision.ndim == 1:
                decision = np.stack((-decision, decision), axis=1)
            if decision.shape != (int(x.shape[0]), self.num_classes):
                raise RuntimeError("LogisticRegression returned an invalid decision shape.")
            if not np.isfinite(decision).all():
                raise FloatingPointError("LogisticRegression returned non-finite logits.")
            return torch.as_tensor(
                decision,
                dtype=x.dtype,
                device=x.device,
            )

    def _pooled_features(self, x: torch.Tensor) -> torch.Tensor:
        signal = _path_universe.oracle_execute_path(x.detach(), self.raw_path)
        if int(signal.shape[2]) != self.in_channels:
            raise ValueError("Inputs do not match the classifier channel count.")
        return torch.cat(
            (signal.mean(dim=1), signal.var(dim=1, unbiased=False)), dim=1
        )


def seed_all_rng(seed: int) -> int:
    """Seed Python, NumPy, torch CPU, and every visible CUDA generator."""

    normalized = _validate_seed(seed)
    random.seed(normalized)
    np.random.seed(normalized)
    torch.manual_seed(normalized)
    torch.cuda.manual_seed_all(normalized)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return normalized


def independent_oracle_target(
    inputs: torch.Tensor,
    target_path: Sequence[str],
) -> torch.Tensor:
    """Construct a detached target exclusively with the public independent oracle."""

    _validate_blc(inputs, "inputs")
    with torch.no_grad():
        target = _path_universe.oracle_execute_path(inputs.detach(), target_path)
    return target.detach()


def train_synthetic_reconstruction(
    core: nn.Module,
    train_inputs: torch.Tensor,
    validation_inputs: torch.Tensor,
    *,
    target_path: Sequence[str],
    optimization_seed: int,
    budget: TrainingBudget = TrainingBudget(),
) -> TrainingTrace:
    """Train one fresh per-composition core using validation-only early stopping."""

    _validate_training_budget(budget)
    seed = seed_all_rng(optimization_seed)
    device, dtype = _model_device_dtype(core)
    train_x = _coerce_blc(train_inputs, "train_inputs", device=device, dtype=dtype)
    validation_x = _coerce_blc(
        validation_inputs,
        "validation_inputs",
        device=device,
        dtype=dtype,
    )
    raw_path = _path_universe.validate_raw_path(target_path)
    train_y = independent_oracle_target(train_x, raw_path)
    validation_y = independent_oracle_target(validation_x, raw_path)
    optimizer = _adamw(core, budget)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    best_loss = math.inf
    best_epoch = 0
    best_state: Optional[dict[str, Any]] = None
    bad_epochs = 0
    updates = 0
    train_history: list[float] = []
    validation_history: list[float] = []
    stop_reason: Literal["early_stopping", "max_epochs", "max_updates"] = (
        "max_epochs"
    )

    for epoch in range(1, budget.max_epochs + 1):
        core.train()
        order = torch.randperm(int(train_x.shape[0]), generator=generator)
        epoch_loss_sum = 0.0
        epoch_sample_count = 0
        for start in range(0, len(order), budget.batch_size):
            if updates >= budget.max_updates:
                stop_reason = "max_updates"
                break
            index = order[start : start + budget.batch_size].to(device=device)
            batch_x = train_x.index_select(0, index)
            batch_y = train_y.index_select(0, index)
            optimizer.zero_grad(set_to_none=True)
            prediction = _reconstruction_output(core, batch_x)
            _validate_prediction(prediction, batch_y, "training reconstruction")
            loss = F.mse_loss(prediction, batch_y)
            _require_finite_scalar(loss, "training reconstruction loss")
            loss.backward()
            _require_finite_gradients(core)
            optimizer.step()
            _require_finite_parameters(core)
            batch_count = int(batch_x.shape[0])
            epoch_loss_sum += float(loss.detach().item()) * batch_count
            epoch_sample_count += batch_count
            updates += 1
        if epoch_sample_count == 0:
            break
        train_history.append(epoch_loss_sum / epoch_sample_count)
        validation_loss = _reconstruction_loss(
            core,
            validation_x,
            validation_y,
            batch_size=budget.batch_size,
        )
        validation_history.append(validation_loss)
        if validation_loss < best_loss - budget.min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(core.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= budget.patience and budget.patience > 0:
            stop_reason = "early_stopping"
            break
        if updates >= budget.max_updates:
            stop_reason = "max_updates"
            break

    if best_state is None or best_epoch <= 0 or not math.isfinite(best_loss):
        raise RuntimeError("Training completed without a finite validation checkpoint.")
    core.load_state_dict(best_state, strict=True)
    return TrainingTrace(
        optimization_seed=seed,
        epochs_completed=len(train_history),
        updates_completed=updates,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        training_losses=tuple(train_history),
        validation_losses=tuple(validation_history),
        stopped_early=stop_reason == "early_stopping",
        stop_reason=stop_reason,
    )


def evaluate_synthetic_reconstruction(
    core: nn.Module,
    inputs: torch.Tensor,
    *,
    target_path: Sequence[str],
    batch_size: int = 64,
) -> ReconstructionEvaluation:
    """Evaluate a fixed core without updating it or selecting a threshold."""

    size = _positive_int(batch_size, "batch_size")
    device, dtype = _model_device_dtype(core)
    x = _coerce_blc(inputs, "inputs", device=device, dtype=dtype)
    target = independent_oracle_target(x, target_path)
    loss = _reconstruction_loss(core, x, target, batch_size=size)
    return ReconstructionEvaluation(loss=loss, sample_count=int(x.shape[0]))


def select_exhaustive_oracle_path(
    validation_inputs: torch.Tensor,
    validation_targets: torch.Tensor,
    *,
    evaluation_budget: int = PRIMARY_EXHAUSTIVE_EVALUATION_BUDGET,
    primary: bool = True,
) -> ExhaustiveSelectionResult:
    """Select from the public 216-path oracle using validation loss only."""

    if isinstance(evaluation_budget, bool) or not isinstance(evaluation_budget, int):
        raise TypeError("evaluation_budget must be an integer, not boolean.")
    records = _path_universe.enumerate_path_records()
    if len(records) != PRIMARY_EXHAUSTIVE_EVALUATION_BUDGET:
        raise RuntimeError("The frozen path universe no longer contains exactly 216 paths.")
    if not 1 <= evaluation_budget <= len(records):
        raise ValueError("evaluation_budget must be in [1, 216].")
    if not isinstance(primary, bool):
        raise TypeError("primary must be a boolean.")
    if primary and evaluation_budget != PRIMARY_EXHAUSTIVE_EVALUATION_BUDGET:
        raise ValueError("The primary exhaustive selector requires evaluation_budget=216.")
    x = _validate_blc(validation_inputs, "validation_inputs").detach()
    y = _validate_blc(validation_targets, "validation_targets").detach()
    if x.shape != y.shape or x.device != y.device or x.dtype != y.dtype:
        raise ValueError(
            "validation_inputs and validation_targets must share shape, dtype, and device."
        )

    evaluations: list[PathLossAtom] = []
    best_index = -1
    best_loss = math.inf
    with torch.no_grad():
        for index, record in enumerate(records[:evaluation_budget]):
            prediction = _path_universe.oracle_execute_path(x, record.raw_path)
            loss_tensor = F.mse_loss(prediction, y)
            _require_finite_scalar(loss_tensor, "exhaustive validation loss")
            loss = float(loss_tensor.item())
            evaluations.append(
                PathLossAtom(
                    raw_path=record.raw_path,
                    raw_path_id=record.raw_path_id,
                    class_id=record.class_id,
                    validation_loss=loss,
                )
            )
            if loss < best_loss:
                best_index = index
                best_loss = loss
    if best_index < 0:
        raise RuntimeError("Exhaustive selector did not evaluate any path.")
    selected = evaluations[best_index]
    return ExhaustiveSelectionResult(
        selected_path=selected.raw_path,
        selected_raw_path_id=selected.raw_path_id,
        selected_class_id=selected.class_id,
        selected_validation_loss=selected.validation_loss,
        evaluated_paths=len(evaluations),
        evaluation_budget=evaluation_budget,
        primary=primary,
        evaluations=tuple(evaluations),
    )


def compute_recovery_atoms(
    target_path: Sequence[str],
    predicted_path: Sequence[str],
) -> RecoveryAtoms:
    """Return per-run recovery atoms without aggregating a claim decision."""

    target = _path_universe.validate_raw_path(target_path)
    predicted = _path_universe.validate_raw_path(predicted_path)
    target_record = _path_record(target)
    predicted_record = _path_record(predicted)
    return RecoveryAtoms(
        target_path=target,
        predicted_path=predicted,
        target_raw_path_id=target_record.raw_path_id,
        predicted_raw_path_id=predicted_record.raw_path_id,
        target_class_id=target_record.class_id,
        predicted_class_id=predicted_record.class_id,
        exact_match=target == predicted,
        semantic_match=target_record.class_id == predicted_record.class_id,
        raw_edit_distance=_levenshtein(target, predicted),
        canonical_edit_distance=_levenshtein(
            target_record.canonical_path,
            predicted_record.canonical_path,
        ),
    )


def compute_stability_atoms(
    paths_by_seed: Mapping[int, Sequence[str]],
) -> tuple[StabilityAtom, ...]:
    """Return every pairwise seed agreement atom; do not average them."""

    if not isinstance(paths_by_seed, Mapping) or len(paths_by_seed) < 2:
        raise ValueError("paths_by_seed must contain at least two optimization seeds.")
    normalized: dict[int, _path_universe.RawPath] = {}
    for seed, path in paths_by_seed.items():
        normalized[_validate_seed(seed)] = _path_universe.validate_raw_path(path)
    atoms: list[StabilityAtom] = []
    for left_seed, right_seed in combinations(sorted(normalized), 2):
        left = normalized[left_seed]
        right = normalized[right_seed]
        left_record = _path_record(left)
        right_record = _path_record(right)
        atoms.append(
            StabilityAtom(
                left_seed=left_seed,
                right_seed=right_seed,
                left_raw_path_id=left_record.raw_path_id,
                right_raw_path_id=right_record.raw_path_id,
                left_class_id=left_record.class_id,
                right_class_id=right_record.class_id,
                exact_path_agreement=left == right,
                semantic_path_agreement=left_record.class_id
                == right_record.class_id,
                raw_edit_distance=_levenshtein(left, right),
                canonical_edit_distance=_levenshtein(
                    left_record.canonical_path,
                    right_record.canonical_path,
                ),
            )
        )
    return tuple(atoms)


def load_cwru_fold(
    manifest: CWRUManifest,
    fold_id: str,
    *,
    read_fn: Callable[[ManifestSpecimen], Any],
    dtype: torch.dtype = torch.float32,
) -> CWRUFoldData:
    """Load one fold solely from manifest file keys and window coordinates."""

    _validate_cwru_manifest_root(manifest)
    if not isinstance(fold_id, str) or not fold_id.strip():
        raise ValueError("fold_id must be non-empty text.")
    if not callable(read_fn):
        raise TypeError("read_fn must be callable.")
    if dtype not in {torch.float32, torch.float64}:
        raise TypeError("dtype must be torch.float32 or torch.float64.")
    matches = [fold for fold in manifest.folds if fold.fold_id == fold_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one manifest fold {fold_id!r}.")
    fold = matches[0]
    _validate_fold_membership(fold)
    by_key = {item.specimen_key: item for item in manifest.specimens}
    if len(by_key) != len(manifest.specimens):
        raise ValueError("Manifest contains duplicate specimen keys.")
    referenced = set(
        (*fold.train_specimen_keys, *fold.validation_specimen_keys, *fold.test_specimen_keys)
    )
    if not referenced.issubset(by_key):
        raise ValueError("Fold references a specimen key absent from the manifest.")

    def load_split(
        split: Literal["train", "validation", "test"],
        keys: Sequence[str],
    ) -> tuple[CWRUFileBatch, ...]:
        files: list[CWRUFileBatch] = []
        for key in keys:
            specimen = by_key[key]
            coordinates = tuple(specimen.windows)
            stacked = materialize_manifest_windows(
                read_fn(specimen), specimen, dtype=dtype
            )
            files.append(
                CWRUFileBatch(
                    file_key=specimen.specimen_key,
                    file_name=specimen.file_name,
                    split=split,
                    label=specimen.label,
                    class_index=specimen.label - 1,
                    file_weight=float(specimen.file_weight),
                    windows=stacked,
                    coordinates=coordinates,
                )
            )
        return tuple(files)

    return CWRUFoldData(
        fold_id=fold.fold_id,
        manifest_root_sha256=manifest.root_sha256,
        train_files=load_split("train", fold.train_specimen_keys),
        validation_files=load_split("validation", fold.validation_specimen_keys),
        test_files=load_split("test", fold.test_specimen_keys),
    )


def train_file_macro_classifier(
    model: nn.Module,
    train_files: Sequence[CWRUFileBatch],
    validation_files: Sequence[CWRUFileBatch],
    *,
    optimization_seed: int,
    budget: TrainingBudget = TrainingBudget(),
) -> TrainingTrace:
    """Train with one equally weighted optimizer contribution per file."""

    _validate_training_budget(budget)
    train_data = _validate_file_batches(train_files, required_split="train")
    validation_data = _validate_file_batches(
        validation_files, required_split="validation"
    )
    if {item.file_key for item in train_data}.intersection(
        item.file_key for item in validation_data
    ):
        raise ValueError("Training and validation file keys must be disjoint.")
    seed = seed_all_rng(optimization_seed)
    device, dtype = _model_device_dtype(model)
    optimizer = _adamw(model, budget)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    best_loss = math.inf
    best_epoch = 0
    best_state: Optional[dict[str, Any]] = None
    bad_epochs = 0
    updates = 0
    train_history: list[float] = []
    validation_history: list[float] = []
    stop_reason: Literal["early_stopping", "max_epochs", "max_updates"] = (
        "max_epochs"
    )

    for epoch in range(1, budget.max_epochs + 1):
        model.train()
        order = torch.randperm(len(train_data), generator=generator).tolist()
        per_file_losses: list[float] = []
        for file_index in order:
            if updates >= budget.max_updates:
                stop_reason = "max_updates"
                break
            item = train_data[file_index]
            windows = item.windows.to(device=device, dtype=dtype)
            optimizer.zero_grad(set_to_none=True)
            file_loss_sum = 0.0
            for start in range(0, item.window_count, budget.batch_size):
                batch = windows[start : start + budget.batch_size]
                logits = _classification_output(model, batch)
                _validate_classification_logits(logits, int(batch.shape[0]))
                target = torch.full(
                    (int(batch.shape[0]),),
                    item.class_index,
                    dtype=torch.long,
                    device=device,
                )
                chunk_sum = F.cross_entropy(logits, target, reduction="sum")
                _require_finite_scalar(chunk_sum, "training classification loss")
                (chunk_sum / item.window_count).backward()
                file_loss_sum += float(chunk_sum.detach().item())
            _require_finite_gradients(model)
            optimizer.step()
            _require_finite_parameters(model)
            per_file_losses.append(file_loss_sum / item.window_count)
            updates += 1
        if not per_file_losses:
            break
        train_history.append(sum(per_file_losses) / len(per_file_losses))
        validation_result = evaluate_file_macro_classifier(
            model,
            validation_data,
            batch_size=budget.batch_size,
        )
        validation_loss = validation_result.macro_loss
        validation_history.append(validation_loss)
        if validation_loss < best_loss - budget.min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= budget.patience and budget.patience > 0:
            stop_reason = "early_stopping"
            break
        if updates >= budget.max_updates:
            stop_reason = "max_updates"
            break

    if best_state is None or best_epoch <= 0 or not math.isfinite(best_loss):
        raise RuntimeError("Training completed without a finite validation checkpoint.")
    model.load_state_dict(best_state, strict=True)
    return TrainingTrace(
        optimization_seed=seed,
        epochs_completed=len(train_history),
        updates_completed=updates,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        training_losses=tuple(train_history),
        validation_losses=tuple(validation_history),
        stopped_early=stop_reason == "early_stopping",
        stop_reason=stop_reason,
    )


def evaluate_file_macro_classifier(
    model: nn.Module,
    files: Sequence[CWRUFileBatch],
    *,
    batch_size: int = 64,
) -> FileMacroEvaluation:
    """Evaluate files equally; window count is never the inferential sample size."""

    size = _positive_int(batch_size, "batch_size")
    data = _validate_file_batches(files)
    split_values = {item.split for item in data}
    if len(split_values) != 1:
        raise ValueError("One file-macro evaluation may contain exactly one split.")
    device, dtype = _model_device_dtype(model, require_trainable=False)
    was_training = model.training
    predictions: list[FilePrediction] = []
    try:
        model.eval()
        with torch.no_grad():
            for item in data:
                windows = item.windows.to(device=device, dtype=dtype)
                chunks: list[torch.Tensor] = []
                loss_sum = 0.0
                for start in range(0, item.window_count, size):
                    batch = windows[start : start + size]
                    logits = _classification_output(model, batch)
                    _validate_classification_logits(logits, int(batch.shape[0]))
                    if not 0 <= item.class_index < int(logits.shape[1]):
                        raise ValueError(
                            f"File {item.file_key} class_index is outside model output."
                        )
                    target = torch.full(
                        (int(batch.shape[0]),),
                        item.class_index,
                        dtype=torch.long,
                        device=device,
                    )
                    chunk_loss = F.cross_entropy(logits, target, reduction="sum")
                    _require_finite_scalar(chunk_loss, "file evaluation loss")
                    loss_sum += float(chunk_loss.item())
                    chunks.append(logits)
                file_logits = torch.cat(chunks, dim=0)
                predicted = int(file_logits.mean(dim=0).argmax().item())
                predictions.append(
                    FilePrediction(
                        file_key=item.file_key,
                        label=item.label,
                        class_index=item.class_index,
                        predicted_class_index=predicted,
                        correct=predicted == item.class_index,
                        window_count=item.window_count,
                        mean_window_loss=loss_sum / item.window_count,
                    )
                )
    finally:
        model.train(was_training)

    macro_accuracy = sum(float(item.correct) for item in predictions) / len(
        predictions
    )
    macro_loss = sum(item.mean_window_loss for item in predictions) / len(predictions)
    return FileMacroEvaluation(
        split=next(iter(split_values)),
        macro_accuracy=macro_accuracy,
        macro_loss=macro_loss,
        independent_unit_count=len(predictions),
        total_window_count=sum(item.window_count for item in predictions),
        predictions=tuple(predictions),
    )


def select_full_discrete_path_classifier(
    train_files: Sequence[CWRUFileBatch],
    validation_files: Sequence[CWRUFileBatch],
    *,
    bookkeeping_seed: int,
    num_classes: int = 3,
    evaluation_budget: int = PRIMARY_EXHAUSTIVE_EVALUATION_BUDGET,
    primary: bool = True,
) -> DiscreteClassifierSelectionResult:
    """Fit train-only ridge candidates and select without accepting test data.

    ``bookkeeping_seed`` is recorded for orchestration but is not consumed by
    the deterministic L-BFGS fit.  This comparator is fitted once per fold, not
    repeated as if it supplied independent optimization-seed observations.
    """

    _require_frozen_sklearn_version()
    seed = _validate_seed(bookkeeping_seed)
    classes = _positive_int(num_classes, "num_classes")
    if classes < 2:
        raise ValueError("num_classes must be at least two.")
    train_data = _validate_file_batches(train_files, required_split="train")
    validation_data = _validate_file_batches(
        validation_files, required_split="validation"
    )
    if {item.file_key for item in train_data}.intersection(
        item.file_key for item in validation_data
    ):
        raise ValueError("Training and validation file keys must be disjoint.")
    channel_counts = {
        int(item.windows.shape[2]) for item in (*train_data, *validation_data)
    }
    if len(channel_counts) != 1:
        raise ValueError("All discrete-search files must share one channel count.")
    channels = next(iter(channel_counts))
    if any(item.class_index >= classes for item in (*train_data, *validation_data)):
        raise ValueError("A file class_index is outside num_classes.")

    if isinstance(evaluation_budget, bool) or not isinstance(evaluation_budget, int):
        raise TypeError("evaluation_budget must be an integer, not boolean.")
    records = _path_universe.enumerate_path_records()
    if len(records) != PRIMARY_EXHAUSTIVE_EVALUATION_BUDGET:
        raise RuntimeError("The frozen path universe must contain exactly 216 paths.")
    if not 1 <= evaluation_budget <= len(records):
        raise ValueError("evaluation_budget must be in [1, 216].")
    if not isinstance(primary, bool):
        raise TypeError("primary must be boolean.")
    if primary and evaluation_budget != PRIMARY_EXHAUSTIVE_EVALUATION_BUDGET:
        raise ValueError(
            "The primary discrete classifier search requires evaluation_budget=216."
        )

    device = train_data[0].windows.device
    dtype = train_data[0].windows.dtype
    classifier_spec = MultinomialRidgeSpec()
    evaluations: list[DiscreteClassifierCandidate] = []
    selected_model: Optional[FrozenPathClassifier] = None
    selected_key: Optional[tuple[float, float, int]] = None
    selected_index = -1
    started = time.perf_counter()
    for registry_index, record in enumerate(records[:evaluation_budget]):
        model = FrozenPathClassifier(
            record.raw_path,
            in_channels=channels,
            num_classes=classes,
        ).to(device=device, dtype=dtype)
        solver_iterations = model.fit_train_files(train_data)
        validation = evaluate_file_macro_classifier(
            model,
            validation_data,
            batch_size=classifier_spec.evaluation_batch_size,
        )
        atom = DiscreteClassifierCandidate(
            raw_path=record.raw_path,
            raw_path_id=record.raw_path_id,
            class_id=record.class_id,
            validation_macro_accuracy=validation.macro_accuracy,
            validation_macro_loss=validation.macro_loss,
            solver_iterations=solver_iterations,
        )
        evaluations.append(atom)
        selection_key = (
            -atom.validation_macro_accuracy,
            atom.validation_macro_loss,
            registry_index,
        )
        if selected_key is None or selection_key < selected_key:
            selected_key = selection_key
            selected_index = registry_index
            selected_model = model
    elapsed = time.perf_counter() - started

    if selected_model is None or selected_index < 0:
        raise RuntimeError("Discrete classifier search evaluated no candidates.")
    selected_model.eval()
    selected = evaluations[selected_index]
    compute = DiscreteClassifierCompute(
        candidate_fits=len(evaluations),
        candidate_validation_evaluations=len(evaluations),
        total_solver_iterations=sum(item.solver_iterations for item in evaluations),
        wall_time_seconds=elapsed,
    )
    return DiscreteClassifierSelectionResult(
        selected_path=selected.raw_path,
        selected_raw_path_id=selected.raw_path_id,
        selected_class_id=selected.class_id,
        selected_validation_macro_accuracy=selected.validation_macro_accuracy,
        selected_validation_macro_loss=selected.validation_macro_loss,
        selected_model=selected_model,
        bookkeeping_seed=seed,
        evaluated_paths=len(evaluations),
        evaluation_budget=evaluation_budget,
        primary=primary,
        evaluations=tuple(evaluations),
        compute=compute,
        classifier_spec=classifier_spec,
    )


def build_cwru_arms(
    *,
    in_channels: int = 2,
    num_classes: int = 3,
    initialization_seed: int = 0,
    random_dictionary_seed: int = 701,
    dropout: float = 0.1,
    maximum_relative_parameter_gap: float = PARAMETER_MATCH_TOLERANCE,
) -> tuple[ExperimentArm, ...]:
    """Build the frozen CWRU neural arms and the parameter-free search arm."""

    channels = _positive_int(in_channels, "in_channels")
    classes = _positive_int(num_classes, "num_classes")
    seed_all_rng(initialization_seed)
    _validate_seed(random_dictionary_seed)
    core_cfg = ExecutableOperatorPathConfig()
    proposed = XOANOperatorPath(
        SimpleNamespace(
            in_channels=channels,
            num_classes=classes,
            classifier_hidden_dim=66,
            dropout=dropout,
            inference_mode="discrete",
            operator_path=core_cfg,
        )
    )
    dense = OperatorCoreClassifier(
        DenseOperatorMixture1D(channels, ExecutableOperatorPathConfig()),
        in_channels=channels,
        num_classes=classes,
        classifier_hidden_dim=66,
        dropout=dropout,
    )
    random_dictionary = OperatorCoreClassifier(
        RandomDictionaryOperatorPath1D(
            channels,
            random_dictionary_seed=random_dictionary_seed,
            cfg=ExecutableOperatorPathConfig(),
        ),
        in_channels=channels,
        num_classes=classes,
        classifier_hidden_dim=66,
        dropout=dropout,
    )
    feature_attention = AttentionCNN(
        SimpleNamespace(
            input_dim=channels,
            channels=[20],
            use_attention=True,
            dropout=dropout,
            num_classes=classes,
        )
    )
    black_box = ExplainableCNN(
        SimpleNamespace(
            in_channels=channels,
            width=7,
            dropout=dropout,
            num_classes=classes,
        )
    )
    arms = (
        _neural_arm("proposed", "proposed", proposed, False),
        _neural_arm("dense_operator_mixture", "path_producing", dense, True),
        _neural_arm("random_dictionary", "negative_control", random_dictionary, False),
        _neural_arm("attention_cnn", "predictive_only", feature_attention, True),
        _neural_arm("explainable_cnn", "predictive_only", black_box, True),
        ExperimentArm(
            arm_id="discrete_search",
            role="path_producing",
            model=None,
            trainable_parameter_count=None,
            parameter_match_required=False,
        ),
    )
    validate_parameter_matched_arms(
        arms,
        maximum_relative_gap=maximum_relative_parameter_gap,
    )
    return arms


def build_dirg_arms(
    *,
    initialization_seed: int = 0,
    random_dictionary_seed: int = 701,
    maximum_relative_parameter_gap: float = PARAMETER_MATCH_TOLERANCE,
) -> tuple[ExperimentArm, ...]:
    """Build the explicit frozen DIRG six-channel, two-class arm profile.

    The architecture widths and dropout are not inferred from data and are not
    caller-configurable.  Exact trainable counts are checked in addition to the
    relative parameter-matching rule so an upstream model change fails before
    training rather than silently changing the preregistered comparison.
    """

    seed_all_rng(initialization_seed)
    _validate_seed(random_dictionary_seed)
    core_cfg = ExecutableOperatorPathConfig()
    proposed = XOANOperatorPath(
        SimpleNamespace(
            in_channels=DIRG_IN_CHANNELS,
            num_classes=DIRG_NUM_CLASSES,
            classifier_hidden_dim=DIRG_OPERATOR_CLASSIFIER_HIDDEN_DIM,
            dropout=DIRG_DROPOUT,
            inference_mode="discrete",
            operator_path=core_cfg,
        )
    )
    dense = OperatorCoreClassifier(
        DenseOperatorMixture1D(
            DIRG_IN_CHANNELS,
            ExecutableOperatorPathConfig(),
        ),
        in_channels=DIRG_IN_CHANNELS,
        num_classes=DIRG_NUM_CLASSES,
        classifier_hidden_dim=DIRG_OPERATOR_CLASSIFIER_HIDDEN_DIM,
        dropout=DIRG_DROPOUT,
    )
    random_dictionary = OperatorCoreClassifier(
        RandomDictionaryOperatorPath1D(
            DIRG_IN_CHANNELS,
            random_dictionary_seed=random_dictionary_seed,
            cfg=ExecutableOperatorPathConfig(),
        ),
        in_channels=DIRG_IN_CHANNELS,
        num_classes=DIRG_NUM_CLASSES,
        classifier_hidden_dim=DIRG_OPERATOR_CLASSIFIER_HIDDEN_DIM,
        dropout=DIRG_DROPOUT,
    )
    feature_attention = AttentionCNN(
        SimpleNamespace(
            input_dim=DIRG_IN_CHANNELS,
            channels=list(DIRG_ATTENTION_CHANNELS),
            use_attention=True,
            dropout=DIRG_DROPOUT,
            num_classes=DIRG_NUM_CLASSES,
        )
    )
    black_box = ExplainableCNN(
        SimpleNamespace(
            in_channels=DIRG_IN_CHANNELS,
            width=DIRG_EXPLAINABLE_CNN_WIDTH,
            dropout=DIRG_DROPOUT,
            num_classes=DIRG_NUM_CLASSES,
        )
    )
    arms = (
        _neural_arm("proposed", "proposed", proposed, False),
        _neural_arm("dense_operator_mixture", "path_producing", dense, True),
        _neural_arm(
            "random_dictionary",
            "negative_control",
            random_dictionary,
            False,
        ),
        _neural_arm("attention_cnn", "predictive_only", feature_attention, True),
        _neural_arm("explainable_cnn", "predictive_only", black_box, True),
        ExperimentArm(
            arm_id="discrete_search",
            role="path_producing",
            model=None,
            trainable_parameter_count=None,
            parameter_match_required=False,
        ),
    )
    validate_dirg_arms(
        arms,
        maximum_relative_gap=maximum_relative_parameter_gap,
    )
    return arms


def validate_dirg_arms(
    arms: Sequence[ExperimentArm],
    *,
    maximum_relative_gap: float = PARAMETER_MATCH_TOLERANCE,
) -> None:
    """Validate exact DIRG architecture counts and the shared 5% guard."""

    values = tuple(arms)
    expected = dict(DIRG_EXPECTED_TRAINABLE_PARAMETER_COUNTS)
    observed_ids = tuple(item.arm_id for item in values)
    expected_ids = (*expected, "discrete_search")
    if observed_ids != expected_ids:
        raise RuntimeError(
            "DIRG arms must retain the frozen order and exact arm ID set."
        )
    for arm_id, expected_count in expected.items():
        arm = next(item for item in values if item.arm_id == arm_id)
        if arm.model is None:
            raise RuntimeError(f"Frozen DIRG neural arm {arm_id} has no model.")
        recomputed = count_trainable_parameters(arm.model)
        if (
            arm.trainable_parameter_count != expected_count
            or recomputed != expected_count
        ):
            raise RuntimeError(
                f"DIRG arm {arm_id} frozen trainable parameter count is "
                f"{expected_count}, declared {arm.trainable_parameter_count}, "
                f"recomputed {recomputed}."
            )
    search = values[-1]
    if search.model is not None or search.trainable_parameter_count is not None:
        raise RuntimeError("DIRG exhaustive search must remain parameter-free.")
    validate_parameter_matched_arms(
        values,
        maximum_relative_gap=maximum_relative_gap,
    )


def validate_parameter_matched_arms(
    arms: Sequence[ExperimentArm],
    *,
    maximum_relative_gap: float = PARAMETER_MATCH_TOLERANCE,
) -> None:
    """Fail before training if any designated neural comparator exceeds 5%."""

    if isinstance(maximum_relative_gap, bool) or not isinstance(
        maximum_relative_gap, (int, float)
    ):
        raise TypeError("maximum_relative_gap must be a real number.")
    gap_limit = float(maximum_relative_gap)
    if (
        not math.isfinite(gap_limit)
        or not 0.0 <= gap_limit <= PARAMETER_MATCH_TOLERANCE
    ):
        raise ValueError(
            "maximum_relative_gap must be finite and no larger than the frozen 5% limit."
        )
    values = tuple(arms)
    ids = [item.arm_id for item in values]
    if len(ids) != len(set(ids)):
        raise ValueError("Experiment arm IDs must be unique.")
    proposed_matches = [item for item in values if item.arm_id == "proposed"]
    if len(proposed_matches) != 1:
        raise ValueError("Exactly one proposed arm is required.")
    reference = proposed_matches[0].trainable_parameter_count
    if reference is None or reference <= 0:
        raise ValueError("Proposed arm must declare a positive parameter count.")
    required = [item for item in values if item.parameter_match_required]
    if len(required) != 3:
        raise ValueError("Exactly three neural comparator arms must be parameter matched.")
    for arm in required:
        if arm.model is None or arm.trainable_parameter_count is None:
            raise ValueError(f"Parameter-matched arm {arm.arm_id} must be neural.")
        relative_gap = abs(arm.trainable_parameter_count - reference) / reference
        if relative_gap > gap_limit:
            raise ValueError(
                f"Arm {arm.arm_id} parameter gap {relative_gap:.6f} exceeds "
                f"the frozen limit {gap_limit:.6f}."
            )
    for arm_id in ("dense_operator_mixture", "random_dictionary"):
        matches = [item for item in values if item.arm_id == arm_id]
        if len(matches) != 1 or matches[0].trainable_parameter_count != reference:
            raise ValueError(
                "Proposed, dense_operator_mixture, and random_dictionary arms "
                "must have exactly equal trainable parameter counts."
            )


def count_trainable_parameters(model: nn.Module) -> int:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _neural_arm(
    arm_id: str,
    role: str,
    model: nn.Module,
    parameter_match_required: bool,
) -> ExperimentArm:
    return ExperimentArm(
        arm_id=arm_id,
        role=role,
        model=model,
        trainable_parameter_count=count_trainable_parameters(model),
        parameter_match_required=parameter_match_required,
    )


def _require_frozen_sklearn_version() -> None:
    if sklearn.__version__ != FROZEN_RIDGE_SKLEARN_VERSION:
        raise RuntimeError(
            "Full-216 ridge requires scikit-learn "
            f"{FROZEN_RIDGE_SKLEARN_VERSION}, observed {sklearn.__version__}."
        )


def _validate_training_budget(budget: Any) -> TrainingBudget:
    if not isinstance(budget, TrainingBudget):
        raise TypeError("budget must be a TrainingBudget.")
    return budget


def _validate_seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer, not boolean.")
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32).")
    return seed


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _validate_blc(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim != 3 or any(int(size) <= 0 for size in value.shape):
        raise ValueError(f"{name} must have non-empty (batch,length,channels) shape.")
    if int(value.shape[1]) < 2:
        raise ValueError(f"{name} length must be at least two.")
    if not torch.is_floating_point(value) or torch.is_complex(value):
        raise TypeError(f"{name} must be a real floating tensor.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values.")
    return value


def _coerce_blc(
    value: Any,
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return _validate_blc(value, name).detach().to(device=device, dtype=dtype)


def _model_device_dtype(
    model: Any,
    *,
    require_trainable: bool = True,
) -> tuple[torch.device, torch.dtype]:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not isinstance(require_trainable, bool):
        raise TypeError("require_trainable must be boolean.")
    parameters = tuple(
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad or not require_trainable
    )
    state_tensors = parameters
    if not state_tensors and not require_trainable:
        state_tensors = tuple(model.buffers())
    if not state_tensors:
        qualifier = "trainable " if require_trainable else ""
        raise ValueError(f"model must have at least one {qualifier}parameter or buffer.")
    device = state_tensors[0].device
    dtype = state_tensors[0].dtype
    if not torch.is_floating_point(state_tensors[0]) or torch.is_complex(state_tensors[0]):
        raise TypeError("model state must use a real floating dtype.")
    if any(value.device != device or value.dtype != dtype for value in state_tensors):
        raise ValueError("model state tensors must share one device and dtype.")
    return device, dtype


def _adamw(model: nn.Module, budget: TrainingBudget) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(budget.learning_rate),
        weight_decay=float(budget.weight_decay),
    )


def _reconstruction_output(core: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    value = core(inputs)
    output = value[0] if isinstance(value, tuple) else value
    if not isinstance(output, torch.Tensor):
        raise TypeError("Reconstruction core must return a tensor or tensor-first tuple.")
    return output


def _classification_output(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    output = model(inputs)
    if not isinstance(output, torch.Tensor):
        raise TypeError("Classifier must return a logits tensor.")
    return output


def _validate_prediction(
    prediction: torch.Tensor,
    target: torch.Tensor,
    label: str,
) -> None:
    if prediction.shape != target.shape:
        raise ValueError(f"{label} shape does not match its oracle target.")
    if prediction.device != target.device or prediction.dtype != target.dtype:
        raise ValueError(f"{label} dtype/device does not match its oracle target.")
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError(f"{label} produced non-finite values.")


def _validate_classification_logits(logits: torch.Tensor, batch_size: int) -> None:
    if logits.ndim != 2 or int(logits.shape[0]) != batch_size or int(logits.shape[1]) < 2:
        raise ValueError("Classifier logits must have shape (batch,num_classes>=2).")
    if not torch.is_floating_point(logits) or torch.is_complex(logits):
        raise TypeError("Classifier logits must be real floating values.")
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Classifier produced non-finite logits.")


def _reconstruction_loss(
    core: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
) -> float:
    was_training = core.training
    loss_sum = 0.0
    sample_count = 0
    try:
        core.eval()
        with torch.no_grad():
            for start in range(0, int(inputs.shape[0]), batch_size):
                batch_x = inputs[start : start + batch_size]
                batch_y = targets[start : start + batch_size]
                prediction = _reconstruction_output(core, batch_x)
                _validate_prediction(prediction, batch_y, "validation reconstruction")
                batch_loss = F.mse_loss(prediction, batch_y, reduction="sum")
                _require_finite_scalar(batch_loss, "validation reconstruction loss")
                loss_sum += float(batch_loss.item())
                sample_count += int(batch_y.numel())
    finally:
        core.train(was_training)
    if sample_count <= 0:
        raise RuntimeError("Reconstruction evaluation saw no target values.")
    return loss_sum / sample_count


def _require_finite_scalar(value: torch.Tensor, label: str) -> None:
    if value.ndim != 0 or not bool(torch.isfinite(value)):
        raise FloatingPointError(f"{label} must be a finite scalar.")


def _require_finite_gradients(model: nn.Module) -> None:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("Training produced no gradients.")
    if not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise FloatingPointError("Training produced non-finite gradients.")


def _require_finite_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        if torch.is_floating_point(parameter) and not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError("Optimizer produced non-finite parameters.")


def _path_record(path: _path_universe.RawPath) -> _path_universe.PathRecord:
    for record in _path_universe.enumerate_path_records():
        if record.raw_path == path:
            return record
    raise RuntimeError("Validated path is absent from the frozen universe.")


def _levenshtein(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _validate_cwru_manifest_root(manifest: Any) -> CWRUManifest:
    if not isinstance(manifest, CWRUManifest):
        raise TypeError("manifest must be a CWRUManifest.")
    payload = json.dumps(
        manifest.payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    expected = hashlib.sha256(payload).hexdigest()
    if manifest.root_sha256 != expected:
        raise ValueError("CWRU manifest root hash is invalid.")
    return manifest


def _validate_fold_membership(fold: CWRUFold) -> None:
    groups = (
        tuple(fold.train_specimen_keys),
        tuple(fold.validation_specimen_keys),
        tuple(fold.test_specimen_keys),
        tuple(fold.excluded_specimen_keys),
    )
    if any(len(group) != len(set(group)) for group in groups):
        raise ValueError("Fold contains duplicate file keys within a split.")
    for left, right in combinations(groups, 2):
        if set(left).intersection(right):
            raise ValueError("Fold file keys overlap across split boundaries.")
    if not all(groups[:3]):
        raise ValueError("Train, validation, and test file-key sets must be non-empty.")
    if fold.evaluation_unit != "file" or fold.weighting != "equal_file":
        raise ValueError("CWRU fold must use equal-file evaluation.")


def _validate_file_batches(
    files: Sequence[CWRUFileBatch],
    *,
    required_split: Optional[Literal["train", "validation", "test"]] = None,
) -> tuple[CWRUFileBatch, ...]:
    if isinstance(files, (str, bytes)) or not isinstance(files, Sequence):
        raise TypeError("files must be a sequence of CWRUFileBatch objects.")
    values = tuple(files)
    if not values or any(not isinstance(item, CWRUFileBatch) for item in values):
        raise ValueError("files must contain at least one CWRUFileBatch.")
    if len({item.file_key for item in values}) != len(values):
        raise ValueError("File-macro inputs must have unique file keys.")
    for item in values:
        if required_split is not None and item.split != required_split:
            raise ValueError(
                f"Expected only {required_split} files, observed split {item.split}."
            )
        if item.windows.ndim != 3 or int(item.windows.shape[0]) <= 0:
            raise ValueError(f"File {item.file_key} windows must use non-empty WLC shape.")
        if not torch.is_floating_point(item.windows) or torch.is_complex(item.windows):
            raise TypeError(f"File {item.file_key} windows must be real floating tensors.")
        if not bool(torch.isfinite(item.windows).all()):
            raise ValueError(f"File {item.file_key} windows contain non-finite values.")
        if isinstance(item.class_index, bool) or not isinstance(item.class_index, int):
            raise TypeError(f"File {item.file_key} class_index must be an integer.")
        if item.class_index < 0:
            raise ValueError(f"File {item.file_key} class_index must be non-negative.")
        if not math.isfinite(float(item.file_weight)) or float(item.file_weight) != 1.0:
            raise ValueError("File-macro protocol requires every file_weight to equal 1.0.")
    return values


__all__ = [
    "CWRUFileBatch",
    "CWRUFoldData",
    "DIRG_ATTENTION_CHANNELS",
    "DIRG_DROPOUT",
    "DIRG_EXPECTED_TRAINABLE_PARAMETER_COUNTS",
    "DIRG_EXPLAINABLE_CNN_WIDTH",
    "DIRG_IN_CHANNELS",
    "DIRG_NUM_CLASSES",
    "DIRG_OPERATOR_CLASSIFIER_HIDDEN_DIM",
    "DiscreteClassifierCandidate",
    "DiscreteClassifierCompute",
    "DiscreteClassifierSelectionResult",
    "ExperimentArm",
    "ExhaustiveSelectionResult",
    "FROZEN_RIDGE_SKLEARN_VERSION",
    "FileMacroEvaluation",
    "FilePrediction",
    "FrozenPathClassifier",
    "MultinomialRidgeSpec",
    "OperatorCoreClassifier",
    "PARAMETER_MATCH_TOLERANCE",
    "PRIMARY_EXHAUSTIVE_EVALUATION_BUDGET",
    "PathLossAtom",
    "ReconstructionEvaluation",
    "RecoveryAtoms",
    "StabilityAtom",
    "TrainingBudget",
    "TrainingTrace",
    "build_cwru_arms",
    "build_dirg_arms",
    "compute_recovery_atoms",
    "compute_stability_atoms",
    "count_trainable_parameters",
    "evaluate_file_macro_classifier",
    "evaluate_synthetic_reconstruction",
    "independent_oracle_target",
    "load_cwru_fold",
    "seed_all_rng",
    "select_exhaustive_oracle_path",
    "select_full_discrete_path_classifier",
    "standardize_window",
    "train_file_macro_classifier",
    "train_synthetic_reconstruction",
    "validate_dirg_arms",
    "validate_parameter_matched_arms",
]
