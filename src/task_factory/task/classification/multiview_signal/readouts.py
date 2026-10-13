"""Frozen-logit offset readouts and source-only baseline-retaining decisions.

Numerical objective, optimizer, Brier segment and empirical guard adapted from
liq22/phm-agent-benchmark at commit 76a195133c23149147d204fd2883eebcc2e9289a,
signal_programs.py, operator_gain.py and operator_study.py. No runtime imports
from Benchmark are used. Single-view fits leave the original four fits intact.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from .features import view_indices
from .metrics import equal_unit_weights, group_means, validate_labels, validate_probabilities


def _logits(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or len(value) == 0 or value.shape[1] < 2 or not np.isfinite(value).all():
        raise ValueError("Expected finite nonempty logits with shape [items,classes], classes >= 2")
    return value


def _probabilities(base: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base, candidate = validate_probabilities(base), validate_probabilities(candidate)
    if base.shape != candidate.shape:
        raise ValueError("Aligned base and candidate probabilities are required")
    return base, candidate


def project_brier(base: np.ndarray, candidate: np.ndarray, y: np.ndarray,
                  units: np.ndarray) -> dict[str, Any]:
    """Minimize equal-unit grow Brier on the closed [base,candidate] segment."""
    base, candidate = _probabilities(base, candidate)
    y = validate_labels(y, len(base), base.shape[1])
    units = np.asarray(units)
    if units.shape != y.shape:
        raise ValueError("One source-grow unit per prediction is required")
    weights = equal_unit_weights(units)
    direction = candidate - base
    a = float(weights @ np.sum((np.eye(base.shape[1])[y] - base) * direction, axis=1))
    v = float(weights @ np.sum(direction ** 2, axis=1))
    strength = float(np.clip(a / v, 0, 1)) if v > 0 else 0.0
    gain = 2 * strength * a - strength ** 2 * v
    return dict(strength=strength, alignment=a, displacement=v,
                empirical_gain=gain, units=int(len(np.unique(units))),
                guarantee_scope="empirical_grow_brier_only",
                mechanism="two_predictor_convex_projection")


def source_guard(base: np.ndarray, candidate: np.ndarray, y: np.ndarray,
                 units: np.ndarray) -> dict[str, Any]:
    """Freeze one candidate or recover the baseline using only source-select units.

    Both empirical Brier excess and classification-error excess must be <= 0.
    This is not a population guarantee under an unseen-domain shift.
    """
    base, candidate = _probabilities(base, candidate)
    y = validate_labels(y, len(base), base.shape[1])
    units = np.asarray(units)
    if units.shape != y.shape:
        raise ValueError("One source-select unit per prediction is required")
    base_loss = np.sum((base - np.eye(base.shape[1])[y]) ** 2, axis=1)
    candidate_loss = np.sum((candidate - np.eye(base.shape[1])[y]) ** 2, axis=1)
    difference = candidate_loss - base_loss
    error = (candidate.argmax(1) != y).astype(float) - (base.argmax(1) != y).astype(float)
    brier_units, error_units = group_means(difference, units), group_means(error, units)
    brier_excess, error_excess = float(brier_units.mean()), float(error_units.mean())
    accepted = brier_excess <= 0.0 and error_excess <= 0.0
    return dict(accepted=bool(accepted), mode="empirical",
                independent_units=len(brier_units), brier_excess=brier_excess,
                error_excess=error_excess, brier_upper=brier_excess, error_upper=error_excess,
                max_brier_excess=0.0, max_error_excess=0.0,
                choice="candidate" if accepted else "baseline",
                guarantee_scope="empirical_guard_only_no_population_guarantee")


def baseline_quality(probability: np.ndarray, y: np.ndarray, units: np.ndarray,
                     floor: float = 0.8) -> dict[str, Any]:
    probability = validate_probabilities(probability)
    y = validate_labels(y, len(probability), probability.shape[1])
    if not np.isfinite(floor) or not 0 <= floor <= 1 or np.asarray(units).shape != y.shape:
        raise ValueError("Invalid quality floor or source-tune unit coordinates")
    accuracy = float(group_means((probability.argmax(1) == y).astype(float), units).mean())
    return dict(source_unit_accuracy=accuracy, requested_floor=floor,
                qualified=bool(accuracy >= floor), qualification_role="tune",
                target_accuracy_guaranteed=False)


@dataclass
class OffsetReadout:
    """Convex multinomial fit with a frozen-logit offset and named additive terms.

    Normalization uses the supplied design-grow units only. The baseline is exactly the zero-coefficient
    member. Intercept-only and matched native-feature readouts are separate controls.
    """
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: np.ndarray
    feature_names: list[str]
    fit_info: dict[str, Any]

    @classmethod
    def fit(cls, features: np.ndarray, offset: np.ndarray, y: np.ndarray,
            units: np.ndarray, names: list[str], ridge: float = 0.01,
            maxiter: int = 1000) -> 'OffsetReadout':
        features, offset = np.asarray(features, float), np.asarray(offset, float)
        y, units = np.asarray(y), np.asarray(units)
        if (features.ndim != 2 or offset.ndim != 2 or offset.shape[1] < 2 or len(features) != len(offset)
                or y.shape != (len(offset),) or units.shape != y.shape
                or not np.isfinite(features).all() or not np.isfinite(offset).all()
                or y.dtype.kind not in 'iu' or len(y) == 0
                or y.min() < 0 or y.max() >= offset.shape[1]
                or features.shape[1] != len(names) or len(set(names)) != len(names)
                or not np.isfinite(ridge) or ridge < 0 or maxiter <= 0):
            raise ValueError('Invalid aligned source features, logits, labels or readout settings')
        validate_labels(y, len(offset), offset.shape[1])
        weights = equal_unit_weights(units)
        mean = weights @ features
        scale = np.maximum(np.sqrt(weights @ ((features - mean) ** 2)), 1e-8)
        z = (features - mean) / scale
        design = np.column_stack((z, np.ones(len(z))))
        classes = offset.shape[1]
        onehot = np.eye(classes)[y]

        def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
            matrix = flat.reshape(design.shape[1], classes)
            logits = offset + design @ matrix
            losses = logsumexp(logits, axis=1) - logits[np.arange(len(y)), y]
            gradient = design.T @ (weights[:, None] * (softmax(logits, axis=1) - onehot))
            gradient += ridge * matrix
            return (float(weights @ losses + ridge / 2 * np.sum(matrix ** 2)), gradient.ravel())

        initial = np.zeros(design.shape[1] * classes)
        result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'ftol': 1e-11, 'gtol': 1e-7})
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f'Offset-readout optimization failed: {result.message}')
        if result.fun > objective(initial)[0] + 1e-9:
            raise RuntimeError('Source objective is worse than the retained zero correction')
        matrix = result.x.reshape(design.shape[1], classes)
        return cls(mean, scale, matrix[:-1], matrix[-1], list(names),
                   dict(objective='equal_unit_offset_multinomial_nll_plus_l2', ridge=ridge,
                        maxiter=maxiter, iterations=int(result.nit),
                        final_objective=float(result.fun), baseline_objective=objective(initial)[0],
                        independent_units=int(len(np.unique(units))), fitting_role='design_grow'))

    def contributions(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, float)
        if features.ndim != 2 or features.shape[1] != len(self.mean) or not np.isfinite(features).all():
            raise ValueError('Feature coordinates differ from the frozen readout')
        return ((features - self.mean) / self.scale)[:, :, None] * self.coefficient[None]

    def logits(self, features: np.ndarray, offset: np.ndarray) -> np.ndarray:
        terms = self.contributions(features)
        offset = np.asarray(offset, float)
        if offset.shape != (len(terms), len(self.intercept)) or not np.isfinite(offset).all():
            raise ValueError('Frozen offset coordinates differ from the readout')
        return offset + terms.sum(axis=1) + self.intercept

    def predict(self, features: np.ndarray, offset: np.ndarray) -> np.ndarray:
        return softmax(self.logits(features, offset), axis=1)

    def to_dict(self) -> dict[str, Any]:
        return dict(mean=self.mean.tolist(), scale=self.scale.tolist(),
                    coefficient=self.coefficient.tolist(), intercept=self.intercept.tolist(),
                    feature_names=self.feature_names, fit_info=self.fit_info)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> 'OffsetReadout':
        intercept = np.asarray(value['intercept'], float)
        coefficient = np.asarray(value['coefficient'], float)
        # The zero-feature intercept control serializes its matrix as [].
        if len(value['mean']) == 0 and coefficient.shape == (0,):
            coefficient = coefficient.reshape(0, len(intercept))
        mean, scale = np.asarray(value['mean'], float), np.asarray(value['scale'], float)
        names = list(value['feature_names'])
        if (mean.ndim != 1 or scale.shape != mean.shape or intercept.ndim != 1
                or len(intercept) < 2 or coefficient.shape != (len(mean), len(intercept))
                or len(names) != len(mean) or len(set(names)) != len(names)
                or not all(np.isfinite(a).all() for a in (mean, scale, coefficient, intercept))
                or np.any(scale <= 0)):
            raise ValueError('Invalid saved offset-readout coordinates or nonfinite state')
        return cls(mean, scale, coefficient, intercept, names, dict(value['fit_info']))


_ORIGINAL_READOUTS = ("intercept", "native_matched", "native_full", "signal_program")
_SINGLE_READOUTS = ("signal_transient", "signal_band", "signal_modulation", "signal_recurrence")


def _feature_matrix(value: np.ndarray, name: str, rows: int | None = None) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if (value.ndim != 2 or len(value) == 0 or not np.isfinite(value).all()
            or (rows is not None and len(value) != rows)):
        raise ValueError(f"Invalid aligned finite {name} features")
    return value


def _coordinate_views(names: list[str]) -> dict[str, np.ndarray]:
    if not names or len(names) % 42 or len(set(names)) != len(names):
        raise ValueError("Expected distinct ordered P19 signal feature names: 42 per channel")
    indices = view_indices(len(names) // 42)
    for channel in range(len(names) // 42):
        start = channel * 42
        if names[start:start + 2] != [f"ch{channel}/raw/kurtosis", f"ch{channel}/raw/crest"]:
            raise ValueError("P19 raw signal feature coordinates are out of order")
        for band in range(4):
            begin = start + 2 + 10 * band
            block = names[begin:begin + 10]
            prefix = block[0].rsplit("/", 1)[0]
            suffixes = ("power_fraction", "kurtosis", "envelope_modulation", "teager_signed_ratio")
            if (not prefix.startswith(f"ch{channel}/band{band}[")
                    or block[:4] != [f"{prefix}/{suffix}" for suffix in suffixes]
                    or not all(name.startswith(prefix + "/acf_lag") for name in block[4:7])
                    or not all(name.startswith(prefix + "/envelope_power[") for name in block[7:])):
                raise ValueError(f"P19 channel {channel}, band {band} feature coordinates are out of order")
    return indices


@dataclass
class ReadoutState:
    """Source-frozen readouts, native QR map, grow strengths and select decisions."""

    native_projection: np.ndarray
    states: dict[str, dict[str, Any]]
    feature_names: list[str]

    def _blocks(self, features: np.ndarray, native: np.ndarray, rows: int) -> dict[str, np.ndarray]:
        features = _feature_matrix(features, "signal", rows)
        native = _feature_matrix(native, "native", rows)
        if (features.shape[1] != len(self.feature_names)
                or native.shape[1] != self.native_projection.shape[0]):
            raise ValueError("Signal/native coordinates differ from the source-frozen state")
        blocks = dict(intercept=np.empty((rows, 0)), native_matched=native @ self.native_projection,
                      native_full=native, signal_program=features)
        views = _coordinate_views(self.feature_names)
        for view, columns in views.items():
            blocks["signal_" + view] = features[:, columns]
        return blocks

    def predict(self, features: np.ndarray, native: np.ndarray, logits: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate frozen decisions. No labels, units or adaptation enter prediction."""
        offset = _logits(logits)
        base = softmax(offset, axis=1)
        blocks = self._blocks(features, native, len(offset))
        predictions = {"TSPN": base}
        for name, saved in self.states.items():
            readout = OffsetReadout.from_dict(saved["readout"])
            correction = readout.predict(blocks[name], offset)
            strength = float(saved["projection"]["strength"])
            mixture = (1 - strength) * base + strength * correction
            predictions[name + "/raw"] = correction
            predictions[name + "/projected"] = mixture
            predictions[name + "/guarded"] = mixture if saved["guard"]["accepted"] else base
        return predictions

    def to_dict(self) -> dict[str, Any]:
        return dict(native_projection=self.native_projection.tolist(), states=deepcopy(self.states),
                    feature_coordinates=list(self.feature_names),
                    primary="signal_program/guarded",
                    unguarded_mechanism="signal_program/projected",
                    guard_mode="empirical", source_frozen=True)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReadoutState":
        projection = np.asarray(value["native_projection"], dtype=np.float64)
        names = list(value["feature_coordinates"])
        _coordinate_views(names)
        states = dict(value["states"])
        expected = _ORIGINAL_READOUTS + _SINGLE_READOUTS if len(states) == 8 else _ORIGINAL_READOUTS
        if set(states) != set(expected):
            raise ValueError("Saved state must contain the four original readouts and, if included, all four single views")
        if (projection.ndim != 2 or projection.shape[1] != len(names)
                or projection.shape[0] < len(names) or not np.isfinite(projection).all()):
            raise ValueError("Invalid saved native QR feature coordinates")
        # JSON key sorting must not alter the published prediction order.
        ordered = {name: states[name] for name in expected}
        views = _coordinate_views(names)
        expected_names = {"intercept": [],
                          "native_matched": [f"native_projection/{j}" for j in range(len(names))],
                          "native_full": [f"native_feature/{j}" for j in range(projection.shape[0])],
                          "signal_program": names}
        expected_names.update({"signal_" + view: [names[int(j)] for j in columns]
                               for view, columns in views.items()})
        classes = None
        for name, saved in ordered.items():
            readout = OffsetReadout.from_dict(saved["readout"])
            if classes is None:
                classes = len(readout.intercept)
            strength = saved["projection"]["strength"]
            guard = saved["guard"]
            if (readout.feature_names != expected_names[name]
                    or len(readout.intercept) != classes or not np.isfinite(strength)
                    or not 0 <= strength <= 1 or type(guard["accepted"]) is not bool
                    or guard["mode"] != "empirical"):
                raise ValueError("Invalid saved projection, class coordinates or empirical guard")
        return cls(projection, ordered, names)


def fit_readouts(*, grow_features: np.ndarray, grow_native: np.ndarray, grow_logits: np.ndarray,
                 grow_labels: np.ndarray, grow_units: np.ndarray,
                 select_features: np.ndarray, select_native: np.ndarray, select_logits: np.ndarray,
                 select_labels: np.ndarray, select_units: np.ndarray,
                 feature_names: list[str], ridge: float = 0.01, maxiter: int = 1000,
                 include_single_views: bool = True) -> ReadoutState:
    """Fit and freeze using disjoint source grow/select inputs; no target argument.

    Backbone logits must already use the source-tune temperature. Grow alone fits
    normalization, penalized coefficients and segment strength; select chooses
    whether to retain that frozen candidate. No refit follows selection.
    """
    grow_logits, select_logits = _logits(grow_logits), _logits(select_logits)
    if grow_logits.shape[1] != select_logits.shape[1]:
        raise ValueError("Grow/select fixed class orders differ")
    grow_labels = validate_labels(grow_labels, len(grow_logits), grow_logits.shape[1])
    select_labels = validate_labels(select_labels, len(select_logits), select_logits.shape[1])
    grow_units, select_units = np.asarray(grow_units), np.asarray(select_units)
    if grow_units.shape != grow_labels.shape or select_units.shape != select_labels.shape:
        raise ValueError("Grow/select units must align with source labels")
    equal_unit_weights(grow_units)
    equal_unit_weights(select_units)
    if np.intersect1d(grow_units, select_units).size:
        raise ValueError("Physical units cross source-grow/source-select roles")
    if set(grow_labels) != set(range(grow_logits.shape[1])):
        raise ValueError("Source-grow must cover the fixed class set")
    grow_features = _feature_matrix(grow_features, "source-grow signal", len(grow_logits))
    select_features = _feature_matrix(select_features, "source-select signal", len(select_logits))
    grow_native = _feature_matrix(grow_native, "source-grow native", len(grow_logits))
    select_native = _feature_matrix(select_native, "source-select native", len(select_logits))
    names = list(feature_names)
    views = _coordinate_views(names)
    if (grow_features.shape[1] != len(names) or select_features.shape[1] != len(names)
            or grow_native.shape[1] != select_native.shape[1] or grow_native.shape[1] < len(names)):
        raise ValueError("Native matched control requires the declared signal dimension; never pad coordinates")
    if type(include_single_views) is not bool:
        raise ValueError("include_single_views must be boolean")
    projection, _ = np.linalg.qr(
        np.random.default_rng(11).normal(size=(grow_native.shape[1], len(names))), mode="reduced")
    source = (grow_features, select_features)
    native = (grow_native, select_native)
    blocks = {"intercept": (np.empty((len(grow_logits), 0)), np.empty((len(select_logits), 0))),
              "native_matched": tuple(values @ projection for values in native),
              "native_full": native, "signal_program": source}
    block_names = {"intercept": [], "native_matched": [f"native_projection/{j}" for j in range(len(names))],
                   "native_full": [f"native_feature/{j}" for j in range(grow_native.shape[1])],
                   "signal_program": names}
    if include_single_views:
        for view, columns in views.items():
            key = "signal_" + view
            blocks[key] = tuple(values[:, columns] for values in source)
            block_names[key] = [names[int(column)] for column in columns]
    grow_base, select_base = softmax(grow_logits, axis=1), softmax(select_logits, axis=1)
    states: dict[str, dict[str, Any]] = {}
    for name, (grow_block, select_block) in blocks.items():
        readout = OffsetReadout.fit(grow_block, grow_logits, grow_labels, grow_units,
                                   block_names[name], ridge=ridge, maxiter=maxiter)
        grow_prediction = readout.predict(grow_block, grow_logits)
        projected = project_brier(grow_base, grow_prediction, grow_labels, grow_units)
        select_prediction = readout.predict(select_block, select_logits)
        strength = projected["strength"]
        select_mixture = (1 - strength) * select_base + strength * select_prediction
        guard = source_guard(select_base, select_mixture, select_labels, select_units)
        # These are empirical decisions; adding single views does not alter the
        # original four decision rules or fits through a family-size correction.
        states[name] = dict(readout=readout.to_dict(), projection=projected, guard=guard)
    return ReadoutState(projection, states, names)
