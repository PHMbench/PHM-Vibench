"""P06 order-symbol prototype task and exact stored-affine certificates.

The certificate concerns supplied symbol vectors and the saved binary-float
coefficients, interpreted as exact rational numbers. It does not certify FFT
arithmetic, physical transformation validity, or acquisition pairing.

This non-gradient task fits repeat-calibrated predicates and class prototypes.
No trainer registry or Lightning dependency is needed. Paper entrypoints import
this module; there is no separate paper implementation of the method.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

import numpy as np

ORDERS = np.array([3.2, 4.8, 6.4])


def exact(v):
    return Fraction(float(v))


class Head:
    def __init__(self, w, b):
        w, b = np.asarray(w, float), np.asarray(b, float)
        if w.ndim != 2 or b.shape != (len(w),) or len(w) < 2:
            raise ValueError('Expected W[C,K] and b[C], C >= 2')
        if not np.isfinite(w).all() or not np.isfinite(b).all():
            raise ValueError('Nonfinite affine coefficients')
        self.w = [list(map(exact, row)) for row in w]
        self.b = list(map(exact, b))
        self.k = w.shape[1]
        self.l2 = [[sum((u-v)**2 for u, v in zip(a, c))
                    for c in self.w] for a in self.w]

    def scores(self, z):
        if len(z) != self.k or not np.isfinite(z).all():
            raise ValueError('Invalid symbol vector')
        z = list(map(exact, z))
        return [sum(a*v for a, v in zip(w, z))+b
                for w, b in zip(self.w, self.b)]

    def decision(self, z):
        scores = self.scores(z)
        winners = [i for i, s in enumerate(scores) if s == max(scores)]
        return winners[0] if len(winners) == 1 else -1

    def check(self, reference, actual):
        y, ya = self.decision(reference), self.decision(actual)
        d2 = sum((exact(a)-exact(b))**2 for a, b in zip(reference, actual))
        if y < 0:
            return dict(expected=-1, prediction=ya, certified=0, changed=None,
                        delta=math.sqrt(float(d2)), margin=0., rho=None)
        scores = self.scores(reference)
        m = min(scores[y]-s for c, s in enumerate(scores) if c != y)
        l2 = max(self.l2[y])
        # m is positive. Exact squared comparison avoids rounded square roots.
        certified = m*m > l2*d2
        return dict(expected=y, prediction=ya, certified=int(certified),
                    changed=int(y != ya), delta=math.sqrt(float(d2)),
                    margin=float(m), rho=math.sqrt(float(l2*d2/(m*m))))


def features(x, speed, fs, orders=ORDERS):
    x = np.asarray(x, float)
    speed = np.broadcast_to(np.asarray(speed, float), (len(x),))
    if x.ndim != 2 or not np.isfinite(x).all() or not np.isfinite(speed).all():
        raise ValueError('Expected finite X[N,L] and speed[N]')
    if np.any(speed <= 0) or not np.isfinite(fs) or fs <= 0:
        raise ValueError('Positive finite speed and sampling rate required')
    # For constant speed, temporal DFT frequency / shaft speed is the order DFT.
    order = np.fft.rfftfreq(x.shape[1], 1/fs)[None, :]/speed[:, None]
    power = abs(np.fft.rfft(x*np.hanning(x.shape[1]), axis=1))**2
    total = (power*((order >= .5) & (order <= 8))).sum(1)
    if np.any(total <= 0):
        raise ValueError('Zero energy in the declared normalization band')
    return np.stack([(power*(abs(order-o) <= .12)).sum(1)/total
                     for o in orders], axis=1)


def calibrate(e, data):
    train = data['split'] == 'train'
    y = data['y']
    if set(y[train]) != set(range(e.shape[1]+1)):
        raise ValueError('Classes must be 0=healthy and 1..K=declared fault orders')
    theta = np.array([.5*(np.median(e[train & (y == k+1), k])+
                             np.median(e[train & (y != k+1), k])) for k in range(e.shape[1])])
    residuals = []
    for unit in np.unique(data['unit'][train]):
        umask = train & (data['unit'] == unit)
        for speed in np.unique(data['speed_hz'][umask]):
            values = e[umask & (data['speed_hz'] == speed)]
            if len(values) < 2:
                raise ValueError('Uncertainty calibration requires same-unit, same-speed repeats')
            residuals.extend(abs(values-np.median(values, axis=0)))
    gamma = np.quantile(np.asarray(residuals), .95, axis=0)
    if np.any(gamma <= 0):
        raise ValueError('Nonpositive calibrated width; no automatic replacement')
    return theta, gamma


def representations(e, theta, gamma):
    g = e-theta
    return {'hard_order': (g >= 0).astype(float),
            'uncertainty_order': np.clip((g+gamma)/(2*gamma), 0, 1),
            'width_x4': np.clip((g+4*gamma)/(8*gamma), 0, 1),
            'continuous_order': e,
            'collapsed': np.zeros_like(e)}


def fit_head(z, y):
    mu = np.stack([z[y == c].mean(0) for c in sorted(set(y))])
    return Head(2*mu, -(mu*mu).sum(1))


class SymbolicCertificateTask:
    """Fit the P06 method from explicitly selected training acquisitions only.

    ``fit`` requires ``x, y, unit, split, speed_hz, fs``. Every supplied row must
    declare ``split='train'``; selecting source domains is the caller's protocol
    responsibility. Condition-bearing inputs must declare one complete source
    condition, its unique ``nominal_speed_hz`` and unique ``acquisition`` IDs.
    Features use measured speeds; repeats within that condition are grouped by
    physical unit and nominal speed. Without condition metadata, the caller must
    establish homogeneous acquisition settings; repeats use unit and supplied
    speed, as in the synthetic P4 protocol. ``symbols`` and ``predict`` need only
    ``x, speed_hz, fs`` and never update fitted state.

    Pair certificates take explicit expected and actual symbols. Sharing a unit
    identifier is deliberately not an automatic physical-pairing contract.
    """

    def __init__(self, *, orders: Any = ORDERS,
                 representation: str = 'uncertainty_order') -> None:
        orders = np.asarray(orders, dtype=float)
        if (orders.ndim != 1 or not len(orders) or
                not np.isfinite(orders).all() or
                len(np.unique(orders)) != len(orders) or
                np.any(orders - .12 < .5) or np.any(orders + .12 > 8)):
            raise ValueError('Distinct fault-order bands must lie within [0.5, 8]')
        if representation not in {'hard_order', 'uncertainty_order',
                                  'width_x4', 'continuous_order'}:
            raise ValueError('Unknown P06 representation')
        self.orders = orders.copy()
        self.representation = representation
        self.head: Head | None = None
        self.theta: np.ndarray | None = None
        self.gamma: np.ndarray | None = None
        self.fs: float | None = None
        self.training_units: tuple[str, ...] = ()
        self.source_condition: str | None = None
        self.source_nominal_speed_hz: float | None = None

    @staticmethod
    def _signals(data: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
        missing = {'x', 'speed_hz', 'fs'} - set(data)
        if missing:
            raise ValueError(f'Missing signal fields: {sorted(missing)}')
        x = np.asarray(data['x'], dtype=float)
        speed = np.asarray(data['speed_hz'], dtype=float)
        fs = np.asarray(data['fs'], dtype=float)
        if x.ndim != 2 or not len(x) or x.shape[1] < 4:
            raise ValueError('Expected nonempty signals x[N,L], L >= 4')
        if speed.shape != (len(x),):
            raise ValueError('One explicit speed_hz value per acquisition required')
        if fs.shape != () or not np.isfinite(fs) or fs <= 0:
            raise ValueError('Positive finite scalar fs required')
        if not np.isfinite(x).all() or not np.isfinite(speed).all() or np.any(speed <= 0):
            raise ValueError('Finite signals and positive finite speeds required')
        return x, speed, float(fs)

    def fit(self, training_data: Mapping[str, Any]) -> SymbolicCertificateTask:
        """Calibrate predicates and prototypes without accepting held-out rows."""
        missing = {'y', 'unit', 'split'} - set(training_data)
        if missing:
            raise ValueError(f'Missing explicit training metadata: {sorted(missing)}')
        x, speed, fs = self._signals(training_data)
        y, unit, split = (np.asarray(training_data[k]) for k in ('y', 'unit', 'split'))
        if any(values.shape != (len(x),) for values in (y, unit, split)):
            raise ValueError('Training metadata must contain one value per acquisition')
        if set(split) != {'train'}:
            raise ValueError('fit accepts explicitly selected train rows only; no val/test')
        if y.dtype.kind not in 'iu' or set(y) != set(range(len(self.orders) + 1)):
            raise ValueError('Labels must be integer 0=healthy and 1..K=fault orders')
        if unit.dtype.kind != 'U' or np.any(np.char.strip(unit) == ''):
            raise ValueError('Nonempty Unicode physical-unit identifiers required')
        source_condition, source_nominal_speed = None, None
        calibration_speed = speed
        if 'condition' in training_data:
            condition = np.asarray(training_data['condition'])
            if (condition.shape != (len(x),) or condition.dtype.kind != 'U' or
                    np.any(np.char.strip(condition) == '') or len(np.unique(condition)) != 1):
                raise ValueError('Select one complete source condition before calibration')
            source_condition = str(condition[0])
            if 'nominal_speed_hz' not in training_data:
                raise ValueError('A complete source condition requires nominal_speed_hz')
            nominal = np.asarray(training_data['nominal_speed_hz'], dtype=float)
            if (nominal.shape != (len(x),) or not np.isfinite(nominal).all() or
                    np.any(nominal <= 0) or len(np.unique(nominal)) != 1):
                raise ValueError('One positive finite nominal_speed_hz required for the source condition')
            if 'acquisition' not in training_data:
                raise ValueError('Condition-bearing inputs require independent acquisition IDs')
            source_nominal_speed = float(nominal[0])
            calibration_speed = nominal
        if 'acquisition' in training_data:
            acquisition = np.asarray(training_data['acquisition'])
            if (acquisition.shape != (len(x),) or acquisition.dtype.kind != 'U' or
                    np.any(np.char.strip(acquisition) == '') or
                    len(np.unique(acquisition)) != len(x)):
                raise ValueError('Unique nonempty acquisition IDs required; windows are not independent repeats')
        for identity in np.unique(unit):
            if len(np.unique(y[unit == identity])) != 1:
                raise ValueError(f'Contradictory physical-unit label: {identity}')
        e = features(x, speed, fs, self.orders)
        calibration_data = dict(y=y, unit=unit, split=split, speed_hz=calibration_speed)
        theta, gamma = calibrate(e, calibration_data)
        z = representations(e, theta, gamma)[self.representation]
        head = fit_head(z, y)
        # Commit fitted state only after every calculation succeeds.
        self.theta, self.gamma, self.head, self.fs = theta, gamma, head, fs
        self.training_units = tuple(map(str, np.unique(unit)))
        self.source_condition = source_condition
        self.source_nominal_speed_hz = source_nominal_speed
        return self

    def _fitted_head(self) -> Head:
        if self.head is None:
            raise RuntimeError('Call fit with explicit training data before inference')
        return self.head

    def symbols(self, data: Mapping[str, Any]) -> np.ndarray:
        """Transform acquisitions with frozen training calibration."""
        self._fitted_head()
        x, speed, fs = self._signals(data)
        if fs != self.fs:
            raise ValueError('Sampling rate differs from the fitted acquisition protocol')
        e = features(x, speed, fs, self.orders)
        return representations(e, self.theta, self.gamma)[self.representation]

    def predict(self, data: Mapping[str, Any]) -> np.ndarray:
        """Return exact-affine class decisions; -1 denotes a tied decision."""
        head = self._fitted_head()
        return np.asarray([head.decision(z) for z in self.symbols(data)], dtype=int)

    def check_pairs(self, actual_symbols: Any, *,
                    expected_symbols: Any) -> list[dict[str, Any]]:
        """Certify against caller-declared expected symbols, row by row.

        The caller must establish that each expected row is the intended symbolic
        action of the declared physical transformation. This method certifies
        only the resulting affine decision implication, not that premise.
        """
        head = self._fitted_head()
        actual = np.asarray(actual_symbols, dtype=float)
        expected = np.asarray(expected_symbols, dtype=float)
        if (actual.ndim != 2 or not len(actual) or actual.shape != expected.shape or
                actual.shape[1] != head.k or not np.isfinite(actual).all() or
                not np.isfinite(expected).all()):
            raise ValueError('Expected matching finite symbol matrices [N,K]')
        return [head.check(reference, observed)
                for reference, observed in zip(expected, actual)]

    def state(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the fitted scientific object."""
        head = self._fitted_head()
        return dict(orders=self.orders.tolist(), representation=self.representation,
                    theta=self.theta.tolist(), gamma=self.gamma.tolist(),
                    fs=self.fs, training_units=list(self.training_units),
                    source_condition=self.source_condition,
                    source_nominal_speed_hz=self.source_nominal_speed_hz,
                    calibration_groups=('unit_and_nominal_speed_within_source_condition'
                        if self.source_condition is not None else 'unit_and_supplied_speed'),
                    feature=dict(window='numpy.hanning', order_half_width=.12,
                                 normalization_order_band=[.5, 8.]),
                    weights=[[float(value) for value in row] for row in head.w],
                    bias=[float(value) for value in head.b],
                    semantics='exact_binary_rational_affine',
                    fit_population='explicit_train_rows_only')
