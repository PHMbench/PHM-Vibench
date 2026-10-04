"""Small counterexamples and exhaustive checks, not a substitute for proofs."""
import sys
from itertools import product
from pathlib import Path
import numpy as np
from scipy.stats import beta, binom
from src.task_factory.task.classification.selective_diagnosis.core import (box_bounds, class_certificate, max_box_average, calibrate,
                  matched_risk, risk_coverage)


def test_box_certificate_against_all_vertices():
    rng = np.random.default_rng(7)
    for _ in range(100):
        lo = rng.uniform(.02, .3, (1, 4)); hi = lo + rng.uniform(.01, .5, (1, 4))
        q = rng.dirichlet(np.ones(3), 4); cost = rng.uniform(0, 1, 4)
        vertices = np.array([np.where(b, hi[0], lo[0]) for b in product([0, 1], repeat=4)])
        w = vertices / vertices.sum(axis=1, keepdims=True)
        pred = np.array([((lo + hi) @ q).argmax()])
        margins = w @ q
        actual = all(np.all(p[pred[0]] > np.delete(p, pred[0])) for p in margins)
        assert bool(class_certificate(lo, hi, q, pred)[0]) == actual
        np.testing.assert_allclose(max_box_average(lo, hi, cost), (w @ cost).max(), atol=1e-12)


def test_prediction_stability_does_not_imply_acceptance_stability():
    lo = np.array([[.1, .1]]); hi = np.array([[.9, .9]])
    q = np.array([[.9, .1], [.8, .2]]); cost = np.array([0., 1.])
    assert class_certificate(lo, hi, q, np.array([0]))[0]
    assert max_box_average(lo, hi, cost)[0] > .6  # nominal cost=.5 accepts; a revision rejects


def test_gaussian_interval_contains_actual_interventions():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(10, 3)); c = rng.normal(size=(4, 3)); s = np.ones((4, 3)); eps = .2
    lo, hi, _ = box_bounds(x, c, s, eps)
    d = np.abs(x[:, None, :] - c)
    shift = (-.5 * np.sum(np.maximum(d - eps, 0) ** 2, axis=2)).max(axis=1, keepdims=True)
    for _ in range(50):
        xp = x + rng.uniform(-eps, eps, x.shape)
        h = np.exp(-.5 * np.sum((xp[:, None, :] - c) ** 2, axis=2) - shift)
        assert np.all(h >= lo - 1e-12) and np.all(h <= hi + 1e-12)


def test_exact_binomial_bound_has_required_coverage():
    # Integrate the failure event against the exact binomial distribution.
    n, eta = 40, .005
    k = np.arange(n + 1)
    u = np.ones(n + 1); u[:-1] = beta.ppf(1 - eta, k[:-1] + 1, n - k[:-1])
    for p in np.linspace(.001, .999, 99):
        assert binom.pmf(k, n, p)[u < p].sum() <= eta + 1e-12


def test_zero_acceptance_is_not_zero_conditional_risk():
    n = 5
    result = calibrate(np.zeros(n), np.ones(n), np.ones(n, bool), np.arange(n),
                       np.linspace(0, 1, 21), .05, .05, 'iid')
    assert not result['certified'] and result['threshold'] is None
    assert risk_coverage(np.zeros(n), np.zeros(n), np.arange(n)) == (0., None)


def test_cluster_calibration_does_not_count_duplicate_windows_as_units():
    e = np.array([0, 1, 0, 0]); s = np.array([.1, .9, .2, .3]); u = np.arange(4)
    a = calibrate(e, s, np.ones(4, bool), u, np.array([.5, 1.]), .1, .05)
    b = calibrate(np.repeat(e, 30), np.repeat(s, 30), np.ones(120, bool),
                  np.repeat(u, 30), np.array([.5, 1.]), .1, .05)
    assert a == b and a['calibration_units'] == 4


def test_matched_risk_uses_physical_unit_weighting():
    e = np.array([0, 1, 1]); s = np.array([0., 1., 2.]); u = np.array(['a', 'b', 'b'])
    assert matched_risk(e, s, np.ones(3, bool), u, .5) == 0
    np.testing.assert_allclose(matched_risk(e, s, np.ones(3, bool), u, .75), 1/3)


def test_matched_risk_boundary_ties_are_row_order_invariant():
    error = np.array([0., 0., 1., 1.])
    score, eligible, unit = np.full(4, .5), np.ones(4, bool), np.arange(4)
    for order in (np.arange(4), np.arange(4)[::-1], np.array([2, 0, 3, 1])):
        assert matched_risk(error[order], score[order], eligible[order], unit[order], .5) == .5


def test_matched_risk_ties_keep_unit_weights_under_unequal_window_counts():
    error = np.array([0., 1., 1., 1.])
    unit = np.array(['a', 'b', 'b', 'b'])
    score, eligible = np.full(4, .5), np.ones(4, bool)
    np.testing.assert_allclose(matched_risk(error, score, eligible, unit, .25), .5)
    np.testing.assert_allclose(matched_risk(error[::-1], score, eligible, unit[::-1], .75), .5)
