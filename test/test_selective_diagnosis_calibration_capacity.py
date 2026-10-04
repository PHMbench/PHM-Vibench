"""Verify the sample-count limit against the actual, unchanged calibrator."""
from pathlib import Path
import sys
import numpy as np
import pytest
from src.task_factory.task.classification.selective_diagnosis.core import calibration_capacity, calibrate


@pytest.mark.parametrize('mode,minimum', [('cluster', 2348), ('iid', 195)])
def test_best_case_transition_matches_calibrator(mode, minimum):
    grid = np.linspace(0, 1, 21)
    delta = .05 / (10 * 5)
    for m, expected in [(minimum - 1, False), (minimum, True)]:
        capacity = calibration_capacity(m, len(grid), .05, delta, mode)
        result = calibrate(np.zeros(m), np.zeros(m), np.ones(m, bool),
                           np.arange(m), grid, .05, delta, mode)
        assert capacity['minimum_units_best_case'] == minimum
        assert capacity['structurally_feasible'] == expected
        assert result['certified'] == expected
        if expected:
            np.testing.assert_allclose(result['upper'], capacity['best_possible_upper'])


def test_feasible_count_does_not_certify_a_bad_policy():
    m = 2348
    capacity = calibration_capacity(m, 21, .05, .001)
    result = calibrate(np.ones(m), np.zeros(m), np.ones(m, bool), np.arange(m),
                       np.linspace(0, 1, 21), .05, .001)
    assert capacity['structurally_feasible'] and not result['certified']


def test_multiplicity_is_not_omitted_from_count_limit():
    single = calibration_capacity(5, 21, .05, .05)
    full = calibration_capacity(5, 21, .05, .05 / 50)
    assert single['minimum_units_best_case'] == 1485
    assert full['minimum_units_best_case'] == 2348
    assert full['best_possible_upper'] == 1.0


@pytest.mark.parametrize('mode', ['cluster', 'iid'])
def test_observed_upper_cannot_beat_best_case(mode):
    rng = np.random.default_rng(25)
    m = 300
    for _ in range(6):
        score = rng.uniform(size=m)
        errors = rng.integers(0, 2, size=m)
        capacity = calibration_capacity(m, 21, .9, .05, mode)
        result = calibrate(errors, score, np.ones(m, bool), np.arange(m),
                           np.linspace(0, 1, 21), .9, .05, mode)
        if result['certified']:
            assert result['upper'] >= capacity['best_possible_upper'] - 1e-12
