"""Numerical software evidence; synthetic windows are not DG performance results."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from scipy.special import softmax

from src.task_factory.task.classification.multiview_signal import (
    OffsetReadout, ProgramSpec, ReadoutState, aggregate_metrics,
    equal_unit_weights, feature_names, fit_readouts, paired_brier_summary,
    per_unit_metrics, program_features, project_brier, source_guard, view_indices,
)


def spec() -> ProgramSpec:
    return ProgramSpec.from_dict({
        "sample_rate_hz": 1024.0,
        "bands_hz": [[64., 12.], [128., 24.], [192., 32.], [256., 48.]],
        "modulation_bands_hz": [[12., 20.], [24., 40.], [48., 80.]],
        "lags_samples": [4, 8, 16],
    })


def test_fixed_views_have_exact_coordinates_and_physical_operator_values():
    time = np.arange(1024) / 1024.
    carrier = np.cos(2 * np.pi * 128 * time)
    windows = np.stack((carrier, (1 + .6 * np.cos(2 * np.pi * 16 * time)) * carrier))[:, :, None]
    features, names = program_features(windows, spec())
    assert names == feature_names(spec())
    assert features.shape == (2, 42)
    views = view_indices()
    assert {key: len(value) for key, value in views.items()} == {
        "transient": 10, "band": 4, "modulation": 16, "recurrence": 12}
    np.testing.assert_array_equal(np.sort(np.concatenate(list(views.values()))), np.arange(42))
    assert views["band"].tolist() == [2, 12, 22, 32]
    env = names.index("ch0/band1[128Hz,24Hz]/envelope_modulation")
    mod = names.index("ch0/band1[128Hz,24Hz]/envelope_power[12,20Hz]")
    teager = names.index("ch0/band1[128Hz,24Hz]/teager_signed_ratio")
    acf = names.index("ch0/band1[128Hz,24Hz]/acf_lag8[0.0078125s]")
    assert features[1, env] > features[0, env] + .05
    assert features[1, mod] > .99
    assert features[0, teager] == pytest.approx(1., abs=1e-10)
    assert features[0, acf] == pytest.approx(1. - 8 / 1024, abs=1e-10)
    multi, multi_names = program_features(np.concatenate((windows, windows), axis=2), spec())
    assert multi.shape == (2, 84) and len(multi_names) == 84
    np.testing.assert_array_equal(multi[:, :42], multi[:, 42:])
    zero, _ = program_features(np.zeros_like(windows), spec())
    np.testing.assert_array_equal(zero, np.zeros((2, 42)))
    invalid = spec().to_dict()
    invalid["bands_hz"] = [(600., 12.), *spec().bands_hz[1:]]
    with pytest.raises(ValueError, match="Nyquist"):
        ProgramSpec.from_dict(invalid)
    invalid = spec().to_dict()
    invalid["modulation_bands_hz"] = [(1.01, 1.02), *spec().modulation_bands_hz[1:]]
    with pytest.raises(ValueError, match="DFT bin"):
        program_features(windows, ProgramSpec.from_dict(invalid))
    invalid = spec().to_dict()
    invalid["lags_samples"] = [4, 8, 1023]
    with pytest.raises(ValueError, match="lag"):
        program_features(windows, ProgramSpec.from_dict(invalid))


def source_fixture() -> dict:
    rng = np.random.default_rng(19)
    result = {"feature_names": feature_names(spec())}
    for role, unit_count in (("grow", 9), ("select", 6)):
        labels = np.repeat(np.arange(unit_count) % 3, 10)
        features = rng.normal(scale=.5, size=(len(labels), 42))
        features[:, :3] += 2. * np.eye(3)[labels]
        result[role + "_features"] = features
        result[role + "_native"] = rng.normal(size=(len(labels), 64))
        result[role + "_logits"] = .4 * np.eye(3)[labels] + rng.normal(scale=.5, size=(len(labels), 3))
        result[role + "_labels"] = labels
        result[role + "_units"] = np.repeat([f"{role}-{i}" for i in range(unit_count)], 10)
    return result


def test_25_identities_preserve_original_13_and_saved_source_state_replays():
    source = source_fixture()
    original = fit_readouts(**source, include_single_views=False)
    complete = fit_readouts(**source)
    values = (source["select_features"], source["select_native"], source["select_logits"])
    original_predictions, predictions = original.predict(*values), complete.predict(*values)
    assert len(original_predictions) == 13 and len(predictions) == 25
    assert list(predictions)[:13] == list(original_predictions)
    for key in original_predictions:
        np.testing.assert_array_equal(predictions[key], original_predictions[key])
    for view, count in (("transient", 10), ("band", 4), ("modulation", 16), ("recurrence", 12)):
        assert len(complete.states["signal_" + view]["readout"]["mean"]) == count
    # Replay after JSON key sorting and with arbitrary unseen sample values.
    saved = json.dumps(complete.to_dict(), sort_keys=True, allow_nan=False)
    restored = ReadoutState.from_dict(json.loads(saved))
    replay = restored.predict(*values)
    assert list(replay) == list(predictions)
    for key in predictions:
        np.testing.assert_array_equal(replay[key], predictions[key])
    invalid_state = json.loads(saved)
    coefficient = np.array(invalid_state["states"]["signal_program"]["readout"]["coefficient"])
    invalid_state["states"]["signal_program"]["readout"]["coefficient"] = coefficient.T.tolist()
    with pytest.raises(ValueError, match="saved offset-readout"):
        ReadoutState.from_dict(invalid_state)
    shifted = (values[0] + 25, values[1] - 10, values[2] + .1)
    restored.predict(*shifted)
    assert json.dumps(restored.to_dict(), sort_keys=True, allow_nan=False) == saved

    readout = OffsetReadout.from_dict(complete.states["signal_program"]["readout"])
    terms = readout.contributions(values[0])
    raw_logits = readout.logits(values[0], values[2])
    np.testing.assert_allclose(raw_logits, values[2] + readout.intercept + terms.sum(axis=1), atol=1e-12)
    relative = (values[2][:, :-1] - values[2][:, -1:]
                + readout.intercept[:-1] - readout.intercept[-1]
                + (terms[:, :, :-1] - terms[:, :, -1:]).sum(axis=1))
    np.testing.assert_allclose(raw_logits[:, :-1] - raw_logits[:, -1:], relative, atol=1e-12)
    probability = predictions["signal_program/raw"]
    np.testing.assert_allclose(np.log(probability[:, :-1] / probability[:, -1:]), relative, atol=1e-12)
    zero = copy.deepcopy(readout)
    zero.coefficient[:] = 0.
    zero.intercept[:] = 0.
    np.testing.assert_array_equal(zero.predict(values[0], values[2]), softmax(values[2], axis=1))
    recovered = complete.to_dict()
    for state in recovered["states"].values():
        state["projection"]["strength"] = 0.
    for key, probability in ReadoutState.from_dict(recovered).predict(*values).items():
        if key == "TSPN" or not key.endswith("/raw"):
            np.testing.assert_array_equal(probability, softmax(values[2], axis=1))
    invalid = source_fixture()
    invalid["select_units"][0] = invalid["grow_units"][0]
    with pytest.raises(ValueError, match="cross"):
        fit_readouts(**invalid)


def test_grow_projection_and_select_guard_require_both_losses():
    rng = np.random.default_rng(7)
    base, candidate = softmax(rng.normal(size=(27, 3)), axis=1), softmax(rng.normal(size=(27, 3)), axis=1)
    labels, units = rng.integers(0, 3, 27), np.repeat(np.arange(9), 3)
    projection = project_brier(base, candidate, labels, units)
    strength = projection["strength"]
    mixture = (1 - strength) * base + strength * candidate
    expected_gain = (aggregate_metrics(base, labels, units)["brier"]
                     - aggregate_metrics(mixture, labels, units)["brier"])
    assert expected_gain == pytest.approx(projection["empirical_gain"], abs=1e-12)
    assert expected_gain >= -1e-12
    assert project_brier(base, base, labels, units)["strength"] == 0.
    labels, units = np.array([0, 1]), np.array(["A", "B"])
    # Better Brier with worse error must be rejected.
    base = np.array([[.4, .35, .25], [.98, .01, .01]])
    candidate = np.array([[.4, .41, .19], [.34, .33, .33]])
    guard = source_guard(base, candidate, labels, units)
    assert guard["brier_excess"] < 0 and guard["error_excess"] > 0 and not guard["accepted"]
    # Better error with worse Brier must also be rejected.
    base = np.array([[.9, .05, .05], [.49, .48, .03]])
    candidate = np.array([[.34, .33, .33], [.33, .34, .33]])
    guard = source_guard(base, candidate, labels, units)
    assert guard["brier_excess"] > 0 and guard["error_excess"] < 0 and not guard["accepted"]
    equal = source_guard(base, base, labels, units)
    assert equal["accepted"] and equal["guarantee_scope"] == "empirical_guard_only_no_population_guarantee"


def test_equal_unit_macro_f1_and_seed_averaged_paired_bootstrap():
    probabilities = np.array([[.9, .05, .05]] * 100 + [[.8, .15, .05]])
    labels, units = np.array([0] * 100 + [1]), np.array(["A"] * 100 + ["B"])
    result = aggregate_metrics(probabilities, labels, units)
    assert result["accuracy"] == pytest.approx(.5)
    # Confusion has A -> 0 with mass 1/2 and B -> 0 with mass 1/2;
    # fixed three-class Macro-F1 = (2/3 + 0 + 0)/3, not mean unit F1.
    assert result["macro_f1"] == pytest.approx(2 / 9)
    expected = np.array([[.5, 0, 0], [.5, 0, 0], [0, 0, 0]])
    np.testing.assert_allclose(result["confusion_matrix"], expected, atol=1e-12)
    np.testing.assert_allclose(equal_unit_weights(units).sum(), 1.)
    losses = per_unit_metrics(probabilities, labels, units)
    assert result["brier"] == pytest.approx(losses["brier"].mean())
    replicated = aggregate_metrics(np.repeat(probabilities, 3, axis=0),
                                   np.repeat(labels, 3), np.repeat(units, 3))
    for key in ("brier", "accuracy", "macro_f1", "independent_units"):
        assert replicated[key] == pytest.approx(result[key])
    np.testing.assert_allclose(replicated["confusion_matrix"], result["confusion_matrix"], atol=1e-12)
    # Four units, three repeated fitted models, two methods; n remains four.
    unit_losses = np.array([[[.2, .1], [.5, .3], [.7, .9], [.8, .4]],
                            [[.4, .2], [.4, .3], [.6, .8], [.9, .5]],
                            [[.3, .2], [.6, .4], [.8, 1.], [.7, .4]]])
    first = paired_brier_summary(unit_losses, ["TSPN", "signal_program/guarded"],
                                 np.array(["A", "B", "C", "D"]), bootstrap=400, seed=17)
    second = paired_brier_summary(np.repeat(unit_losses, 2, axis=0),
                                  ["TSPN", "signal_program/guarded"],
                                  np.array(["A", "B", "C", "D"]), bootstrap=400, seed=17)
    assert first["independent_units"] == second["independent_units"] == 4
    for key in ("mean_excess", "ci_low", "ci_high"):
        assert first["comparisons"][0][key] == pytest.approx(second["comparisons"][0][key], abs=1e-14)
    average = unit_losses.mean(axis=0)
    draws = np.random.default_rng(17).integers(0, 4, size=(400, 4))
    expected_delta = (average[:, 1] - average[:, 0])[draws].mean(axis=1)
    np.testing.assert_allclose([first["comparisons"][0]["ci_low"], first["comparisons"][0]["ci_high"]],
                               np.quantile(expected_delta, [.025, .975]), atol=1e-14)
    strata = {"A": "healthy", "B": "healthy", "C": "inner", "D": "outer"}
    stratified = paired_brier_summary(unit_losses, ["TSPN", "signal_program/guarded"],
                                     np.array(["A", "B", "C", "D"]),
                                     bootstrap=400, seed=17, unit_strata=strata)
    rng = np.random.default_rng(17)
    draws = np.concatenate([rng.choice(indices, size=(400, len(indices)), replace=True)
                            for indices in (np.array([0, 1]), np.array([2]), np.array([3]))], axis=1)
    expected_delta = (average[:, 1] - average[:, 0])[draws].mean(axis=1)
    assert stratified["singleton_strata"] == ["inner", "outer"]
    np.testing.assert_allclose([stratified["comparisons"][0]["ci_low"], stratified["comparisons"][0]["ci_high"]],
                               np.quantile(expected_delta, [.025, .975]), atol=1e-14)
    for invalid in (np.array([[np.nan, 0, 1]]), np.array([[.6, .5, -.1]]),
                    np.array([[.2, .2, .2]])):
        with pytest.raises(ValueError):
            aggregate_metrics(invalid, np.array([0]), np.array(["A"]))
    with pytest.raises(ValueError, match="integer"):
        aggregate_metrics(np.array([[.9, .1]]), np.array([0.]), np.array(["A"]))
