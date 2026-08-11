from __future__ import annotations

import hashlib
import inspect
import json
import random
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.utils.p07_protocol.experiment_runner as runner_module
from src.utils.p07_protocol.cwru_manifest import (
    CWRUFold,
    CWRUManifest,
    ManifestSpecimen,
    WindowCoordinate,
)
from src.utils.p07_protocol.experiment_runner import (
    CWRUFileBatch,
    FrozenPathClassifier,
    TrainingBudget,
    build_cwru_arms,
    build_dirg_arms,
    compute_recovery_atoms,
    compute_stability_atoms,
    evaluate_file_macro_classifier,
    load_cwru_fold,
    seed_all_rng,
    select_exhaustive_oracle_path,
    select_full_discrete_path_classifier,
    train_file_macro_classifier,
    train_synthetic_reconstruction,
    validate_dirg_arms,
    validate_parameter_matched_arms,
)
from src.utils.p07_protocol.path_universe import (
    enumerate_path_records,
    oracle_execute_path,
)


class ScaleCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, x: torch.Tensor):
        return x * self.scale, None


class NaNCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale * torch.tensor(float("nan"), device=x.device)


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.mean(dim=1))


class SignClassifier(nn.Module):
    def __init__(self, *, nan: bool = False) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.nan = nan

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = x[:, 0, 0] + self.anchor * 0.0
        logits = torch.stack((score, -score), dim=1)
        return logits * float("nan") if self.nan else logits


def _small_budget(**overrides) -> TrainingBudget:
    values = {
        "learning_rate": 0.05,
        "weight_decay": 0.0,
        "batch_size": 2,
        "max_epochs": 5,
        "max_updates": 8,
        "patience": 2,
        "min_delta": 1.0e-8,
    }
    values.update(overrides)
    return TrainingBudget(**values)


def _file_batch(
    file_key: str,
    *,
    split: str,
    class_index: int,
    window_count: int,
    value: float,
) -> CWRUFileBatch:
    windows = torch.full((window_count, 4, 2), value, dtype=torch.float32)
    coordinates = tuple(
        WindowCoordinate(index=index, start=0, stop=4)
        for index in range(window_count)
    )
    return CWRUFileBatch(
        file_key=file_key,
        file_name=f"{file_key}.mat",
        split=split,  # type: ignore[arg-type]
        label=class_index + 1,
        class_index=class_index,
        file_weight=1.0,
        windows=windows,
        coordinates=coordinates,
    )


def _discrete_search_files() -> tuple[
    tuple[CWRUFileBatch, ...],
    tuple[CWRUFileBatch, ...],
]:
    train_files = (
        _file_batch(
            "search-train-0",
            split="train",
            class_index=0,
            window_count=1,
            value=0.0,
        ),
        _file_batch(
            "search-train-1",
            split="train",
            class_index=1,
            window_count=1,
            value=0.0,
        ),
        _file_batch(
            "search-train-2",
            split="train",
            class_index=2,
            window_count=1,
            value=0.0,
        ),
    )
    validation_files = (
        _file_batch(
            "search-validation-0",
            split="validation",
            class_index=0,
            window_count=1,
            value=0.0,
        ),
        _file_batch(
            "search-validation-1",
            split="validation",
            class_index=1,
            window_count=1,
            value=0.0,
        ),
        _file_batch(
            "search-validation-2",
            split="validation",
            class_index=2,
            window_count=1,
            value=0.0,
        ),
    )
    return train_files, validation_files


def _manifest_fixture() -> tuple[CWRUManifest, dict[str, torch.Tensor]]:
    definitions = (
        ("train-a", "a.mat", 1, "007", 0),
        ("train-b", "b.mat", 2, "014", 1),
        ("validation-a", "c.mat", 1, "007", 2),
        ("test-a", "d.mat", 2, "021", 3),
        ("excluded-a", "e.mat", 1, "021", 0),
    )
    specimens = []
    recordings: dict[str, torch.Tensor] = {}
    for index, (key, name, label, diameter, load) in enumerate(definitions):
        coordinates = (
            WindowCoordinate(index=0, start=0, stop=4),
            WindowCoordinate(index=1, start=8, stop=12),
        )
        specimens.append(
            ManifestSpecimen(
                specimen_key=key,
                metadata_id=100 + index,
                file_name=name,
                raw_size_bytes=10,
                raw_sha256=f"{index + 1:x}" * 64,
                dataset_id=1,
                dataset_name="RM_001_CWRU",
                fault_type="IR" if label == 1 else "B",
                label=label,
                diameter_code=diameter,
                diameter_mils=int(diameter),
                fault_level=1,
                domain_id=load,
                load_hp=load,
                sample_rate_hz=12000,
                channels=2,
                sample_length=12,
                file_weight=1.0,
                windows=coordinates,
            )
        )
        base = torch.arange(12, dtype=torch.float64) + index
        recordings[key] = torch.stack((base, base.square() + 1.0), dim=1)
    fold = CWRUFold(
        fold_id="fixture-fold",
        train_diameter_code="007",
        validation_diameter_code="014",
        test_diameter_code="021",
        train_specimen_keys=("train-a", "train-b"),
        validation_specimen_keys=("validation-a",),
        test_specimen_keys=("test-a",),
        excluded_specimen_keys=("excluded-a",),
    )
    provisional = CWRUManifest(
        schema_version=1,
        subset_id="fixture",
        official_source_url="https://example.invalid/read-only-fixture",
        metadata_subset_sha256="a" * 64,
        reader_source_sha256="b" * 64,
        preprocessing_source_sha256="c" * 64,
        specimens=tuple(specimens),
        folds=(fold,),
        root_sha256="",
    )
    payload = json.dumps(
        provisional.payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return replace(
        provisional, root_sha256=hashlib.sha256(payload).hexdigest()
    ), recordings


def test_training_budget_defaults_are_exact_and_frozen() -> None:
    budget = TrainingBudget()

    assert budget.optimizer == "AdamW"
    assert budget.learning_rate == 1.0e-3
    assert budget.weight_decay == 1.0e-4
    assert budget.batch_size == 64
    assert budget.max_epochs == 200
    assert budget.max_updates == 1600
    assert budget.patience == 20
    assert budget.min_delta == 1.0e-5
    with pytest.raises(FrozenInstanceError):
        budget.max_epochs = 1  # type: ignore[misc]


def test_seed_all_rng_covers_python_numpy_and_torch() -> None:
    seed_all_rng(19)
    first = (random.random(), float(np.random.rand()), torch.rand(4))
    seed_all_rng(19)
    second = (random.random(), float(np.random.rand()), torch.rand(4))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_synthetic_training_is_deterministic_and_uses_oracle_targets() -> None:
    generator = torch.Generator().manual_seed(3)
    train_x = torch.randn(6, 12, 1, generator=generator) * 0.1
    validation_x = torch.randn(4, 12, 1, generator=generator) * 0.1
    first = ScaleCore()
    second = ScaleCore()
    second.load_state_dict(first.state_dict())
    budget = _small_budget()

    first_trace = train_synthetic_reconstruction(
        first,
        train_x,
        validation_x,
        target_path=("I", "I", "I"),
        optimization_seed=31,
        budget=budget,
    )
    second_trace = train_synthetic_reconstruction(
        second,
        train_x,
        validation_x,
        target_path=("I", "I", "I"),
        optimization_seed=31,
        budget=budget,
    )

    assert first_trace == second_trace
    assert torch.equal(first.scale, second.scale)
    assert first_trace.best_state_restored is True
    assert first_trace.updates_completed <= budget.max_updates


def test_synthetic_training_early_stops_and_nan_fails_closed() -> None:
    x = torch.linspace(-0.2, 0.2, 48).reshape(4, 12, 1)
    trace = train_synthetic_reconstruction(
        ScaleCore(),
        x,
        x,
        target_path=("I", "I", "I"),
        optimization_seed=7,
        budget=_small_budget(max_epochs=10, max_updates=30, patience=1, min_delta=1.0),
    )
    assert trace.stopped_early is True
    assert trace.stop_reason == "early_stopping"

    with pytest.raises(FloatingPointError, match="non-finite"):
        train_synthetic_reconstruction(
            NaNCore(),
            x,
            x,
            target_path=("I", "I", "I"),
            optimization_seed=7,
            budget=_small_budget(max_epochs=1),
        )


def test_exhaustive_selector_enforces_primary_216_and_uses_public_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.linspace(-0.2, 0.3, 16).reshape(2, 8, 1)
    target = oracle_execute_path(x, ("I", "I", "I"))
    original = runner_module._path_universe.oracle_execute_path
    calls = 0

    def counted(inputs: torch.Tensor, path):
        nonlocal calls
        calls += 1
        return original(inputs, path)

    monkeypatch.setattr(runner_module._path_universe, "oracle_execute_path", counted)
    result = select_exhaustive_oracle_path(x, target)

    assert result.evaluated_paths == 216
    assert len(result.evaluations) == 216
    assert calls == 216
    assert result.selected_path == ("I", "I", "I")
    with pytest.raises(ValueError, match="requires evaluation_budget=216"):
        select_exhaustive_oracle_path(x, target, evaluation_budget=8)
    with pytest.raises(ValueError, match=r"\[1, 216\]"):
        select_exhaustive_oracle_path(
            x, target, evaluation_budget=217, primary=False
        )
    smoke = select_exhaustive_oracle_path(
        x, target, evaluation_budget=3, primary=False
    )
    assert smoke.evaluated_paths == 3


def test_recovery_and_stability_return_atoms_without_claim_decisions() -> None:
    target = ("SQUARE", "ABS", "I")
    semantically_equal = ("I", "SQUARE", "I")
    atoms = compute_recovery_atoms(target, semantically_equal)

    assert atoms.exact_match is False
    assert atoms.semantic_match is True
    assert atoms.raw_edit_distance > 0
    assert atoms.canonical_edit_distance == 0

    stability = compute_stability_atoms(
        {7: target, 20: semantically_equal, 31: ("D1", "I", "I")}
    )
    assert len(stability) == 3
    pair = next(item for item in stability if (item.left_seed, item.right_seed) == (7, 20))
    assert pair.exact_path_agreement is False
    assert pair.semantic_path_agreement is True


def test_cwru_loader_reads_only_fold_files_and_manifest_coordinates() -> None:
    manifest, recordings = _manifest_fixture()
    calls: list[str] = []

    def read_fn(specimen: ManifestSpecimen) -> torch.Tensor:
        calls.append(specimen.specimen_key)
        return recordings[specimen.specimen_key]

    fold = load_cwru_fold(
        manifest,
        "fixture-fold",
        read_fn=read_fn,
        dtype=torch.float64,
    )

    assert calls == ["train-a", "train-b", "validation-a", "test-a"]
    assert "excluded-a" not in calls
    assert [item.file_key for item in fold.train_files] == ["train-a", "train-b"]
    assert [item.file_key for item in fold.validation_files] == ["validation-a"]
    assert [item.file_key for item in fold.test_files] == ["test-a"]
    for item in (*fold.train_files, *fold.validation_files, *fold.test_files):
        torch.testing.assert_close(
            item.windows.mean(dim=1), torch.zeros(2, 2, dtype=torch.float64), atol=1e-12, rtol=0
        )
        torch.testing.assert_close(
            item.windows.square().mean(dim=1).sqrt(),
            torch.ones(2, 2, dtype=torch.float64),
            atol=1e-12,
            rtol=0,
        )
        specimen = next(
            specimen
            for specimen in manifest.specimens
            if specimen.specimen_key == item.file_key
        )
        assert item.coordinates == tuple(specimen.windows)


def test_file_macro_evaluation_weights_files_not_windows() -> None:
    correct_short = _file_batch(
        "short", split="test", class_index=0, window_count=1, value=1.0
    )
    wrong_long = _file_batch(
        "long", split="test", class_index=1, window_count=9, value=1.0
    )
    result = evaluate_file_macro_classifier(
        SignClassifier(), (correct_short, wrong_long), batch_size=3
    )

    correct_loss = float(F.cross_entropy(torch.tensor([[1.0, -1.0]]), torch.tensor([0])))
    wrong_loss = float(F.cross_entropy(torch.tensor([[1.0, -1.0]]), torch.tensor([1])))
    assert result.macro_accuracy == 0.5
    assert result.macro_loss == pytest.approx((correct_loss + wrong_loss) / 2.0)
    assert result.independent_unit_count == 2
    assert result.total_window_count == 10
    assert result.evaluation_unit == "file"


def test_classifier_training_rejects_test_as_validation_and_is_deterministic() -> None:
    train_files = (
        _file_batch("train-0", split="train", class_index=0, window_count=2, value=1.0),
        _file_batch("train-1", split="train", class_index=1, window_count=2, value=-1.0),
    )
    validation_files = (
        _file_batch("validation-0", split="validation", class_index=0, window_count=1, value=1.0),
        _file_batch("validation-1", split="validation", class_index=1, window_count=1, value=-1.0),
    )
    test_files = (
        _file_batch("test-0", split="test", class_index=0, window_count=1, value=1.0),
    )
    with pytest.raises(ValueError, match="Expected only validation files"):
        train_file_macro_classifier(
            TinyClassifier(),
            train_files,
            test_files,
            optimization_seed=11,
            budget=_small_budget(max_epochs=1),
        )

    first = TinyClassifier()
    second = TinyClassifier()
    second.load_state_dict(first.state_dict())
    budget = _small_budget(max_epochs=2, max_updates=4)
    first_trace = train_file_macro_classifier(
        first,
        train_files,
        validation_files,
        optimization_seed=11,
        budget=budget,
    )
    second_trace = train_file_macro_classifier(
        second,
        train_files,
        validation_files,
        optimization_seed=11,
        budget=budget,
    )
    assert first_trace == second_trace
    for left, right in zip(first.parameters(), second.parameters()):
        assert torch.equal(left, right)


def test_classifier_nan_output_fails_closed() -> None:
    file_batch = _file_batch(
        "nan", split="test", class_index=0, window_count=1, value=1.0
    )
    with pytest.raises(FloatingPointError, match="non-finite logits"):
        evaluate_file_macro_classifier(SignClassifier(nan=True), (file_batch,))


def test_frozen_path_classifier_uses_public_oracle_and_population_pooling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = runner_module._path_universe.oracle_execute_path
    calls: list[tuple[str, str, str]] = []

    def counted(inputs: torch.Tensor, path):
        calls.append(tuple(path))
        return original(inputs, path)

    monkeypatch.setattr(
        runner_module._path_universe,
        "oracle_execute_path",
        counted,
    )
    model = FrozenPathClassifier(
        ("I", "I", "I"),
        in_channels=2,
        num_classes=3,
    )
    train_files = tuple(
        _file_batch(
            f"ridge-train-{class_index}",
            split="train",
            class_index=class_index,
            window_count=1,
            value=float(class_index),
        )
        for class_index in range(3)
    )
    iterations = model.fit_train_files(train_files)

    assert iterations >= 0
    assert model.is_fitted is True
    assert model.scaler is not None
    assert model.estimator is not None
    np.testing.assert_allclose(model.scaler.mean_, [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(model.scaler.var_, [2.0 / 3.0, 2.0 / 3.0, 0.0, 0.0])
    np.testing.assert_allclose(model.scaler.scale_[2:], [1.0, 1.0])
    assert model.estimator.penalty == "l2"
    assert model.estimator.C == 1.0
    assert model.estimator.solver == "lbfgs"
    assert model.estimator.multi_class == "multinomial"
    assert model.estimator.fit_intercept is True
    assert model.estimator.tol == 1.0e-8
    assert model.estimator.max_iter == 1000
    np.testing.assert_array_equal(model.estimator.classes_, [0, 1, 2])
    inputs = torch.tensor(
        [[[1.0, -1.0], [3.0, 2.0], [-2.0, 4.0]]],
        dtype=torch.float32,
    )
    observed_features: list[np.ndarray] = []
    original_transform = model.scaler.transform

    def capture_transform(features: np.ndarray) -> np.ndarray:
        observed_features.append(features.copy())
        return original_transform(features)

    monkeypatch.setattr(model.scaler, "transform", capture_transform)

    logits = model(inputs)
    transformed = original(inputs, model.raw_path)
    expected_features = torch.cat(
        (
            transformed.mean(dim=1),
            transformed.var(dim=1, unbiased=False),
        ),
        dim=1,
    ).numpy()

    assert calls == [("I", "I", "I")] * 4
    assert logits.shape == (1, 3)
    np.testing.assert_allclose(observed_features[0], expected_features)
    with pytest.raises(RuntimeError, match="fitted exactly once"):
        model.fit_train_files(train_files)


def test_primary_discrete_classifier_search_evaluates_216_and_registry_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_files, validation_files = _discrete_search_files()
    original_fit = runner_module.LogisticRegression.fit
    fit_calls = 0

    def counted_fit(estimator, features, labels):
        nonlocal fit_calls
        fit_calls += 1
        return original_fit(estimator, features, labels)

    monkeypatch.setattr(runner_module.LogisticRegression, "fit", counted_fit)
    result = select_full_discrete_path_classifier(
        train_files,
        validation_files,
        bookkeeping_seed=43,
        num_classes=3,
    )

    first = enumerate_path_records()[0]
    assert fit_calls == 216
    assert result.evaluated_paths == 216
    assert result.evaluation_budget == 216
    assert result.primary is True
    assert len(result.evaluations) == 216
    assert result.selected_path == first.raw_path
    assert result.selected_raw_path_id == first.raw_path_id
    assert result.selected_class_id == first.class_id
    assert len(
        {
            (item.validation_macro_accuracy, item.validation_macro_loss)
            for item in result.evaluations
        }
    ) == 1
    assert all(item.solver_iterations >= 0 for item in result.evaluations)
    assert result.compute.candidate_fits == 216
    assert result.compute.candidate_validation_evaluations == 216
    assert result.compute.total_solver_iterations == sum(
        item.solver_iterations for item in result.evaluations
    )
    assert result.compute.wall_time_seconds > 0.0
    assert result.classifier_spec.sklearn_version == "1.2.2"
    assert result.classifier_spec.c == 1.0
    assert result.classifier_spec.stochastic_fit is False
    assert result.bookkeeping_seed == 43
    assert result.seed_role == "bookkeeping_only_seed_invariant_fit"
    assert result.fit_reuse_scope == "once_per_fold_not_per_optimization_seed"
    assert result.selected_model.training is False
    assert result.selected_model.is_fitted is True
    assert tuple(result.selected_model.parameters()) == ()
    assert result.selected_model.scaler is not None
    np.testing.assert_allclose(result.selected_model.scaler.scale_, np.ones(4))

    test_evaluation = evaluate_file_macro_classifier(
        result.selected_model,
        (
            _file_batch(
                "search-test",
                split="test",
                class_index=0,
                window_count=1,
                value=0.0,
            ),
        ),
    )
    assert test_evaluation.split == "test"
    assert test_evaluation.independent_unit_count == 1
    assert test_evaluation.predictions[0].predicted_class_index == 0


def test_discrete_classifier_search_is_truthless_deterministic_and_test_blind() -> None:
    train_files, validation_files = _discrete_search_files()
    test_files = (
        _file_batch(
            "forbidden-test",
            split="test",
            class_index=0,
            window_count=1,
            value=1.0,
        ),
    )
    parameters = inspect.signature(select_full_discrete_path_classifier).parameters

    assert "test_files" not in parameters
    assert "target_path" not in parameters
    assert "budget" not in parameters
    assert "optimization_seed" not in parameters
    first = select_full_discrete_path_classifier(
        train_files,
        validation_files,
        bookkeeping_seed=47,
        num_classes=3,
        evaluation_budget=3,
        primary=False,
    )
    second = select_full_discrete_path_classifier(
        train_files,
        validation_files,
        bookkeeping_seed=999,
        num_classes=3,
        evaluation_budget=3,
        primary=False,
    )
    assert first.evaluations == second.evaluations
    assert first.selected_path == second.selected_path
    assert first.selected_validation_macro_accuracy == (
        second.selected_validation_macro_accuracy
    )
    assert first.selected_model.estimator is not None
    assert second.selected_model.estimator is not None
    np.testing.assert_array_equal(
        first.selected_model.estimator.coef_,
        second.selected_model.estimator.coef_,
    )

    with pytest.raises(TypeError, match="unexpected keyword argument 'test_files'"):
        select_full_discrete_path_classifier(
            train_files,
            validation_files,
            bookkeeping_seed=47,
            num_classes=3,
            test_files=test_files,  # type: ignore[call-arg]
        )
    with pytest.raises(ValueError, match="Expected only validation files"):
        select_full_discrete_path_classifier(
            train_files,
            test_files,
            bookkeeping_seed=47,
            num_classes=3,
            evaluation_budget=1,
            primary=False,
        )
    with pytest.raises(ValueError, match="requires evaluation_budget=216"):
        select_full_discrete_path_classifier(
            train_files,
            validation_files,
            bookkeeping_seed=47,
            num_classes=3,
            evaluation_budget=3,
        )


def test_cwru_arm_builder_freezes_configs_counts_and_parameter_guard() -> None:
    arms = build_cwru_arms(initialization_seed=13, random_dictionary_seed=701)
    by_id = {item.arm_id: item for item in arms}

    assert tuple(by_id) == (
        "proposed",
        "dense_operator_mixture",
        "random_dictionary",
        "attention_cnn",
        "explainable_cnn",
        "discrete_search",
    )
    assert by_id["proposed"].trainable_parameter_count == 2864
    assert by_id["proposed"].model.inference_mode == "discrete"
    assert by_id["dense_operator_mixture"].trainable_parameter_count == 2864
    assert by_id["random_dictionary"].trainable_parameter_count == 2864
    assert by_id["attention_cnn"].trainable_parameter_count == 2917
    assert by_id["explainable_cnn"].trainable_parameter_count == 3006
    assert by_id["discrete_search"].model is None
    assert by_id["discrete_search"].trainable_parameter_count is None
    assert by_id["attention_cnn"].model.channels == [20]  # type: ignore[union-attr]
    first_conv = by_id["explainable_cnn"].model.features[0]  # type: ignore[union-attr]
    assert first_conv.out_channels == 7

    forged = tuple(
        replace(item, trainable_parameter_count=4000)
        if item.arm_id == "attention_cnn"
        else item
        for item in arms
    )
    with pytest.raises(ValueError, match="parameter gap"):
        validate_parameter_matched_arms(forged)
    with pytest.raises(ValueError, match="frozen 5% limit"):
        validate_parameter_matched_arms(arms, maximum_relative_gap=0.051)


def test_dirg_arm_builder_freezes_explicit_profile_counts_and_gaps() -> None:
    parameters = inspect.signature(build_dirg_arms).parameters
    assert "in_channels" not in parameters
    assert "num_classes" not in parameters
    assert "dropout" not in parameters

    arms = build_dirg_arms(initialization_seed=13, random_dictionary_seed=701)
    by_id = {item.arm_id: item for item in arms}

    assert tuple(by_id) == (
        "proposed",
        "dense_operator_mixture",
        "random_dictionary",
        "attention_cnn",
        "explainable_cnn",
        "discrete_search",
    )
    assert by_id["proposed"].trainable_parameter_count == 4892
    assert by_id["dense_operator_mixture"].trainable_parameter_count == 4892
    assert by_id["random_dictionary"].trainable_parameter_count == 4892
    assert by_id["attention_cnn"].trainable_parameter_count == 4720
    assert by_id["explainable_cnn"].trainable_parameter_count == 5123
    assert by_id["discrete_search"].model is None
    assert by_id["discrete_search"].trainable_parameter_count is None

    proposed = by_id["proposed"].model
    assert proposed is not None
    assert proposed.in_channels == 6  # type: ignore[union-attr]
    assert proposed.num_classes == 2  # type: ignore[union-attr]
    assert proposed.inference_mode == "discrete"  # type: ignore[union-attr]
    for arm_id in ("proposed", "dense_operator_mixture", "random_dictionary"):
        core_model = by_id[arm_id].model
        assert core_model is not None
        assert core_model.classifier[1].in_features == 12  # type: ignore[union-attr]
        assert core_model.classifier[1].out_features == 67  # type: ignore[union-attr]
        assert core_model.classifier[3].p == 0.1  # type: ignore[union-attr]
        assert core_model.classifier[4].out_features == 2  # type: ignore[union-attr]

    attention = by_id["attention_cnn"].model
    assert attention is not None
    assert attention.input_dim == 6  # type: ignore[union-attr]
    assert attention.channels == [24]  # type: ignore[union-attr]
    assert attention.use_attention is True  # type: ignore[union-attr]
    assert attention.num_classes == 2  # type: ignore[union-attr]
    assert attention.dropout == 0.1  # type: ignore[union-attr]

    explainable = by_id["explainable_cnn"].model
    assert explainable is not None
    assert explainable.features[0].in_channels == 6  # type: ignore[union-attr]
    assert explainable.features[0].out_channels == 9  # type: ignore[union-attr]
    assert explainable.head.out_features == 2  # type: ignore[union-attr]
    assert explainable.dropout.p == 0.1  # type: ignore[union-attr]

    reference = by_id["proposed"].trainable_parameter_count
    assert reference is not None
    for arm_id in ("dense_operator_mixture", "random_dictionary"):
        assert by_id[arm_id].trainable_parameter_count == reference
    assert abs(4720 - reference) / reference == pytest.approx(0.03515944399)
    assert abs(5123 - reference) / reference == pytest.approx(0.04721995094)


def test_dirg_arm_profile_fails_closed_on_count_or_core_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arms = build_dirg_arms()
    forged_random = tuple(
        replace(item, trainable_parameter_count=4891)
        if item.arm_id == "random_dictionary"
        else item
        for item in arms
    )
    with pytest.raises(RuntimeError, match="frozen trainable parameter count"):
        validate_dirg_arms(forged_random)
    with pytest.raises(ValueError, match="exactly equal trainable parameter counts"):
        validate_parameter_matched_arms(forged_random)

    original_counter = runner_module.count_trainable_parameters

    def drifted_counter(model: nn.Module) -> int:
        return original_counter(model) + 1

    monkeypatch.setattr(
        runner_module,
        "count_trainable_parameters",
        drifted_counter,
    )
    with pytest.raises(RuntimeError, match="frozen trainable parameter count"):
        build_dirg_arms()
