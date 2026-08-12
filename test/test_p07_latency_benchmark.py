from __future__ import annotations

import ast
import inspect
import json
from collections import Counter
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch

import src.utils.p07_protocol.latency_benchmark as latency_module
from src.utils.p07_protocol.latency_benchmark import (
    BACKEND_ID,
    DEFAULT_SCHEDULE_SEED,
    MIN_TIMED_CALLS_PER_ARM,
    MIN_WARMUP_CALLS_PER_ARM,
    LatencyArm,
    LatencyBenchmarkConfig,
    balanced_interleaved_schedule,
    benchmark_cpu_latency,
    conservative_order_statistic,
)


class SequenceTimer:
    def __init__(self, values: list[int | float]) -> None:
        self._values = iter(values)
        self.call_count = 0

    def __call__(self):
        self.call_count += 1
        return next(self._values)


def _config(**updates) -> LatencyBenchmarkConfig:
    base = LatencyBenchmarkConfig(
        precision="float32",
        backend=BACKEND_ID,
        required_intraop_threads=torch.get_num_threads(),
        required_interop_threads=torch.get_num_interop_threads(),
    )
    return replace(base, **updates)


def _safe_arms() -> tuple[LatencyArm, LatencyArm]:
    return (
        LatencyArm("proposed", lambda: 0.5),
        LatencyArm("reference", lambda: 0.4),
    )


def _timer_values(durations: list[int]) -> list[int]:
    current = 10_000
    values = []
    for duration in durations:
        values.extend((current, current + duration))
        current += duration + 7
    return values


def test_schedule_is_balanced_deterministic_domain_separated_and_rng_isolated() -> None:
    names = ("proposed", "fast", "slow")
    np.random.seed(19)
    expected_global_draw = np.random.random(4)
    np.random.seed(19)
    first = balanced_interleaved_schedule(
        names, 123, DEFAULT_SCHEDULE_SEED, domain="timed"
    )
    observed_global_draw = np.random.random(4)
    second = balanced_interleaved_schedule(
        tuple(reversed(names)), 123, DEFAULT_SCHEDULE_SEED, domain="timed"
    )
    warmup = balanced_interleaved_schedule(
        names, 123, DEFAULT_SCHEDULE_SEED, domain="warmup"
    )
    other_seed = balanced_interleaved_schedule(
        names, 123, DEFAULT_SCHEDULE_SEED + 1, domain="timed"
    )

    assert first == second
    assert Counter(first) == {name: 123 for name in names}
    assert len(first) == 369
    assert first != warmup
    assert first != other_seed
    np.testing.assert_array_equal(observed_global_draw, expected_global_draw)


def test_full_injected_timer_math_warmup_exclusion_quantiles_and_ratio() -> None:
    config = _config()
    names = ("proposed", "fast", "slow")
    timed_schedule = balanced_interleaved_schedule(
        names,
        config.timed_calls_per_arm,
        config.schedule_seed,
        domain="timed",
    )
    timed_occurrences = Counter()
    durations = []
    for arm_name in timed_schedule:
        timed_occurrences[arm_name] += 1
        if arm_name == "fast":
            durations.append(timed_occurrences[arm_name])
        elif arm_name == "slow":
            durations.append(1_000)
        else:
            durations.append(2_000)
    timer = SequenceTimer(_timer_values(durations))
    callback_counts = Counter()

    def callback(name: str):
        def run():
            assert torch.is_inference_mode_enabled()
            callback_counts[name] += 1
            return {"score": np.float64(0.5)}

        return run

    arms = tuple(LatencyArm(name, callback(name)) for name in names)
    result = benchmark_cpu_latency(
        arms,
        config,
        proposed_arm="proposed",
        eligible_comparators=("slow", "fast"),
        timer_ns=timer,
        timer_id="deterministic-sequence-timer-v1",
    )
    summaries = {item.arm_name: item for item in result.summaries}

    assert callback_counts == {
        name: config.warmup_calls_per_arm + config.timed_calls_per_arm
        for name in names
    }
    assert timer.call_count == 2 * len(timed_schedule)
    assert len(result.observations) == 3_000
    assert tuple(item.schedule_position for item in result.observations) == tuple(
        range(3_000)
    )
    assert tuple(item.arm_name for item in result.observations) == timed_schedule
    assert tuple(item.duration_ns for item in result.observations) == tuple(durations)
    assert summaries["fast"].median_ns == 501
    assert summaries["fast"].p95_ns == 951
    assert summaries["slow"].median_ns == 1_000
    assert summaries["proposed"].median_ns == 2_000
    assert result.ratio.fastest_comparator_arm == "fast"
    assert result.ratio.eligible_comparators == ("fast", "slow")
    assert result.ratio.proposed_to_fastest_comparator_ratio == pytest.approx(
        2_000 / 501
    )
    assert result.environment.backend == BACKEND_ID
    assert result.environment.precision == "float32"
    assert result.environment.timer_id == "deterministic-sequence-timer-v1"
    assert result.environment.torch_intraop_threads == torch.get_num_threads()
    assert result.environment.torch_interop_threads == torch.get_num_interop_threads()
    assert result.environment.python_version
    assert result.environment.numpy_version == np.__version__
    assert result.environment.torch_version == str(torch.__version__)
    json.dumps(result.to_dict(), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        result.schedule_seed = 1  # type: ignore[misc]


def test_conservative_order_statistic_is_higher_without_interpolation() -> None:
    assert conservative_order_statistic([1, 2, 3, 4], 0.5) == 3
    assert conservative_order_statistic(list(range(1, 1001)), 0.95) == 951
    assert conservative_order_statistic([9, 1, 5], 0.0) == 1
    assert conservative_order_statistic([9, 1, 5], 1.0) == 9
    with pytest.raises(ValueError, match="nonempty"):
        conservative_order_statistic([], 0.5)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        conservative_order_statistic([1, 2], 1.1)


def test_minimum_warmups_and_timed_calls_are_enforced() -> None:
    assert MIN_WARMUP_CALLS_PER_ARM == 100
    assert MIN_TIMED_CALLS_PER_ARM == 1_000
    with pytest.raises(ValueError, match="warmup_calls_per_arm must be at least 100"):
        _config(warmup_calls_per_arm=99)
    with pytest.raises(ValueError, match="timed_calls_per_arm must be at least 1000"):
        _config(timed_calls_per_arm=999)


def test_fixed_precision_backend_and_gpu_declarations_fail_closed() -> None:
    with pytest.raises(ValueError, match="float32.*float64"):
        _config(precision="float16")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="frozen CPU backend"):
        _config(backend="onnx_cpu")
    with pytest.raises(ValueError, match="Physical GPU declaration 2"):
        _config(physical_gpu_ids=(2,))
    with pytest.raises(ValueError, match="Multi-GPU"):
        _config(physical_gpu_ids=(0, 1))
    with pytest.raises(ValueError, match="CPU-only"):
        _config(physical_gpu_ids=(0,))


def test_arm_uniqueness_sync_and_comparator_contracts() -> None:
    config = _config()
    with pytest.raises(ValueError, match="At least two"):
        benchmark_cpu_latency(
            (LatencyArm("only", lambda: 1.0),),
            config,
            proposed_arm="only",
            eligible_comparators=("missing",),
        )
    with pytest.raises(ValueError, match="unique"):
        benchmark_cpu_latency(
            (LatencyArm("same", lambda: 1.0), LatencyArm("same", lambda: 2.0)),
            config,
            proposed_arm="same",
            eligible_comparators=("other",),
        )

    async def async_callback():
        return 1.0

    with pytest.raises(TypeError, match="async callbacks"):
        LatencyArm("async", async_callback)
    with pytest.raises(ValueError, match="excluded"):
        benchmark_cpu_latency(
            _safe_arms(),
            config,
            proposed_arm="proposed",
            eligible_comparators=("proposed", "reference"),
        )
    with pytest.raises(ValueError, match="missing"):
        benchmark_cpu_latency(
            _safe_arms(),
            config,
            proposed_arm="proposed",
            eligible_comparators=("absent",),
        )
    with pytest.raises(ValueError, match="At least one eligible"):
        benchmark_cpu_latency(
            _safe_arms(),
            config,
            proposed_arm="proposed",
            eligible_comparators=(),
        )


@pytest.mark.parametrize(
    ("bad_callback", "exception", "match"),
    [
        (
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            RuntimeError,
            "failed during warmup",
        ),
        (lambda: float("nan"), ValueError, "non-finite scalar"),
        (lambda: None, TypeError, "real score"),
        (lambda: {"score": np.asarray([0.2, np.inf])}, ValueError, "non-finite"),
        (lambda: torch.empty(1, device="meta"), ValueError, "non-CPU tensor"),
    ],
)
def test_failed_missing_nonfinite_and_non_cpu_results_are_rejected(
    bad_callback, exception, match: str
) -> None:
    arms = (LatencyArm("bad", bad_callback), LatencyArm("safe", lambda: 0.2))
    with pytest.raises(exception, match=match):
        benchmark_cpu_latency(
            arms,
            _config(),
            proposed_arm="bad",
            eligible_comparators=("safe",),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_tensor_result_is_rejected_when_cuda_is_available() -> None:
    arms = (
        LatencyArm("cuda", lambda: torch.ones(1, device="cuda")),
        LatencyArm("safe", lambda: 0.2),
    )
    with pytest.raises(ValueError, match="non-CPU tensor"):
        benchmark_cpu_latency(
            arms,
            _config(),
            proposed_arm="cuda",
            eligible_comparators=("safe",),
        )


def test_timer_nonmonotonicity_and_custom_timer_identity_are_rejected() -> None:
    with pytest.raises(ValueError, match="explicit timer_id"):
        benchmark_cpu_latency(
            _safe_arms(),
            _config(),
            proposed_arm="proposed",
            eligible_comparators=("reference",),
            timer_ns=SequenceTimer([1, 2]),
        )
    with pytest.raises(RuntimeError, match="within a timed call"):
        benchmark_cpu_latency(
            _safe_arms(),
            _config(),
            proposed_arm="proposed",
            eligible_comparators=("reference",),
            timer_ns=SequenceTimer([10, 9]),
            timer_id="backward-test-timer",
        )
    with pytest.raises(RuntimeError, match="across timed calls"):
        benchmark_cpu_latency(
            _safe_arms(),
            _config(),
            proposed_arm="proposed",
            eligible_comparators=("reference",),
            timer_ns=SequenceTimer([10, 20, 15, 25]),
            timer_id="regressing-test-timer",
        )
    with pytest.raises(TypeError, match="integer nanoseconds"):
        benchmark_cpu_latency(
            _safe_arms(),
            _config(),
            proposed_arm="proposed",
            eligible_comparators=("reference",),
            timer_ns=SequenceTimer([10.5]),
            timer_id="float-test-timer",
        )


def test_thread_and_affinity_mismatch_fail_without_mutation(monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="intra-op thread mismatch"):
        benchmark_cpu_latency(
            _safe_arms(),
            _config(required_intraop_threads=torch.get_num_threads() + 1),
            proposed_arm="proposed",
            eligible_comparators=("reference",),
        )
    with pytest.raises(RuntimeError, match="inter-op thread mismatch"):
        benchmark_cpu_latency(
            _safe_arms(),
            _config(required_interop_threads=torch.get_num_interop_threads() + 1),
            proposed_arm="proposed",
            eligible_comparators=("reference",),
        )

    monkeypatch.setattr(latency_module.os, "sched_getaffinity", lambda pid: {0, 1})
    with pytest.raises(RuntimeError, match="affinity does not match"):
        benchmark_cpu_latency(
            _safe_arms(),
            _config(expected_cpu_affinity=(0,)),
            proposed_arm="proposed",
            eligible_comparators=("reference",),
        )


def test_schedule_imbalance_is_detected_before_callbacks(monkeypatch) -> None:
    calls = 0

    def callback():
        nonlocal calls
        calls += 1
        return 0.5

    def imbalanced(arm_names, calls_per_arm, schedule_seed, *, domain):
        del schedule_seed, domain
        return tuple([arm_names[0]] * (len(arm_names) * calls_per_arm))

    monkeypatch.setattr(latency_module, "balanced_interleaved_schedule", imbalanced)
    with pytest.raises(RuntimeError, match="not exactly balanced"):
        benchmark_cpu_latency(
            (LatencyArm("a", callback), LatencyArm("b", callback)),
            _config(),
            proposed_arm="a",
            eligible_comparators=("b",),
        )
    assert calls == 0


def test_immutable_input_fingerprint_detects_callback_mutation() -> None:
    mutable_input: list[int] = []

    def mutating_callback():
        mutable_input.append(1)
        return 0.5

    arms = (
        LatencyArm("mutating", mutating_callback),
        LatencyArm("safe", lambda: 0.4),
    )
    with pytest.raises(RuntimeError, match="mutated the immutable input fingerprint"):
        benchmark_cpu_latency(
            arms,
            _config(),
            proposed_arm="mutating",
            eligible_comparators=("safe",),
            immutable_input_fingerprint=lambda: repr(tuple(mutable_input)),
        )


def test_sync_callback_returning_awaitable_is_rejected_and_closed() -> None:
    async def value():
        return 0.5

    arms = (
        LatencyArm("wrapped_async", lambda: value()),
        LatencyArm("safe", lambda: 0.4),
    )
    with pytest.raises(TypeError, match="returned an async result"):
        benchmark_cpu_latency(
            arms,
            _config(),
            proposed_arm="wrapped_async",
            eligible_comparators=("safe",),
        )


def test_latency_module_has_no_io_affinity_or_thread_mutation_dependency() -> None:
    source = inspect.getsource(latency_module)
    tree = ast.parse(source)
    imported_modules: list[str] = []
    called_names: list[str] = []
    attribute_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.append(node.module or "")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called_names.append(node.func.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attribute_calls.append(node.func.attr)

    assert not any(
        token in module
        for module in imported_modules
        for token in ("pathlib", "model", "trainer", "experiment_runner")
    )
    assert "open" not in called_names
    assert "write" not in attribute_calls
    assert "set_num_threads" not in attribute_calls
    assert "set_num_interop_threads" not in attribute_calls
    assert "sched_setaffinity" not in attribute_calls
