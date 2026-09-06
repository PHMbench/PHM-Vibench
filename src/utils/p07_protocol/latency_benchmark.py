"""Frozen in-memory CPU latency primitive for the P07 E11 protocol.

Callers provide batch-one end-to-end callbacks that already include
preprocessing, gate/export-or-path execution, classifier evaluation, and score
production.  This module validates and times those callbacks; it never writes
files, mutates CPU affinity, sets Torch thread counts, bootstraps ratios, or
makes a claim decision.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import os
import platform
import time
from collections import Counter
from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable, Final, Literal, Mapping, Sequence, cast

import numpy as np
import torch


Precision = Literal["float32", "float64"]
ScheduleDomain = Literal["warmup", "timed"]
Timer = Callable[[], int]
FingerprintHook = Callable[[], str]

SCHEMA_VERSION: Final[int] = 1
PROTOCOL_ID: Final[str] = "P07-E11-CPU-LATENCY-v1"
BACKEND_ID: Final[str] = "torch_cpu_eager"
DEFAULT_SCHEDULE_SEED: Final[int] = 2_026_080_111
MIN_WARMUP_CALLS_PER_ARM: Final[int] = 100
MIN_TIMED_CALLS_PER_ARM: Final[int] = 1_000
_DEFAULT_TIMER: Final[Timer] = time.perf_counter_ns


def _require_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be nonempty stripped text.")
    return value


def _require_integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, not boolean.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _require_seed(value: Any) -> int:
    seed = _require_integer(value, name="schedule_seed")
    if seed >= 2**64:
        raise ValueError("schedule_seed must be in [0, 2**64).")
    return seed


def _is_async_callable(callback: Any) -> bool:
    if inspect.iscoroutinefunction(callback) or inspect.isasyncgenfunction(callback):
        return True
    call_method = getattr(callback, "__call__", None)
    return bool(
        call_method is not None
        and (
            inspect.iscoroutinefunction(call_method)
            or inspect.isasyncgenfunction(call_method)
        )
    )


def _validate_sync_callable(callback: Any, *, name: str) -> None:
    if not callable(callback):
        raise TypeError(f"{name} must be callable.")
    if _is_async_callable(callback):
        raise TypeError(f"{name} must be synchronous; async callbacks are forbidden.")


@dataclass(frozen=True, slots=True)
class LatencyArm:
    """Named batch-one end-to-end callback supplied by the caller."""

    arm_name: str
    callback: Callable[[], Any]

    def __post_init__(self) -> None:
        _require_text(self.arm_name, name="arm_name")
        _validate_sync_callable(self.callback, name=f"callback for {self.arm_name!r}")


@dataclass(frozen=True, slots=True)
class LatencyBenchmarkConfig:
    """Frozen execution metadata and caller-preconfigured CPU contract."""

    precision: Precision
    backend: str
    required_intraop_threads: int
    required_interop_threads: int
    warmup_calls_per_arm: int = MIN_WARMUP_CALLS_PER_ARM
    timed_calls_per_arm: int = MIN_TIMED_CALLS_PER_ARM
    schedule_seed: int = DEFAULT_SCHEDULE_SEED
    expected_cpu_affinity: tuple[int, ...] | None = None
    physical_gpu_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.precision not in {"float32", "float64"}:
            raise ValueError("precision must be exactly 'float32' or 'float64'.")
        if self.backend != BACKEND_ID:
            raise ValueError(f"backend must be the frozen CPU backend {BACKEND_ID!r}.")
        _require_integer(
            self.required_intraop_threads,
            name="required_intraop_threads",
            minimum=1,
        )
        _require_integer(
            self.required_interop_threads,
            name="required_interop_threads",
            minimum=1,
        )
        _require_integer(
            self.warmup_calls_per_arm,
            name="warmup_calls_per_arm",
            minimum=MIN_WARMUP_CALLS_PER_ARM,
        )
        _require_integer(
            self.timed_calls_per_arm,
            name="timed_calls_per_arm",
            minimum=MIN_TIMED_CALLS_PER_ARM,
        )
        _require_seed(self.schedule_seed)
        if not isinstance(self.physical_gpu_ids, tuple):
            raise TypeError("physical_gpu_ids must be an immutable tuple.")
        gpu_ids = tuple(
            _require_integer(item, name="physical_gpu_id")
            for item in self.physical_gpu_ids
        )
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("physical_gpu_ids must be unique.")
        if len(gpu_ids) > 1:
            raise ValueError("Multi-GPU declarations are forbidden for E11 CPU latency.")
        if 2 in gpu_ids:
            raise ValueError("Physical GPU declaration 2 is forbidden for E11 CPU latency.")
        if gpu_ids:
            raise ValueError("E11 primary latency is CPU-only; physical_gpu_ids must be empty.")
        if self.expected_cpu_affinity is not None:
            if not isinstance(self.expected_cpu_affinity, tuple):
                raise TypeError("expected_cpu_affinity must be an immutable tuple or None.")
            if not self.expected_cpu_affinity:
                raise ValueError("expected_cpu_affinity must not be empty when supplied.")
            affinity = tuple(
                _require_integer(item, name="expected affinity CPU")
                for item in self.expected_cpu_affinity
            )
            if affinity != tuple(sorted(set(affinity))):
                raise ValueError(
                    "expected_cpu_affinity must be sorted and contain unique CPU IDs."
                )


@dataclass(frozen=True, slots=True)
class LatencyObservation:
    schedule_position: int
    arm_name: str
    duration_ns: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schedule_position": self.schedule_position,
            "arm_name": self.arm_name,
            "duration_ns": self.duration_ns,
        }


@dataclass(frozen=True, slots=True)
class ArmLatencySummary:
    arm_name: str
    timed_calls: int
    median_ns: int
    p95_ns: int
    minimum_ns: int
    maximum_ns: int

    def to_dict(self) -> dict[str, object]:
        return {
            "arm_name": self.arm_name,
            "timed_calls": self.timed_calls,
            "median_ns": self.median_ns,
            "p95_ns": self.p95_ns,
            "minimum_ns": self.minimum_ns,
            "maximum_ns": self.maximum_ns,
            "quantile_rule": "sorted[ceil(q*(n-1))] (higher order statistic)",
        }


@dataclass(frozen=True, slots=True)
class LatencyRatio:
    proposed_arm: str
    eligible_comparators: tuple[str, ...]
    fastest_comparator_arm: str
    proposed_median_ns: int
    fastest_comparator_median_ns: int
    proposed_to_fastest_comparator_ratio: float

    def to_dict(self) -> dict[str, object]:
        return {
            "proposed_arm": self.proposed_arm,
            "eligible_comparators": list(self.eligible_comparators),
            "fastest_comparator_arm": self.fastest_comparator_arm,
            "proposed_median_ns": self.proposed_median_ns,
            "fastest_comparator_median_ns": self.fastest_comparator_median_ns,
            "proposed_to_fastest_comparator_ratio": (
                self.proposed_to_fastest_comparator_ratio
            ),
            "inference": "descriptive ratio only; no bootstrap or claim decision",
        }


@dataclass(frozen=True, slots=True)
class LatencyEnvironment:
    precision: Precision
    backend: str
    timer_id: str
    python_version: str
    numpy_version: str
    torch_version: str
    operating_system: str
    platform_release: str
    machine: str
    cpu_model: str
    logical_cpu_count: int | None
    torch_intraop_threads: int
    torch_interop_threads: int
    cpu_affinity: tuple[int, ...] | None
    physical_gpu_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "precision": self.precision,
            "backend": self.backend,
            "timer_id": self.timer_id,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "torch_version": self.torch_version,
            "operating_system": self.operating_system,
            "platform_release": self.platform_release,
            "machine": self.machine,
            "cpu_model": self.cpu_model,
            "logical_cpu_count": self.logical_cpu_count,
            "torch_intraop_threads": self.torch_intraop_threads,
            "torch_interop_threads": self.torch_interop_threads,
            "cpu_affinity": (
                None if self.cpu_affinity is None else list(self.cpu_affinity)
            ),
            "physical_gpu_ids": list(self.physical_gpu_ids),
        }


@dataclass(frozen=True, slots=True)
class LatencyBenchmarkResult:
    warmup_calls_per_arm: int
    timed_calls_per_arm: int
    schedule_seed: int
    warmup_domain_seed: int
    timed_domain_seed: int
    warmup_schedule: tuple[str, ...]
    observations: tuple[LatencyObservation, ...]
    summaries: tuple[ArmLatencySummary, ...]
    ratio: LatencyRatio
    environment: LatencyEnvironment

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "batch_size": 1,
            "callback_scope": (
                "preprocessing + gate/export-or-path execution + classifier + score"
            ),
            "device": "cpu",
            "warmup_calls_per_arm": self.warmup_calls_per_arm,
            "timed_calls_per_arm": self.timed_calls_per_arm,
            "schedule_seed": self.schedule_seed,
            "warmup_domain_seed": self.warmup_domain_seed,
            "timed_domain_seed": self.timed_domain_seed,
            "warmup_schedule": list(self.warmup_schedule),
            "raw_observations": [item.to_dict() for item in self.observations],
            "summaries": [item.to_dict() for item in self.summaries],
            "ratio": self.ratio.to_dict(),
            "environment": self.environment.to_dict(),
        }


def _domain_separated_seed(schedule_seed: int, domain: ScheduleDomain) -> int:
    digest = hashlib.sha256(
        f"{PROTOCOL_ID}|schedule|{schedule_seed}|{domain}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _validate_arm_names(arm_names: Any) -> tuple[str, ...]:
    if isinstance(arm_names, (str, bytes)) or not isinstance(arm_names, Sequence):
        raise TypeError("arm_names must be a sequence of names.")
    names = tuple(_require_text(item, name="arm_name") for item in arm_names)
    if len(names) < 2:
        raise ValueError("At least two latency arms are required.")
    if len(set(names)) != len(names):
        raise ValueError("Latency arm names must be unique.")
    return tuple(sorted(names))


def balanced_interleaved_schedule(
    arm_names: Sequence[str],
    calls_per_arm: int,
    schedule_seed: int,
    *,
    domain: ScheduleDomain,
) -> tuple[str, ...]:
    """Build a deterministic, domain-separated, exactly balanced schedule."""

    names = _validate_arm_names(arm_names)
    calls = _require_integer(calls_per_arm, name="calls_per_arm", minimum=1)
    seed = _require_seed(schedule_seed)
    if domain not in {"warmup", "timed"}:
        raise ValueError("domain must be exactly 'warmup' or 'timed'.")
    schedule = np.asarray(
        [name for name in names for _ in range(calls)], dtype=object
    )
    generator = np.random.default_rng(_domain_separated_seed(seed, domain))
    generator.shuffle(schedule)
    return tuple(str(item) for item in schedule.tolist())


def _validate_schedule(
    schedule: Sequence[str],
    arm_names: Sequence[str],
    calls_per_arm: int,
    *,
    phase: str,
) -> None:
    expected_length = len(arm_names) * calls_per_arm
    if len(schedule) != expected_length:
        raise RuntimeError(f"{phase} schedule length is imbalanced or incomplete.")
    counts = Counter(schedule)
    expected = {name: calls_per_arm for name in arm_names}
    if dict(counts) != expected:
        raise RuntimeError(f"{phase} schedule is not exactly balanced across arms.")


def _validate_arms(arms: Any) -> tuple[LatencyArm, ...]:
    if isinstance(arms, (str, bytes)) or not isinstance(arms, Sequence):
        raise TypeError("arms must be a sequence of LatencyArm objects.")
    result = tuple(arms)
    if not all(isinstance(item, LatencyArm) for item in result):
        raise TypeError("Every arm must be a LatencyArm.")
    _validate_arm_names([item.arm_name for item in result])
    return tuple(sorted(result, key=lambda item: item.arm_name))


def _current_cpu_affinity() -> tuple[int, ...] | None:
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return None
    try:
        return tuple(sorted(int(item) for item in getter(0)))
    except OSError as error:
        raise RuntimeError("Could not read CPU affinity without mutation.") from error


def _verify_runtime_contract(config: LatencyBenchmarkConfig) -> tuple[int, int]:
    intraop = int(torch.get_num_threads())
    interop = int(torch.get_num_interop_threads())
    if intraop != config.required_intraop_threads:
        raise RuntimeError(
            "Torch intra-op thread mismatch: caller must freeze the required value "
            "before benchmarking."
        )
    if interop != config.required_interop_threads:
        raise RuntimeError(
            "Torch inter-op thread mismatch: caller must freeze the required value "
            "before benchmarking."
        )
    if config.expected_cpu_affinity is not None:
        actual = _current_cpu_affinity()
        if actual is None:
            raise RuntimeError("CPU affinity verification was requested but is unavailable.")
        if actual != config.expected_cpu_affinity:
            raise RuntimeError("CPU affinity does not match the caller-frozen expected set.")
    return intraop, interop


def _read_fingerprint(hook: FingerprintHook, *, location: str) -> str:
    try:
        value = hook()
    except Exception as error:
        raise RuntimeError(f"Immutable input fingerprint hook failed at {location}.") from error
    if not isinstance(value, str) or not value:
        raise ValueError("Immutable input fingerprint hook must return nonempty text.")
    return value


def _validate_output(value: Any, *, location: str) -> None:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{location} returned a non-CPU tensor on {value.device}.")
        if value.numel() == 0:
            raise ValueError(f"{location} returned an empty tensor.")
        if value.is_complex():
            raise TypeError(f"{location} returned a complex tensor score.")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{location} returned a non-finite tensor value.")
        return
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError(f"{location} returned an empty NumPy array.")
        if value.dtype.kind not in "iuf":
            raise TypeError(f"{location} returned a non-real NumPy score array.")
        if not bool(np.isfinite(value).all()):
            raise ValueError(f"{location} returned a non-finite NumPy value.")
        return
    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"{location} returned an empty result mapping.")
        for key, item in value.items():
            _require_text(key, name="result mapping key")
            _validate_output(item, location=f"{location}.{key}")
        return
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError(f"{location} returned an empty result sequence.")
        for index, item in enumerate(value):
            _validate_output(item, location=f"{location}[{index}]")
        return
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{location} must return a real score or nested real scores.")
    if not math.isfinite(float(value)):
        raise ValueError(f"{location} returned a non-finite scalar value.")


def _close_awaitable(value: Any) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()


def _invoke_arm(arm: LatencyArm, *, phase: str, position: int) -> Any:
    try:
        result = arm.callback()
    except Exception as error:
        raise RuntimeError(
            f"Arm {arm.arm_name!r} failed during {phase} position {position}."
        ) from error
    if inspect.isawaitable(result) or inspect.isasyncgen(result):
        _close_awaitable(result)
        raise TypeError(
            f"Arm {arm.arm_name!r} returned an async result during {phase}; async is forbidden."
        )
    return result


def _timer_value(timer_ns: Timer, *, location: str) -> int:
    try:
        value = timer_ns()
    except Exception as error:
        raise RuntimeError(f"Timer failed at {location}.") from error
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("timer_ns must return integer nanoseconds.")
    result = int(value)
    if result < 0:
        raise ValueError("timer_ns must return nonnegative nanoseconds.")
    return result


def conservative_order_statistic(values: Sequence[int], quantile: float) -> int:
    """Return NumPy's ``method='higher'`` order statistic without interpolation."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("values must be a sequence of integer durations.")
    durations = tuple(
        _require_integer(item, name="duration_ns", minimum=1) for item in values
    )
    if not durations:
        raise ValueError("values must be nonempty.")
    if isinstance(quantile, bool) or not isinstance(quantile, Real):
        raise TypeError("quantile must be a real number.")
    probability = float(quantile)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("quantile must lie in [0,1].")
    ordered = sorted(durations)
    index = int(math.ceil(probability * (len(ordered) - 1)))
    return int(ordered[index])


def _summaries(
    arm_names: Sequence[str], observations: Sequence[LatencyObservation]
) -> tuple[ArmLatencySummary, ...]:
    grouped: dict[str, list[int]] = {name: [] for name in arm_names}
    for observation in observations:
        grouped[observation.arm_name].append(observation.duration_ns)
    result = []
    for name in arm_names:
        durations = grouped[name]
        result.append(
            ArmLatencySummary(
                arm_name=name,
                timed_calls=len(durations),
                median_ns=conservative_order_statistic(durations, 0.50),
                p95_ns=conservative_order_statistic(durations, 0.95),
                minimum_ns=min(durations),
                maximum_ns=max(durations),
            )
        )
    return tuple(result)


def _validate_comparators(
    proposed_arm: Any,
    eligible_comparators: Any,
    arm_names: Sequence[str],
) -> tuple[str, tuple[str, ...]]:
    proposed = _require_text(proposed_arm, name="proposed_arm")
    if proposed not in arm_names:
        raise ValueError("proposed_arm is absent from the supplied latency arms.")
    if isinstance(eligible_comparators, (str, bytes)) or not isinstance(
        eligible_comparators, Sequence
    ):
        raise TypeError("eligible_comparators must be a sequence of arm names.")
    comparators = tuple(
        _require_text(item, name="eligible comparator")
        for item in eligible_comparators
    )
    if not comparators:
        raise ValueError("At least one eligible comparator is required.")
    if len(set(comparators)) != len(comparators):
        raise ValueError("eligible_comparators must be unique.")
    if proposed in comparators:
        raise ValueError("The proposed arm must be excluded from eligible_comparators.")
    missing = sorted(set(comparators).difference(arm_names))
    if missing:
        raise ValueError(f"Eligible comparator arms are missing: {missing}.")
    return proposed, tuple(sorted(comparators))


def _environment(
    config: LatencyBenchmarkConfig,
    *,
    timer_id: str,
    affinity: tuple[int, ...] | None,
) -> LatencyEnvironment:
    return LatencyEnvironment(
        precision=config.precision,
        backend=config.backend,
        timer_id=timer_id,
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        torch_version=str(torch.__version__),
        operating_system=platform.system() or "unknown",
        platform_release=platform.release() or "unknown",
        machine=platform.machine() or "unknown",
        cpu_model=platform.processor() or platform.machine() or "unknown",
        logical_cpu_count=os.cpu_count(),
        torch_intraop_threads=int(torch.get_num_threads()),
        torch_interop_threads=int(torch.get_num_interop_threads()),
        cpu_affinity=affinity,
        physical_gpu_ids=config.physical_gpu_ids,
    )


def benchmark_cpu_latency(
    arms: Sequence[LatencyArm],
    config: LatencyBenchmarkConfig,
    *,
    proposed_arm: str,
    eligible_comparators: Sequence[str],
    timer_ns: Timer = _DEFAULT_TIMER,
    timer_id: str | None = None,
    immutable_input_fingerprint: FingerprintHook | None = None,
) -> LatencyBenchmarkResult:
    """Benchmark caller-owned end-to-end callbacks under the frozen E11 contract."""

    if not isinstance(config, LatencyBenchmarkConfig):
        raise TypeError("config must be a LatencyBenchmarkConfig.")
    checked_arms = _validate_arms(arms)
    arm_names = tuple(item.arm_name for item in checked_arms)
    arm_by_name = {item.arm_name: item for item in checked_arms}
    proposed, comparators = _validate_comparators(
        proposed_arm, eligible_comparators, arm_names
    )
    _validate_sync_callable(timer_ns, name="timer_ns")
    if timer_id is None:
        if timer_ns is not _DEFAULT_TIMER:
            raise ValueError("A custom timer_ns requires an explicit timer_id.")
        checked_timer_id = "time.perf_counter_ns"
    else:
        checked_timer_id = _require_text(timer_id, name="timer_id")
    if immutable_input_fingerprint is not None:
        _validate_sync_callable(
            immutable_input_fingerprint, name="immutable_input_fingerprint"
        )

    _verify_runtime_contract(config)
    affinity = _current_cpu_affinity()
    baseline_fingerprint = (
        None
        if immutable_input_fingerprint is None
        else _read_fingerprint(
            immutable_input_fingerprint, location="benchmark start"
        )
    )
    warmup_schedule = balanced_interleaved_schedule(
        arm_names,
        config.warmup_calls_per_arm,
        config.schedule_seed,
        domain="warmup",
    )
    timed_schedule = balanced_interleaved_schedule(
        arm_names,
        config.timed_calls_per_arm,
        config.schedule_seed,
        domain="timed",
    )
    _validate_schedule(
        warmup_schedule,
        arm_names,
        config.warmup_calls_per_arm,
        phase="warmup",
    )
    _validate_schedule(
        timed_schedule,
        arm_names,
        config.timed_calls_per_arm,
        phase="timed",
    )

    observations: list[LatencyObservation] = []
    last_timer_value: int | None = None
    with torch.inference_mode():
        for position, arm_name in enumerate(warmup_schedule):
            _verify_runtime_contract(config)
            output = _invoke_arm(
                arm_by_name[arm_name], phase="warmup", position=position
            )
            _validate_output(
                output, location=f"Arm {arm_name!r} warmup position {position}"
            )
            if immutable_input_fingerprint is not None:
                current = _read_fingerprint(
                    immutable_input_fingerprint,
                    location=f"after warmup position {position}",
                )
                if current != baseline_fingerprint:
                    raise RuntimeError(
                        f"Arm {arm_name!r} mutated the immutable input fingerprint "
                        f"during warmup position {position}."
                    )
            _verify_runtime_contract(config)

        for position, arm_name in enumerate(timed_schedule):
            _verify_runtime_contract(config)
            start_ns = _timer_value(
                timer_ns, location=f"timed position {position} start"
            )
            if last_timer_value is not None and start_ns < last_timer_value:
                raise RuntimeError("timer_ns was nonmonotonic across timed calls.")
            output = _invoke_arm(
                arm_by_name[arm_name], phase="timed", position=position
            )
            end_ns = _timer_value(
                timer_ns, location=f"timed position {position} end"
            )
            if end_ns <= start_ns:
                raise RuntimeError(
                    "timer_ns was nonmonotonic or non-advancing within a timed call."
                )
            last_timer_value = end_ns
            _validate_output(
                output, location=f"Arm {arm_name!r} timed position {position}"
            )
            if immutable_input_fingerprint is not None:
                current = _read_fingerprint(
                    immutable_input_fingerprint,
                    location=f"after timed position {position}",
                )
                if current != baseline_fingerprint:
                    raise RuntimeError(
                        f"Arm {arm_name!r} mutated the immutable input fingerprint "
                        f"during timed position {position}."
                    )
            _verify_runtime_contract(config)
            observations.append(
                LatencyObservation(
                    schedule_position=position,
                    arm_name=arm_name,
                    duration_ns=end_ns - start_ns,
                )
            )

    _validate_schedule(
        [item.arm_name for item in observations],
        arm_names,
        config.timed_calls_per_arm,
        phase="recorded timed",
    )
    summaries = _summaries(arm_names, observations)
    summary_by_name = {item.arm_name: item for item in summaries}
    fastest_comparator = min(
        comparators,
        key=lambda name: (summary_by_name[name].median_ns, name),
    )
    proposed_median = summary_by_name[proposed].median_ns
    fastest_median = summary_by_name[fastest_comparator].median_ns
    ratio_value = proposed_median / fastest_median
    if not math.isfinite(ratio_value) or ratio_value <= 0.0:
        raise RuntimeError("Latency ratio must be finite and positive.")
    ratio = LatencyRatio(
        proposed_arm=proposed,
        eligible_comparators=comparators,
        fastest_comparator_arm=fastest_comparator,
        proposed_median_ns=proposed_median,
        fastest_comparator_median_ns=fastest_median,
        proposed_to_fastest_comparator_ratio=float(ratio_value),
    )
    environment = _environment(
        config, timer_id=checked_timer_id, affinity=affinity
    )
    return LatencyBenchmarkResult(
        warmup_calls_per_arm=config.warmup_calls_per_arm,
        timed_calls_per_arm=config.timed_calls_per_arm,
        schedule_seed=config.schedule_seed,
        warmup_domain_seed=_domain_separated_seed(config.schedule_seed, "warmup"),
        timed_domain_seed=_domain_separated_seed(config.schedule_seed, "timed"),
        warmup_schedule=warmup_schedule,
        observations=tuple(observations),
        summaries=summaries,
        ratio=ratio,
        environment=environment,
    )


__all__ = [
    "ArmLatencySummary",
    "BACKEND_ID",
    "DEFAULT_SCHEDULE_SEED",
    "LatencyArm",
    "LatencyBenchmarkConfig",
    "LatencyBenchmarkResult",
    "LatencyEnvironment",
    "LatencyObservation",
    "LatencyRatio",
    "MIN_TIMED_CALLS_PER_ARM",
    "MIN_WARMUP_CALLS_PER_ARM",
    "PROTOCOL_ID",
    "balanced_interleaved_schedule",
    "benchmark_cpu_latency",
    "conservative_order_statistic",
]
