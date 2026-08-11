"""Evaluate P04 anonymous roles and prespecified expert interventions.

The evaluator is deliberately offline: it consumes diagnostic arrays already
produced on the frozen identification and intervention partitions, and it never
loads a model or reruns its router.  Its preferred unified input is a non-pickled
``.npz`` archive with schema ``p04.mechanism-evaluator-input.v1`` and arrays
``sample_id[N]``, ``partition[N]``, ``label[N]``, ``mechanism[N]``,
``diagnosis[N]``, ``nuisance_cell[N]``, ``draw[N]``, ``logits[N,C]``,
``routing_weights[N,4]``, ``expert_features[N,4,H]``,
``expert_logits[N,4,C]``, and ``deleted_logits[N,4,C]``.  Optional
``fixed_mass_swap_logits[N,4,4,C]`` indexes matched expert then replacement
expert.  Scalar provenance and ``designated_role_to_expert[4]`` are also
required.  Optional ``blinding_permutation[4]`` has direction
``canonical_expert_index_at_each_blinded_column`` (blinded column to canonical
slot), whereas ``designated_role_to_expert`` has direction
``canonical_constrained_slot_to_blinded_column``; the two are inverse
permutations.  Neither enters anonymous assignment.  Identification and
intervention rows are selected by ``partition``.

For collectors that write one archive per partition, the evaluator also accepts
the following two explicit schemas.

``p04.role-identification-input.v1`` requires scalar ``seed``, ``arm``,
``partition_name='identification'``, ``generator_manifest_sha256`` and
``partition_manifest_sha256``; unique ``sample_ids[N]``;
``expert_features[N,4,H]``; ``routing_weights[N,4]``; and integer
``mechanism_ids[N]``, ``diagnosis_labels[N]``, ``nuisance_cell_ids[N]`` plus
``draw_ids[N]``, ``designated_role_to_expert[4]``, and the canonical assignment
seal/order fields used by the unified schema.  The designated permutation is
used only after the sealed blinded assignment has been independently verified.

``p04.intervention-input.v1`` has the same scalar provenance and grouping
fields with ``partition_name='intervention'``; it replaces features with
``expert_logits[N,4,C]`` and adds integer ``labels[N]``.  Intact logits are
reconstructed as the fixed weighted expert sum.  Surviving deletion weights
are computed algebraically; the router is never recomputed.
``assignment_seal_sha256`` must repeat the verified identification seal digest.

The CLI creates the canonical ``behavior_signatures.json``,
``role_assignment.json``, ``deletion_losses.npz``, and ``metrics.json`` without
overwriting an existing artifact.  No seed result is inferred or imputed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import zipfile
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


IDENTIFICATION_SCHEMA = "p04.role-identification-input.v1"
INTERVENTION_SCHEMA = "p04.intervention-input.v1"
UNIFIED_SCHEMA = "p04.mechanism-evaluator-input.v1"
OUTPUT_SCHEMA_VERSION = "1.0.0"
ASSIGNMENT_SEAL_SCHEMA = "p04.preintervention-assignment-seal.v1"
COLLECTION_PHASE_ORDER = (
    "identification_read_and_forward",
    "response_only_assignment_sealed",
    "intervention_read_and_forward",
)
ROLE_NAMES = (
    "low_frequency",
    "harmonic",
    "impulsive_envelope",
    "aperiodic_residual",
)
NUM_ROLES = 4
SD_INVALID_THRESHOLD = 1.0e-8
SIGNATURE_NORM_INVALID_THRESHOLD = 1.0e-8
DELETION_DENOMINATOR_INVALID_THRESHOLD = 1.0e-6
MASS_MATCH_CALIPER = 0.05
MASS_MATCH_MINIMUM_COVERAGE = 0.80
FROZEN_CELLS_PER_STRATUM = 5
FROZEN_DRAWS_PER_CELL = 8
EVALUATION_CORRECTION_SCHEMA = "p04.evaluation-correction.v1"
EVALUATION_CORRECTION_SCHEMA_VERSION = "1.0.0"
EVALUATION_CORRECTION_ID = "P04-G050-EVAL-C2"
SUPERSEDED_EVALUATOR_SHA256 = (
    "9848399cae54c1941e52cbb40ca31af508cd770199fb11021399fbec826d9950"
)
VERIFICATION_DTYPE = "float32"
FIXED_MASS_RTOL = 1.0e-5
FIXED_MASS_ATOL = 1.0e-6
CORRECTION_DISCOVERY_BOUNDARY = "no_aggregate_or_claim_decision"
CORRECTION_ARMS = ("FULL", "HOMO", "RAND")
CORRECTION_SEEDS = (42, 123, 456, 789, 1024)
CORRECTION_MANIFEST_KEYS = frozenset(
    {
        "schema_id",
        "schema_version",
        "evaluation_correction_id",
        "status",
        "supersedes_evaluator_sha256",
        "evaluator_source_sha256",
        "verification_dtype",
        "fixed_mass_rtol",
        "fixed_mass_atol",
        "estimand_changed",
        "thresholds_changed",
        "discovery_boundary",
        "traces",
    }
)
CORRECTION_TRACE_KEYS = frozenset(
    {"arm", "seed", "trace_sha256", "assignment_seal_sha256"}
)


def _finite_array(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _integer_vector(value: Any, name: str, length: int | None = None) -> np.ndarray:
    numeric = _finite_array(value, name, ndim=1)
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain integers")
    result = numeric.astype(np.int64)
    if length is not None and result.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    return result


def _mechanism_vector(value: Any, length: int) -> np.ndarray:
    """Resolve either frozen mechanism names or numeric indices to 0..3."""
    array = np.asarray(value)
    if array.shape != (length,):
        raise ValueError(f"mechanism must have shape ({length},)")
    if np.issubdtype(array.dtype, np.number):
        return _integer_vector(array, "mechanism", length)
    names = array.astype(str)
    lookup = {name: index for index, name in enumerate(ROLE_NAMES)}
    unexpected = sorted(set(names) - set(lookup))
    if unexpected:
        raise ValueError(f"mechanism contains unknown frozen names: {unexpected}")
    return np.asarray([lookup[name] for name in names], dtype=np.int64)


def _text_scalar(value: Any, name: str) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar string")
    result = str(array.item())
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _integer_scalar(value: Any, name: str) -> int:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar integer")
    raw = array.item()
    if isinstance(raw, (bool, np.bool_)) or int(raw) != raw:
        raise ValueError(f"{name} must be a scalar integer")
    return int(raw)


def _boolean_scalar(value: Any, name: str) -> bool:
    array = np.asarray(value)
    if array.shape != () or not isinstance(array.item(), (bool, np.bool_)):
        raise ValueError(f"{name} must be a scalar boolean")
    return bool(array.item())


def _validate_sha256(value: Any, name: str) -> str:
    digest = _text_scalar(value, name).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name} must be a 64-character SHA-256 digest")
    return digest


def _json_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest")
    if (
        len(value) != 64
        or value != value.lower()
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest")
    return value


def _load_evaluation_correction_manifest(
    path: Path,
    *,
    trace_sha256: str,
    arm: str,
    seed: int,
    assignment_seal_sha256: str,
) -> dict[str, Any]:
    """Validate and bind the registered C2 correction to one frozen trace."""
    try:
        manifest_bytes = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"could not read correction manifest {path}: {exc}") from exc
    correction_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    try:
        payload = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"correction manifest {path} must be UTF-8 JSON text: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("correction manifest must be a JSON object")
    if set(payload) != CORRECTION_MANIFEST_KEYS:
        missing = sorted(CORRECTION_MANIFEST_KEYS - set(payload))
        unexpected = sorted(set(payload) - CORRECTION_MANIFEST_KEYS)
        raise ValueError(
            "correction manifest top-level keys do not match the frozen schema; "
            f"missing={missing}, unexpected={unexpected}"
        )

    expected_scalars: dict[str, Any] = {
        "schema_id": EVALUATION_CORRECTION_SCHEMA,
        "schema_version": EVALUATION_CORRECTION_SCHEMA_VERSION,
        "evaluation_correction_id": EVALUATION_CORRECTION_ID,
        "status": "registered",
        "supersedes_evaluator_sha256": SUPERSEDED_EVALUATOR_SHA256,
        "verification_dtype": VERIFICATION_DTYPE,
        "fixed_mass_rtol": FIXED_MASS_RTOL,
        "fixed_mass_atol": FIXED_MASS_ATOL,
        "estimand_changed": False,
        "thresholds_changed": False,
        "discovery_boundary": CORRECTION_DISCOVERY_BOUNDARY,
    }
    for field, expected in expected_scalars.items():
        actual = payload[field]
        if isinstance(expected, bool):
            matches = actual is expected
        elif isinstance(expected, float):
            matches = (
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and float(actual) == expected
            )
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"correction manifest {field} must equal {expected!r}, got {actual!r}"
            )

    evaluator_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest_evaluator_sha256 = _json_sha256(
        payload["evaluator_source_sha256"], "evaluator_source_sha256"
    )
    if manifest_evaluator_sha256 != evaluator_source_sha256:
        raise ValueError(
            "correction manifest evaluator_source_sha256 does not equal the "
            "current evaluator bytes"
        )
    _json_sha256(
        payload["supersedes_evaluator_sha256"], "supersedes_evaluator_sha256"
    )

    traces = payload["traces"]
    expected_order = [
        (expected_arm, expected_seed)
        for expected_arm in CORRECTION_ARMS
        for expected_seed in CORRECTION_SEEDS
    ]
    if not isinstance(traces, list) or len(traces) != len(expected_order):
        raise ValueError("correction manifest traces must contain exactly 15 records")
    normalized_traces: list[dict[str, Any]] = []
    for index, (record, expected_identity) in enumerate(zip(traces, expected_order)):
        if not isinstance(record, dict) or set(record) != CORRECTION_TRACE_KEYS:
            raise ValueError(
                "each correction manifest trace must have exact keys "
                "{arm, seed, trace_sha256, assignment_seal_sha256}"
            )
        record_arm = record["arm"]
        record_seed = record["seed"]
        if (
            not isinstance(record_arm, str)
            or isinstance(record_seed, bool)
            or not isinstance(record_seed, int)
        ):
            raise ValueError(f"correction manifest trace {index} has invalid arm/seed")
        if (record_arm, record_seed) != expected_identity:
            raise ValueError(
                "correction manifest traces must be ordered FULL,HOMO,RAND and "
                "seeds 42,123,456,789,1024"
            )
        normalized_traces.append(
            {
                "arm": record_arm,
                "seed": record_seed,
                "trace_sha256": _json_sha256(
                    record["trace_sha256"], f"traces[{index}].trace_sha256"
                ),
                "assignment_seal_sha256": _json_sha256(
                    record["assignment_seal_sha256"],
                    f"traces[{index}].assignment_seal_sha256",
                ),
            }
        )

    matches = [
        record
        for record in normalized_traces
        if record["arm"] == arm and record["seed"] == seed
    ]
    if len(matches) != 1:
        raise ValueError(
            "correction manifest must contain exactly one record for the current arm/seed"
        )
    current = matches[0]
    if current["trace_sha256"] != trace_sha256:
        raise ValueError(
            "correction manifest trace_sha256 does not match the current input artifact"
        )
    if current["assignment_seal_sha256"] != assignment_seal_sha256:
        raise ValueError(
            "correction manifest assignment_seal_sha256 does not match the "
            "verified current assignment seal"
        )
    return {
        "evaluation_correction_id": EVALUATION_CORRECTION_ID,
        "evaluator_source_sha256": evaluator_source_sha256,
        "supersedes_evaluator_sha256": SUPERSEDED_EVALUATOR_SHA256,
        "correction_manifest_sha256": correction_manifest_sha256,
        "verification_dtype": VERIFICATION_DTYPE,
        "fixed_mass_rtol": FIXED_MASS_RTOL,
        "fixed_mass_atol": FIXED_MASS_ATOL,
    }


def _sample_ids(value: Any, length: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape != (length,):
        raise ValueError(f"sample_ids must have shape ({length},)")
    normalized = array.astype(str)
    if np.any(normalized == "") or np.unique(normalized).size != length:
        raise ValueError("sample_ids must be non-empty and unique")
    return normalized


def _validate_permutation(value: Any, name: str) -> tuple[int, ...]:
    permutation = tuple(int(item) for item in _integer_vector(value, name, NUM_ROLES))
    if tuple(sorted(permutation)) != tuple(range(NUM_ROLES)):
        raise ValueError(f"{name} must be a permutation of 0..3")
    return permutation


def _validate_probabilities(weights: Any, name: str, length: int) -> np.ndarray:
    array = _finite_array(weights, name, ndim=2)
    if array.shape != (length, NUM_ROLES):
        raise ValueError(f"{name} must have shape ({length}, {NUM_ROLES})")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must lie in [0, 1]")
    if not np.allclose(array.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-6):
        raise ValueError(f"{name} rows must sum to one within 1e-6")
    return array


def _validate_factorial_labels(
    mechanisms: np.ndarray,
    diagnoses: np.ndarray,
    nuisance_cells: np.ndarray,
    *,
    require_frozen_design: bool,
) -> None:
    if set(np.unique(mechanisms).tolist()) != set(range(NUM_ROLES)):
        raise ValueError("mechanism_ids must contain exactly 0, 1, 2, and 3")
    if set(np.unique(diagnoses).tolist()) != set(range(NUM_ROLES)):
        raise ValueError("diagnosis_labels must contain exactly 0, 1, 2, and 3")
    for mechanism in range(NUM_ROLES):
        for diagnosis in range(NUM_ROLES):
            selected = (mechanisms == mechanism) & (diagnoses == diagnosis)
            if not np.any(selected):
                raise ValueError(
                    f"missing mechanism={mechanism}, diagnosis={diagnosis} stratum"
                )
            cells, counts = np.unique(nuisance_cells[selected], return_counts=True)
            if require_frozen_design:
                if cells.size != FROZEN_CELLS_PER_STRATUM:
                    raise ValueError(
                        "every mechanism/diagnosis stratum must contain exactly "
                        f"{FROZEN_CELLS_PER_STRATUM} nuisance cells"
                    )
                if not np.all(counts == FROZEN_DRAWS_PER_CELL):
                    raise ValueError(
                        "every frozen nuisance cell must contain exactly "
                        f"{FROZEN_DRAWS_PER_CELL} observations"
                    )


def response_magnitudes(expert_features: Any) -> np.ndarray:
    """Return q=sqrt(mean_h(feature**2)+1e-8) for ``[N,4,H]`` features."""
    features = _finite_array(expert_features, "expert_features", ndim=3)
    if features.shape[1] != NUM_ROLES or features.shape[2] < 1:
        raise ValueError("expert_features must have shape [N, 4, H] with H >= 1")
    with np.errstate(over="raise", invalid="raise"):
        try:
            response = np.sqrt(np.mean(np.square(features), axis=2) + 1.0e-8)
        except FloatingPointError as exc:
            raise ValueError("expert_features overflow while computing response") from exc
    if not np.isfinite(response).all():
        raise ValueError("computed responses must be finite")
    return response


def zscore_responses(responses: Any) -> np.ndarray:
    """Population-z-score four responses per observation without a floor."""
    array = _finite_array(responses, "responses", ndim=2)
    if array.shape[1] != NUM_ROLES:
        raise ValueError("responses must have exactly four expert columns")
    standard_deviation = array.std(axis=1, ddof=0)
    invalid = np.flatnonzero(standard_deviation < SD_INVALID_THRESHOLD)
    if invalid.size:
        raise ValueError(
            "invalid mapping: across-expert response SD below 1e-8 at "
            f"{invalid.size} observation(s), first index {int(invalid[0])}"
        )
    return (array - array.mean(axis=1, keepdims=True)) / standard_deviation[:, None]


def aggregate_equal_factorial(
    values: Any,
    mechanism_ids: Any,
    diagnosis_labels: Any,
    nuisance_cell_ids: Any,
    *,
    require_frozen_design: bool = False,
) -> np.ndarray:
    """Aggregate ``[N,D]`` by equal mechanism/diagnosis/cell/observation weights.

    The returned matrix is ``[D,4]``: one row per value column and one column per
    mechanism.  Diagnosis labels and nuisance cells receive equal weight even
    when their observation counts differ.
    """
    matrix = _finite_array(values, "values", ndim=2)
    count = matrix.shape[0]
    mechanisms = _integer_vector(mechanism_ids, "mechanism_ids", count)
    diagnoses = _integer_vector(diagnosis_labels, "diagnosis_labels", count)
    cells = _integer_vector(nuisance_cell_ids, "nuisance_cell_ids", count)
    _validate_factorial_labels(
        mechanisms, diagnoses, cells, require_frozen_design=require_frozen_design
    )

    mechanism_means: list[np.ndarray] = []
    for mechanism in range(NUM_ROLES):
        diagnosis_means: list[np.ndarray] = []
        for diagnosis in range(NUM_ROLES):
            stratum = (mechanisms == mechanism) & (diagnoses == diagnosis)
            cell_means = [
                matrix[stratum & (cells == cell)].mean(axis=0)
                for cell in np.unique(cells[stratum])
            ]
            diagnosis_means.append(np.mean(np.stack(cell_means), axis=0))
        mechanism_means.append(np.mean(np.stack(diagnosis_means), axis=0))
    return np.stack(mechanism_means, axis=1)


def cosine_cost_matrix(signatures: Any, templates: Any) -> np.ndarray:
    observed = _finite_array(signatures, "signatures", ndim=2)
    expected = _finite_array(templates, "templates", ndim=2)
    if observed.shape[1] != expected.shape[1]:
        raise ValueError("signatures and templates must have the same width")
    observed_norm = np.linalg.norm(observed, axis=1)
    expected_norm = np.linalg.norm(expected, axis=1)
    if np.any(observed_norm < SIGNATURE_NORM_INVALID_THRESHOLD):
        raise ValueError("invalid mapping: aggregated signature norm below 1e-8")
    if np.any(expected_norm < SIGNATURE_NORM_INVALID_THRESHOLD):
        raise ValueError("template norm below 1e-8")
    similarity = (observed / observed_norm[:, None]) @ (
        expected / expected_norm[:, None]
    ).T
    return 1.0 - similarity


def exact_cosine_assignment(
    signatures: Any, templates: Any
) -> tuple[tuple[int, ...], float, np.ndarray]:
    """Return role-to-expert assignment by exact lexicographic 4! enumeration."""
    costs = cosine_cost_matrix(signatures, templates)
    if costs.shape != (NUM_ROLES, NUM_ROLES):
        raise ValueError("assignment requires exactly four experts and four roles")
    best: tuple[int, ...] | None = None
    best_cost = math.inf
    for role_to_expert in permutations(range(NUM_ROLES)):
        total = float(
            sum(costs[expert, role] for role, expert in enumerate(role_to_expert))
        )
        if total < best_cost:
            best = tuple(role_to_expert)
            best_cost = total
    if best is None:  # pragma: no cover - permutations(4) is nonempty
        raise RuntimeError("assignment enumeration failed")
    return best, best_cost, costs


def exact_role_chance(correct_counts: Sequence[int]) -> dict[str, Any]:
    """Exact right-tail chance probability under independent random 4! labels."""
    counts = tuple(int(value) for value in correct_counts)
    if not counts:
        raise ValueError("correct_counts must contain at least one seed")
    if any(value not in (0, 1, 2, 4) for value in counts):
        raise ValueError("each seed fixed-point count must be one of 0, 1, 2, or 4")
    one_seed = np.asarray([9, 8, 6, 0, 1], dtype=object)
    distribution = np.asarray([1], dtype=object)
    for _ in counts:
        distribution = np.convolve(distribution, one_seed)
    observed = sum(counts)
    numerator = int(sum(distribution[observed:]))
    denominator = 24 ** len(counts)
    return {
        "seed_count": len(counts),
        "correct_counts": list(counts),
        "observed_correct": observed,
        "total_decisions": NUM_ROLES * len(counts),
        "null_tail_numerator": numerator,
        "null_denominator": denominator,
        "one_sided_p_value": numerator / denominator,
        "null_fixed_point_counts_one_seed": {"0": 9, "1": 8, "2": 6, "3": 0, "4": 1},
    }


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_numeric_array_sha256(value: Any, dtype: str, name: str) -> str:
    array = np.asarray(value)
    canonical = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    header = json.dumps(
        {"dtype": canonical.dtype.str, "name": name, "shape": canonical.shape},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def build_preintervention_assignment_seal(
    expert_features: Any,
    mechanism_ids: Any,
    diagnosis_labels: Any,
    nuisance_cell_ids: Any,
    draw_ids: Any,
    sample_ids: Any,
    *,
    arm: str,
    seed: int,
    require_frozen_design: bool = True,
) -> tuple[dict[str, Any], str]:
    """Build the canonical response-only seal without a designated mapping.

    The seal is intentionally independent of routing, intervention arrays,
    expert names, and the post-assignment recovery target.
    """
    if not isinstance(arm, str) or not arm:
        raise ValueError("seal arm must be a non-empty string")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seal seed must be an integer")
    features = _finite_array(expert_features, "expert_features", ndim=3)
    if features.shape[1] != NUM_ROLES or features.shape[2] < 1:
        raise ValueError("expert_features must have shape [N, 4, H] with H >= 1")
    count = features.shape[0]
    mechanisms = _mechanism_vector(mechanism_ids, count)
    diagnoses = _integer_vector(diagnosis_labels, "diagnosis_labels", count)
    cells = _integer_vector(nuisance_cell_ids, "nuisance_cell_ids", count)
    draws = _integer_vector(draw_ids, "draw_ids", count)
    identifiers = _sample_ids(sample_ids, count)
    if np.any(draws < 0) or np.any(draws >= FROZEN_DRAWS_PER_CELL):
        raise ValueError("seal draw_ids must lie in 0..7")
    _validate_factorial_labels(
        mechanisms,
        diagnoses,
        cells,
        require_frozen_design=require_frozen_design,
    )
    responses = response_magnitudes(features)
    standardized = zscore_responses(responses)
    response_signature = aggregate_equal_factorial(
        standardized,
        mechanisms,
        diagnoses,
        cells,
        require_frozen_design=require_frozen_design,
    )
    role_to_expert, _, _ = exact_cosine_assignment(
        response_signature, np.eye(NUM_ROLES, dtype=np.float64)
    )
    identifiers_sha256 = hashlib.sha256(
        json.dumps(
            identifiers.tolist(), ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    input_components = {
        "sample_ids_sha256": identifiers_sha256,
        "mechanism_ids_sha256": _canonical_numeric_array_sha256(
            mechanisms, "<i8", "mechanism_ids"
        ),
        "diagnosis_labels_sha256": _canonical_numeric_array_sha256(
            diagnoses, "<i8", "diagnosis_labels"
        ),
        "nuisance_cell_ids_sha256": _canonical_numeric_array_sha256(
            cells, "<i8", "nuisance_cell_ids"
        ),
        "draw_ids_sha256": _canonical_numeric_array_sha256(
            draws, "<i8", "draw_ids"
        ),
        "expert_features_sha256": _canonical_numeric_array_sha256(
            features, "<f8", "expert_features"
        ),
    }
    identification_inputs_sha256 = hashlib.sha256(
        _canonical_json_bytes(input_components)
    ).hexdigest()
    seal = {
        "schema_id": ASSIGNMENT_SEAL_SCHEMA,
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "arm": arm,
        "seed": int(seed),
        "phase_order": list(COLLECTION_PHASE_ORDER),
        "sealed_after_phase": COLLECTION_PHASE_ORDER[0],
        "sealed_before_phase": COLLECTION_PHASE_ORDER[2],
        "assignment_source": "identification_response_only",
        "assignment_method": "exact_4_factorial_cosine_cost",
        "identification_observation_count": count,
        "identification_inputs_sha256": identification_inputs_sha256,
        "response_signature_sha256": _canonical_numeric_array_sha256(
            response_signature, "<f8", "response_signature"
        ),
        "role_to_blinded_expert": list(role_to_expert),
    }
    seal_sha256 = hashlib.sha256(_canonical_json_bytes(seal)).hexdigest()
    return seal, seal_sha256


def verify_preintervention_assignment_seal(
    recorded_json: Any,
    recorded_sha256: Any,
    expert_features: Any,
    mechanism_ids: Any,
    diagnosis_labels: Any,
    nuisance_cell_ids: Any,
    draw_ids: Any,
    sample_ids: Any,
    *,
    arm: str,
    seed: int,
    require_frozen_design: bool = True,
) -> tuple[dict[str, Any], str]:
    """Recompute and verify a canonical seal before intervention analysis."""
    serialized = _text_scalar(recorded_json, "assignment_seal_json")
    digest = _validate_sha256(recorded_sha256, "assignment_seal_sha256")
    try:
        parsed = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise ValueError(f"assignment_seal_json is invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("assignment_seal_json must contain an object")
    canonical = _canonical_json_bytes(parsed)
    if serialized.encode("utf-8") != canonical:
        raise ValueError("assignment_seal_json must use canonical JSON serialization")
    if hashlib.sha256(canonical).hexdigest() != digest:
        raise ValueError("assignment seal content/hash mismatch")
    expected, expected_digest = build_preintervention_assignment_seal(
        expert_features,
        mechanism_ids,
        diagnosis_labels,
        nuisance_cell_ids,
        draw_ids,
        sample_ids,
        arm=arm,
        seed=seed,
        require_frozen_design=require_frozen_design,
    )
    if parsed != expected or digest != expected_digest:
        raise ValueError("assignment seal does not match identification data")
    return expected, expected_digest


def _equal_factorial_scalar(
    values: np.ndarray,
    mechanisms: np.ndarray,
    diagnoses: np.ndarray,
    cells: np.ndarray,
    *,
    retained: np.ndarray | None = None,
) -> tuple[float, list[float]]:
    if values.ndim != 1 or values.shape != mechanisms.shape:
        raise ValueError("scalar values must have one value per observation")
    if retained is None:
        retained = np.ones(values.shape[0], dtype=bool)
    if retained.shape != values.shape:
        raise ValueError("retained mask must have one value per observation")
    by_mechanism: list[float] = []
    for mechanism in range(NUM_ROLES):
        by_diagnosis: list[float] = []
        for diagnosis in range(NUM_ROLES):
            stratum = (mechanisms == mechanism) & (diagnoses == diagnosis)
            by_cell: list[float] = []
            for cell in np.unique(cells[stratum]):
                selected = stratum & (cells == cell) & retained
                if not np.any(selected):
                    raise ValueError("a retained nuisance cell has no observations")
                by_cell.append(float(values[selected].mean()))
            by_diagnosis.append(float(np.mean(by_cell)))
        by_mechanism.append(float(np.mean(by_diagnosis)))
    return float(np.mean(by_mechanism)), by_mechanism


def _cross_entropy(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if logits.ndim != 2 or labels.shape != (logits.shape[0],):
        raise ValueError("logits/labels shapes are incompatible")
    if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("labels contain an out-of-range class index")
    maximum = logits.max(axis=1)
    log_normalizer = maximum + np.log(
        np.exp(logits - maximum[:, None]).sum(axis=1)
    )
    return log_normalizer - logits[np.arange(logits.shape[0]), labels]


def evaluate_interventions(
    expert_logits: Any,
    routing_weights: Any,
    labels: Any,
    mechanism_ids: Any,
    diagnosis_labels: Any,
    nuisance_cell_ids: Any,
    role_to_expert: Sequence[int],
    *,
    require_frozen_design: bool = True,
    intact_logits_artifact: Any | None = None,
    deleted_logits_artifact: Any | None = None,
    fixed_mass_swap_logits_artifact: Any | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute primary deletion, mass-match, and fixed-mass substitution estimands."""
    logits = _finite_array(expert_logits, "expert_logits", ndim=3)
    if logits.shape[1] != NUM_ROLES or logits.shape[2] < 2:
        raise ValueError("expert_logits must have shape [N, 4, C] with C >= 2")
    count = logits.shape[0]
    weights = _validate_probabilities(routing_weights, "routing_weights", count)
    targets = _integer_vector(labels, "labels", count)
    mechanisms = _integer_vector(mechanism_ids, "mechanism_ids", count)
    diagnoses = _integer_vector(diagnosis_labels, "diagnosis_labels", count)
    cells = _integer_vector(nuisance_cell_ids, "nuisance_cell_ids", count)
    mapping = _validate_permutation(role_to_expert, "role_to_expert")
    _validate_factorial_labels(
        mechanisms, diagnoses, cells, require_frozen_design=require_frozen_design
    )
    if not np.array_equal(targets, diagnoses):
        raise ValueError(
            "intervention labels must equal diagnosis_labels in the frozen synthetic design"
        )

    reconstructed_intact_logits = np.sum(weights[:, :, None] * logits, axis=1)
    if intact_logits_artifact is None:
        intact_logits = reconstructed_intact_logits
        intact_logits_validated = False
    else:
        intact_logits = _finite_array(intact_logits_artifact, "logits", ndim=2)
        if intact_logits.shape != reconstructed_intact_logits.shape:
            raise ValueError("provided logits must have shape [N, C]")
        if not np.allclose(
            intact_logits,
            reconstructed_intact_logits,
            rtol=1.0e-5,
            atol=1.0e-6,
        ):
            difference = float(
                np.max(np.abs(intact_logits - reconstructed_intact_logits))
            )
            raise ValueError(
                "provided logits do not equal the fixed intact weighted expert sum; "
                f"maximum absolute difference {difference:.6g}"
            )
        intact_logits_validated = True
    baseline_loss = _cross_entropy(intact_logits, targets)
    denominator = 1.0 - weights
    invalid = np.argwhere(denominator < DELETION_DENOMINATOR_INVALID_THRESHOLD)
    if invalid.size:
        first_observation, first_expert = invalid[0]
        raise ValueError(
            "invalid intervention: 1-w_e below 1e-6 at observation "
            f"{int(first_observation)}, expert {int(first_expert)}"
        )

    surviving_weights = np.broadcast_to(
        weights[:, None, :], (count, NUM_ROLES, NUM_ROLES)
    ).copy()
    expert_index = np.arange(NUM_ROLES)
    surviving_weights[:, expert_index, expert_index] = 0.0
    effective_weights = surviving_weights / denominator[:, :, None]
    reconstructed_deleted_logits = np.sum(
        effective_weights[:, :, :, None] * logits[:, None, :, :], axis=2
    )
    if deleted_logits_artifact is None:
        deleted_logits_renormalized = reconstructed_deleted_logits
        deleted_logits_validated = False
    else:
        deleted_logits_renormalized = _finite_array(
            deleted_logits_artifact, "deleted_logits", ndim=3
        )
        if deleted_logits_renormalized.shape != reconstructed_deleted_logits.shape:
            raise ValueError("provided deleted_logits must have shape [N, 4, C]")
        if not np.allclose(
            deleted_logits_renormalized,
            reconstructed_deleted_logits,
            rtol=1.0e-5,
            atol=1.0e-6,
        ):
            difference = float(
                np.max(
                    np.abs(deleted_logits_renormalized - reconstructed_deleted_logits)
                )
            )
            raise ValueError(
                "provided deleted_logits imply rerouting or a non-frozen deletion; "
                f"maximum absolute difference {difference:.6g}"
            )
        deleted_logits_validated = True
    deleted_logits_no_renormalization = (
        intact_logits[:, None, :] - weights[:, :, None] * logits
    )
    deleted_loss_renormalized = np.column_stack(
        [
            _cross_entropy(deleted_logits_renormalized[:, expert, :], targets)
            for expert in range(NUM_ROLES)
        ]
    )
    deleted_loss_no_renormalization = np.column_stack(
        [
            _cross_entropy(deleted_logits_no_renormalization[:, expert, :], targets)
            for expert in range(NUM_ROLES)
        ]
    )

    row_index = np.arange(count)
    matched_expert = np.asarray([mapping[item] for item in mechanisms], dtype=np.int64)
    matched_delta = (
        deleted_loss_renormalized[row_index, matched_expert] - baseline_loss
    )
    matched_delta_no_renorm = (
        deleted_loss_no_renormalization[row_index, matched_expert] - baseline_loss
    )
    nonmatching_delta = np.empty(count, dtype=np.float64)
    nonmatching_delta_no_renorm = np.empty(count, dtype=np.float64)
    for observation in range(count):
        alternatives = [
            expert for expert in range(NUM_ROLES) if expert != matched_expert[observation]
        ]
        nonmatching_delta[observation] = float(
            np.mean(
                deleted_loss_renormalized[observation, alternatives]
                - baseline_loss[observation]
            )
        )
        nonmatching_delta_no_renorm[observation] = float(
            np.mean(
                deleted_loss_no_renormalization[observation, alternatives]
                - baseline_loss[observation]
            )
        )

    primary_observation_contrast = matched_delta - nonmatching_delta
    no_renorm_observation_contrast = (
        matched_delta_no_renorm - nonmatching_delta_no_renorm
    )
    primary, primary_by_mechanism = _equal_factorial_scalar(
        primary_observation_contrast, mechanisms, diagnoses, cells
    )
    no_renorm, no_renorm_by_mechanism = _equal_factorial_scalar(
        no_renorm_observation_contrast, mechanisms, diagnoses, cells
    )

    mass_match_expert = np.full(count, -1, dtype=np.int64)
    mass_match_distance = np.full(count, np.inf, dtype=np.float64)
    for observation in range(count):
        matched = int(matched_expert[observation])
        alternatives = [expert for expert in range(NUM_ROLES) if expert != matched]
        distances = np.asarray(
            [abs(weights[observation, expert] - weights[observation, matched]) for expert in alternatives]
        )
        chosen_offset = int(np.argmin(distances))
        mass_match_expert[observation] = alternatives[chosen_offset]
        mass_match_distance[observation] = distances[chosen_offset]
    mass_match_retained = mass_match_distance <= MASS_MATCH_CALIPER

    coverage_records: list[dict[str, Any]] = []
    all_cells_supported = True
    for mechanism in range(NUM_ROLES):
        for diagnosis in range(NUM_ROLES):
            stratum = (mechanisms == mechanism) & (diagnoses == diagnosis)
            for cell in np.unique(cells[stratum]):
                selected = stratum & (cells == cell)
                retained_count = int(np.sum(mass_match_retained[selected]))
                total_count = int(np.sum(selected))
                coverage = retained_count / total_count
                supported = coverage >= MASS_MATCH_MINIMUM_COVERAGE
                all_cells_supported = all_cells_supported and supported
                coverage_records.append(
                    {
                        "mechanism_id": mechanism,
                        "diagnosis_label": diagnosis,
                        "nuisance_cell_id": int(cell),
                        "retained": retained_count,
                        "total": total_count,
                        "coverage": coverage,
                        "supported": supported,
                    }
                )
    mass_match_summary: dict[str, Any] = {
        "estimable": all_cells_supported,
        "caliper": MASS_MATCH_CALIPER,
        "minimum_coverage_per_cell": MASS_MATCH_MINIMUM_COVERAGE,
        "retained_observations": int(mass_match_retained.sum()),
        "total_observations": count,
        "cell_coverage": coverage_records,
        "interaction": None,
        "interaction_by_mechanism": None,
    }
    if all_cells_supported:
        selected_delta = (
            deleted_loss_renormalized[row_index, mass_match_expert] - baseline_loss
        )
        mass_observation_contrast = matched_delta - selected_delta
        mass_interaction, mass_by_mechanism = _equal_factorial_scalar(
            mass_observation_contrast,
            mechanisms,
            diagnoses,
            cells,
            retained=mass_match_retained,
        )
        mass_match_summary["interaction"] = mass_interaction
        mass_match_summary["interaction_by_mechanism"] = mass_by_mechanism

    matched_weights = weights[row_index, matched_expert]
    matched_outputs = logits[row_index, matched_expert]
    swap_losses = np.empty((count, NUM_ROLES), dtype=np.float64)
    if fixed_mass_swap_logits_artifact is None:
        selected_swap_logits = np.empty(
            (count, NUM_ROLES, logits.shape[2]), dtype=np.float64
        )
        for replacement_expert in range(NUM_ROLES):
            selected_swap_logits[:, replacement_expert, :] = (
                intact_logits
                - matched_weights[:, None] * matched_outputs
                + matched_weights[:, None] * logits[:, replacement_expert, :]
            )
        fixed_mass_swap_logits_validated = False
    else:
        # The collector emits this artifact in model dtype.  Preserve that raw
        # dtype for verification so NumPy reproduces the collector's arithmetic
        # rather than silently promoting the operands and changing cancellation.
        all_swap_logits = np.asarray(fixed_mass_swap_logits_artifact)
        if all_swap_logits.ndim != 4:
            raise ValueError("fixed_mass_swap_logits must have 4 dimensions")
        if not np.issubdtype(all_swap_logits.dtype, np.number):
            raise ValueError("fixed_mass_swap_logits must be numeric")
        if not np.isfinite(all_swap_logits).all():
            raise ValueError("fixed_mass_swap_logits must be finite")
        expected_shape = (count, NUM_ROLES, NUM_ROLES, logits.shape[2])
        if all_swap_logits.shape != expected_shape:
            raise ValueError(
                "provided fixed_mass_swap_logits must have shape [N, 4, 4, C]"
            )
        verification_dtype = all_swap_logits.dtype
        verification_intact = np.asarray(intact_logits, dtype=verification_dtype)
        verification_weights = np.asarray(weights, dtype=verification_dtype)
        verification_logits = np.asarray(logits, dtype=verification_dtype)
        expected_all_swap_logits = np.empty_like(all_swap_logits)
        for source_expert in range(NUM_ROLES):
            source_mass = verification_weights[:, source_expert, None]
            source_output = verification_logits[:, source_expert, :]
            for replacement_expert in range(NUM_ROLES):
                # Exact collector order: (intact - w_s*z_s) + w_s*z_r.
                expected_all_swap_logits[:, source_expert, replacement_expert, :] = (
                    verification_intact - source_mass * source_output
                ) + (
                    source_mass * verification_logits[:, replacement_expert, :]
                )
        diagonal = np.arange(NUM_ROLES)
        expected_all_swap_logits[:, diagonal, diagonal, :] = verification_intact[
            :, None, :
        ]
        if not np.allclose(
            all_swap_logits,
            expected_all_swap_logits,
            rtol=FIXED_MASS_RTOL,
            atol=FIXED_MASS_ATOL,
        ):
            difference = float(
                np.max(np.abs(all_swap_logits - expected_all_swap_logits))
            )
            raise ValueError(
                "provided fixed_mass_swap_logits violate the frozen substitution; "
                f"maximum absolute difference {difference:.6g}"
            )
        # Keep the downstream estimand computation on its historical float64
        # path; only the verifier uses the raw collector dtype.
        selected_swap_logits = all_swap_logits.astype(np.float64, copy=False)[
            row_index, matched_expert
        ]
        fixed_mass_swap_logits_validated = True
    for replacement_expert in range(NUM_ROLES):
        swap_losses[:, replacement_expert] = _cross_entropy(
            selected_swap_logits[:, replacement_expert, :], targets
        )
    substitution_observation = np.empty(count, dtype=np.float64)
    for observation in range(count):
        alternatives = [
            expert for expert in range(NUM_ROLES) if expert != matched_expert[observation]
        ]
        substitution_observation[observation] = float(
            np.mean(swap_losses[observation, alternatives] - baseline_loss[observation])
        )
    substitution, substitution_by_mechanism = _equal_factorial_scalar(
        substitution_observation, mechanisms, diagnoses, cells
    )
    mean_intact_cross_entropy, intact_ce_by_mechanism = _equal_factorial_scalar(
        baseline_loss, mechanisms, diagnoses, cells
    )
    intact_prediction = np.argmax(intact_logits, axis=1)
    recalls: list[float] = []
    label_support: list[int] = []
    for label in range(NUM_ROLES):
        selected = targets == label
        if not np.any(selected):  # guarded by the frozen factorial, kept explicit
            raise ValueError(f"intervention label {label} has no observations")
        recalls.append(float(np.mean(intact_prediction[selected] == label)))
        label_support.append(int(selected.sum()))

    summary = {
        "schema_id": "p04.intervention-summary.v1",
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "inference_unit": "independent_training_seed",
        "aggregation_order": [
            "mechanism",
            "diagnosis_label",
            "nuisance_cell",
            "observation",
        ],
        "primary_deletion": {
            "router_recomputed": False,
            "renormalize_remaining_weights": True,
            "denominator": "1-w_e",
            "denominator_clamped": False,
            "minimum_observed_denominator": float(denominator.min()),
            "interaction": primary,
            "interaction_by_mechanism": primary_by_mechanism,
            "intact_logits_artifact_validated": intact_logits_validated,
            "deleted_logits_artifact_validated": deleted_logits_validated,
        },
        "no_renormalization_sensitivity": {
            "interaction": no_renorm,
            "interaction_by_mechanism": no_renorm_by_mechanism,
        },
        "routing_mass_match_sensitivity": mass_match_summary,
        "fixed_mass_output_substitution": {
            "estimand_J": substitution,
            "estimand_J_by_mechanism": substitution_by_mechanism,
            "nonmatching_outputs_per_observation": NUM_ROLES - 1,
            "swap_logits_artifact_validated": fixed_mass_swap_logits_validated,
        },
        "intact_task_competence": {
            "balanced_accuracy": float(np.mean(recalls)),
            "label_recalls": recalls,
            "label_support": label_support,
            "every_label_recall_positive": all(recall > 0.0 for recall in recalls),
            "mean_cross_entropy_equal_factorial": mean_intact_cross_entropy,
            "cross_entropy_by_mechanism": intact_ce_by_mechanism,
            "argmax_tie_break": "ascending_logit_column_index",
        },
        "observation_count": count,
    }
    arrays = {
        "baseline_loss": baseline_loss,
        "deleted_loss_renormalized": deleted_loss_renormalized,
        "deleted_loss_no_renormalization": deleted_loss_no_renormalization,
        "fixed_mass_swap_loss": swap_losses,
        "matched_expert": matched_expert,
        "primary_observation_contrast": primary_observation_contrast,
        "no_renorm_observation_contrast": no_renorm_observation_contrast,
        "mass_match_expert": mass_match_expert,
        "mass_match_distance": mass_match_distance,
        "mass_match_retained": mass_match_retained,
        "substitution_observation": substitution_observation,
    }
    return summary, arrays


def evaluate_identification(
    expert_features: Any,
    routing_weights: Any,
    mechanism_ids: Any,
    diagnosis_labels: Any,
    nuisance_cell_ids: Any,
    designated_role_to_expert: Sequence[int],
    *,
    require_frozen_design: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    features = _finite_array(expert_features, "expert_features", ndim=3)
    count = features.shape[0]
    weights = _validate_probabilities(routing_weights, "routing_weights", count)
    mechanisms = _integer_vector(mechanism_ids, "mechanism_ids", count)
    diagnoses = _integer_vector(diagnosis_labels, "diagnosis_labels", count)
    cells = _integer_vector(nuisance_cell_ids, "nuisance_cell_ids", count)
    designated = _validate_permutation(
        designated_role_to_expert, "designated_role_to_expert"
    )
    responses = response_magnitudes(features)
    standardized = zscore_responses(responses)
    response_signature = aggregate_equal_factorial(
        standardized,
        mechanisms,
        diagnoses,
        cells,
        require_frozen_design=require_frozen_design,
    )
    routing_signature = aggregate_equal_factorial(
        weights,
        mechanisms,
        diagnoses,
        cells,
        require_frozen_design=require_frozen_design,
    )
    identity = np.eye(NUM_ROLES, dtype=np.float64)
    role_to_expert, total_cost, costs = exact_cosine_assignment(
        response_signature, identity
    )
    routing_assignment, routing_cost, _ = exact_cosine_assignment(
        routing_signature, identity
    )
    combined_assignment, combined_cost, _ = exact_cosine_assignment(
        np.concatenate([response_signature, routing_signature], axis=1),
        np.concatenate([identity, identity], axis=1),
    )
    correct_by_role = [
        role_to_expert[role] == designated[role] for role in range(NUM_ROLES)
    ]
    correct_count = int(sum(correct_by_role))
    chance = exact_role_chance([correct_count])

    signatures = {
        "schema_id": "p04.behavior-signatures.v1",
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "primary_signature": "response_only",
        "response_definition": "sqrt(mean_h(feature^2)+1e-8)",
        "within_observation_transform": "population_zscore_across_four_experts_ddof0_no_floor",
        "aggregation_order": [
            "mechanism",
            "diagnosis_label",
            "nuisance_cell",
            "observation",
        ],
        "canonical_mechanisms": list(ROLE_NAMES),
        "response_only": response_signature.tolist(),
        "routing_only_sensitivity": routing_signature.tolist(),
        "response_plus_routing_sensitivity": np.concatenate(
            [response_signature, routing_signature], axis=1
        ).tolist(),
        "observation_count": count,
    }
    assignment = {
        "schema_id": "p04.role-assignment.v1",
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "assignment_source": "identification_partition_response_only",
        "assignment_method": "exact_4_factorial_cosine_cost",
        "tie_break": "lexicographically_ascending_role_to_expert_permutation",
        "role_to_blinded_expert": list(role_to_expert),
        "total_cosine_cost": total_cost,
        "cosine_cost_expert_by_role": costs.tolist(),
        "designated_role_to_blinded_expert": list(designated),
        "correct_by_role": correct_by_role,
        "correct_count": correct_count,
        "role_recovery": correct_count / NUM_ROLES,
        "single_seed_chance_reference": chance,
        "sensitivity_assignments": {
            "routing_only": {
                "role_to_blinded_expert": list(routing_assignment),
                "total_cosine_cost": routing_cost,
            },
            "response_plus_routing": {
                "role_to_blinded_expert": list(combined_assignment),
                "total_cosine_cost": combined_cost,
            },
        },
    }
    return signatures, assignment


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray], str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except Exception as exc:
        raise ValueError(f"could not read NPZ artifact {path}: {exc}") from exc
    return arrays, digest


def _required(mapping: Mapping[str, Any], names: Iterable[str], schema: str) -> None:
    missing = sorted(set(names) - set(mapping))
    if missing:
        raise ValueError(f"{schema} artifact is missing fields: {', '.join(missing)}")


def _validate_trace_header(
    arrays: Mapping[str, Any], expected_schema: str, expected_partition: str
) -> dict[str, Any]:
    _required(
        arrays,
        (
            "schema_id",
            "seed",
            "arm",
            "partition_name",
            "generator_manifest_sha256",
            "partition_manifest_sha256",
        ),
        expected_schema,
    )
    schema = _text_scalar(arrays["schema_id"], "schema_id")
    if schema != expected_schema:
        raise ValueError(f"schema_id must be {expected_schema!r}, got {schema!r}")
    partition = _text_scalar(arrays["partition_name"], "partition_name")
    if partition != expected_partition:
        raise ValueError(
            f"partition_name must be {expected_partition!r}, got {partition!r}"
        )
    return {
        "seed": _integer_scalar(arrays["seed"], "seed"),
        "arm": _text_scalar(arrays["arm"], "arm"),
        "partition_name": partition,
        "generator_manifest_sha256": _validate_sha256(
            arrays["generator_manifest_sha256"], "generator_manifest_sha256"
        ),
        "partition_manifest_sha256": _validate_sha256(
            arrays["partition_manifest_sha256"], "partition_manifest_sha256"
        ),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON output cannot contain non-finite values")
    return value


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            _json_ready(dict(payload)),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def _write_deterministic_npz_exclusive(
    path: Path, arrays: Mapping[str, np.ndarray]
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    with path.open("xb") as raw_handle:
        with zipfile.ZipFile(
            raw_handle, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(
                    buffer, np.asarray(arrays[name]), allow_pickle=False
                )
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(
                    info,
                    buffer.getvalue(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )


def _write_outputs(
    output_dir: Path,
    signatures: Mapping[str, Any],
    assignment: Mapping[str, Any],
    intervention_summary: Mapping[str, Any],
    deletion_arrays: Mapping[str, np.ndarray],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "behavior_signatures": output_dir / "behavior_signatures.json",
        "role_assignment": output_dir / "role_assignment.json",
        "deletion_losses": output_dir / "deletion_losses.npz",
        "metrics": output_dir / "metrics.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing artifact(s): " + ", ".join(existing)
        )
    role_to_expert = [int(value) for value in assignment["role_to_blinded_expert"]]
    expert_to_role = [0] * NUM_ROLES
    for role, expert in enumerate(role_to_expert):
        expert_to_role[expert] = role
    correct_by_role = [bool(value) for value in assignment["correct_by_role"]]
    metrics = {
        "schema_id": "p04.mechanism-metrics.v1",
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "role_recovery_count": int(assignment["correct_count"]),
        "role_recovery_accuracy": float(assignment["role_recovery"]),
        "per_role_correctness": {
            ROLE_NAMES[index]: correct_by_role[index] for index in range(NUM_ROLES)
        },
        "role_to_blinded_expert": role_to_expert,
        "blinded_expert_to_role": expert_to_role,
        "mechanism_names": list(ROLE_NAMES),
        "primary_deletion_interaction_I": intervention_summary["primary_deletion"][
            "interaction"
        ],
        "intact_task_competence": intervention_summary["intact_task_competence"],
        "role_chance_reference": assignment["single_seed_chance_reference"],
        "intervention": intervention_summary,
        "provenance": intervention_summary.get("provenance"),
    }
    _write_json_exclusive(paths["behavior_signatures"], signatures)
    _write_json_exclusive(paths["role_assignment"], assignment)
    _write_deterministic_npz_exclusive(paths["deletion_losses"], deletion_arrays)
    _write_json_exclusive(paths["metrics"], metrics)
    return paths


def run_evaluation(
    identification_path: Path, intervention_path: Path, output_dir: Path
) -> dict[str, Path]:
    identification, identification_sha256 = _load_npz(identification_path)
    _required(
        identification,
        (
            "sample_ids",
            "expert_features",
            "routing_weights",
            "mechanism_ids",
            "diagnosis_labels",
            "nuisance_cell_ids",
            "draw_ids",
            "designated_role_to_expert",
            "assignment_seal_json",
            "assignment_seal_sha256",
            "collection_phase_order_json",
            "assignment_sealed_before_intervention_read",
        ),
        IDENTIFICATION_SCHEMA,
    )
    identification_header = _validate_trace_header(
        identification, IDENTIFICATION_SCHEMA, "identification"
    )
    identification_count = np.asarray(identification["expert_features"]).shape[0]
    identification_ids = _sample_ids(
        identification["sample_ids"], identification_count
    )
    identification_mechanisms = _integer_vector(
        identification["mechanism_ids"], "mechanism_ids", identification_count
    )
    identification_diagnoses = _integer_vector(
        identification["diagnosis_labels"],
        "diagnosis_labels",
        identification_count,
    )
    identification_cells = _integer_vector(
        identification["nuisance_cell_ids"],
        "nuisance_cell_ids",
        identification_count,
    )
    identification_draws = _integer_vector(
        identification["draw_ids"], "draw_ids", identification_count
    )
    expected_phase_order_json = json.dumps(
        list(COLLECTION_PHASE_ORDER), separators=(",", ":")
    )
    if _text_scalar(
        identification["collection_phase_order_json"],
        "collection_phase_order_json",
    ) != expected_phase_order_json:
        raise ValueError("collector phase order does not match the frozen ordering")
    if not _boolean_scalar(
        identification["assignment_sealed_before_intervention_read"],
        "assignment_sealed_before_intervention_read",
    ):
        raise ValueError("collector did not seal assignment before intervention read")

    # Fail closed before the intervention artifact is opened: the anonymous
    # response-only assignment must exactly match the canonical identification seal.
    assignment_seal, assignment_seal_sha256 = verify_preintervention_assignment_seal(
        identification["assignment_seal_json"],
        identification["assignment_seal_sha256"],
        identification["expert_features"],
        identification_mechanisms,
        identification_diagnoses,
        identification_cells,
        identification_draws,
        identification_ids,
        arm=identification_header["arm"],
        seed=identification_header["seed"],
        require_frozen_design=True,
    )
    signatures, assignment = evaluate_identification(
        identification["expert_features"],
        identification["routing_weights"],
        identification_mechanisms,
        identification_diagnoses,
        identification_cells,
        identification["designated_role_to_expert"],
        require_frozen_design=True,
    )
    if assignment["role_to_blinded_expert"] != assignment_seal[
        "role_to_blinded_expert"
    ]:
        raise ValueError("recomputed assignment disagrees with pre-intervention seal")

    intervention, intervention_sha256 = _load_npz(intervention_path)
    _required(
        intervention,
        (
            "sample_ids",
            "expert_logits",
            "routing_weights",
            "labels",
            "mechanism_ids",
            "diagnosis_labels",
            "nuisance_cell_ids",
            "assignment_seal_sha256",
        ),
        INTERVENTION_SCHEMA,
    )
    intervention_header = _validate_trace_header(
        intervention, INTERVENTION_SCHEMA, "intervention"
    )
    for field in (
        "seed",
        "arm",
        "generator_manifest_sha256",
        "partition_manifest_sha256",
    ):
        if identification_header[field] != intervention_header[field]:
            raise ValueError(f"identification/intervention {field} values do not match")
    if _validate_sha256(
        intervention["assignment_seal_sha256"], "assignment_seal_sha256"
    ) != assignment_seal_sha256:
        raise ValueError("intervention artifact assignment seal SHA-256 does not match")

    intervention_count = np.asarray(intervention["expert_logits"]).shape[0]
    intervention_ids = _sample_ids(intervention["sample_ids"], intervention_count)
    if np.intersect1d(identification_ids, intervention_ids).size:
        raise ValueError("identification and intervention sample_ids must be disjoint")
    intervention_summary, deletion_arrays = evaluate_interventions(
        intervention["expert_logits"],
        intervention["routing_weights"],
        intervention["labels"],
        intervention["mechanism_ids"],
        intervention["diagnosis_labels"],
        intervention["nuisance_cell_ids"],
        assignment["role_to_blinded_expert"],
        require_frozen_design=True,
        intact_logits_artifact=intervention.get("logits"),
        deleted_logits_artifact=intervention.get("deleted_logits"),
        fixed_mass_swap_logits_artifact=intervention.get("fixed_mass_swap_logits"),
    )
    provenance = {
        **identification_header,
        "identification_trace_sha256": identification_sha256,
        "intervention_trace_sha256": intervention_sha256,
        "assignment_seal_sha256": assignment_seal_sha256,
        "assignment_seal": assignment_seal,
        "collection_phase_order": list(COLLECTION_PHASE_ORDER),
        "assignment_seal_verified_before_intervention": True,
    }
    signatures["provenance"] = provenance
    assignment["provenance"] = provenance
    intervention_summary["provenance"] = provenance
    deletion_arrays = {
        **deletion_arrays,
        "sample_ids": intervention_ids,
        "mechanism_ids": _integer_vector(
            intervention["mechanism_ids"], "mechanism_ids", intervention_count
        ),
        "diagnosis_labels": _integer_vector(
            intervention["diagnosis_labels"], "diagnosis_labels", intervention_count
        ),
        "nuisance_cell_ids": _integer_vector(
            intervention["nuisance_cell_ids"], "nuisance_cell_ids", intervention_count
        ),
        "schema_id": np.asarray("p04.deletion-losses.v1"),
        "seed": np.asarray(identification_header["seed"], dtype=np.int64),
        "arm": np.asarray(identification_header["arm"]),
        "assignment_seal_sha256": np.asarray(assignment_seal_sha256),
    }

    return _write_outputs(
        output_dir,
        signatures,
        assignment,
        intervention_summary,
        deletion_arrays,
    )


def run_unified_evaluation(
    input_path: Path,
    output_dir: Path,
    correction_manifest: Path | None = None,
) -> dict[str, Path]:
    """Consume the preferred row-aligned unified mechanism trace."""
    arrays, input_sha256 = _load_npz(input_path)
    _required(
        arrays,
        (
            "schema_id",
            "seed",
            "arm",
            "generator_manifest_sha256",
            "partition_manifest_sha256",
            "designated_role_to_expert",
            "sample_id",
            "partition",
            "label",
            "mechanism",
            "diagnosis",
            "nuisance_cell",
            "draw",
            "logits",
            "routing_weights",
            "expert_features",
            "expert_logits",
            "deleted_logits",
            "assignment_seal_json",
            "assignment_seal_sha256",
            "collection_phase_order_json",
            "assignment_sealed_before_intervention_read",
        ),
        UNIFIED_SCHEMA,
    )
    schema = _text_scalar(arrays["schema_id"], "schema_id")
    if schema != UNIFIED_SCHEMA:
        raise ValueError(f"schema_id must be {UNIFIED_SCHEMA!r}, got {schema!r}")
    if "schema" in arrays:
        schema_alias = _text_scalar(arrays["schema"], "schema")
        if schema_alias != schema:
            raise ValueError("schema and schema_id must be identical")
    trace_seed = _integer_scalar(arrays["seed"], "seed")
    trace_arm = _text_scalar(arrays["arm"], "arm")
    designated = _validate_permutation(
        arrays["designated_role_to_expert"], "designated_role_to_expert"
    )
    if "designated_role_to_expert_direction" in arrays:
        designated_direction = _text_scalar(
            arrays["designated_role_to_expert_direction"],
            "designated_role_to_expert_direction",
        )
        if designated_direction != "canonical_constrained_slot_to_blinded_column":
            raise ValueError(
                "designated_role_to_expert_direction must be "
                "'canonical_constrained_slot_to_blinded_column'"
            )
    if "blinding_permutation" in arrays:
        blinding = _validate_permutation(
            arrays["blinding_permutation"], "blinding_permutation"
        )
        if "blinding_permutation_direction" not in arrays:
            raise ValueError(
                "blinding_permutation requires blinding_permutation_direction"
            )
        blinding_direction = _text_scalar(
            arrays["blinding_permutation_direction"],
            "blinding_permutation_direction",
        )
        if blinding_direction != "canonical_expert_index_at_each_blinded_column":
            raise ValueError(
                "blinding_permutation_direction must be "
                "'canonical_expert_index_at_each_blinded_column'"
            )
        if "designated_role_to_expert_direction" not in arrays:
            raise ValueError(
                "blinding audit requires designated_role_to_expert_direction"
            )
        if tuple(blinding[index] for index in designated) != tuple(range(NUM_ROLES)):
            raise ValueError(
                "designated_role_to_expert must be the inverse of "
                "blinding_permutation under their explicit frozen directions"
            )
    expert_features = np.asarray(arrays["expert_features"])
    if expert_features.ndim != 3:
        raise ValueError("expert_features must have shape [N, 4, H]")
    count = expert_features.shape[0]
    sample_ids = _sample_ids(arrays["sample_id"], count)
    partitions = np.asarray(arrays["partition"])
    if partitions.shape != (count,):
        raise ValueError("partition must have one value per observation")
    partitions = partitions.astype(str)
    identification_mask = partitions == "identification"
    intervention_mask = partitions == "intervention"
    if not np.any(identification_mask) or not np.any(intervention_mask):
        raise ValueError(
            "partition must contain non-empty identification and intervention rows"
        )
    if np.any(~(identification_mask | intervention_mask)):
        unexpected = sorted(set(partitions[~(identification_mask | intervention_mask)]))
        raise ValueError(
            "unified mechanism trace may contain only identification/intervention "
            f"rows, got {unexpected}"
        )

    labels = _integer_vector(arrays["label"], "label", count)
    mechanisms = _mechanism_vector(arrays["mechanism"], count)
    diagnoses = _integer_vector(arrays["diagnosis"], "diagnosis", count)
    cells = _integer_vector(arrays["nuisance_cell"], "nuisance_cell", count)
    draws = _integer_vector(arrays["draw"], "draw", count)
    if np.any(draws < 0) or np.any(draws >= FROZEN_DRAWS_PER_CELL):
        raise ValueError("draw must lie in the frozen range 0..7")
    phase_order_json = _text_scalar(
        arrays["collection_phase_order_json"], "collection_phase_order_json"
    )
    expected_phase_order_json = json.dumps(
        list(COLLECTION_PHASE_ORDER), separators=(",", ":")
    )
    if phase_order_json != expected_phase_order_json:
        raise ValueError("collector phase order does not match the frozen ordering")
    if not _boolean_scalar(
        arrays["assignment_sealed_before_intervention_read"],
        "assignment_sealed_before_intervention_read",
    ):
        raise ValueError("collector did not seal assignment before intervention read")
    # This verification is intentionally completed before expert logits,
    # deletion logits, or fixed-mass intervention arrays are interpreted.
    assignment_seal, assignment_seal_sha256 = verify_preintervention_assignment_seal(
        arrays["assignment_seal_json"],
        arrays["assignment_seal_sha256"],
        expert_features[identification_mask],
        mechanisms[identification_mask],
        diagnoses[identification_mask],
        cells[identification_mask],
        draws[identification_mask],
        sample_ids[identification_mask],
        arm=trace_arm,
        seed=trace_seed,
        require_frozen_design=True,
    )
    correction_provenance = None
    if correction_manifest is not None:
        correction_provenance = _load_evaluation_correction_manifest(
            correction_manifest,
            trace_sha256=input_sha256,
            arm=trace_arm,
            seed=trace_seed,
            assignment_seal_sha256=assignment_seal_sha256,
        )
    for mask, partition_name in (
        (identification_mask, "identification"),
        (intervention_mask, "intervention"),
    ):
        for mechanism in range(NUM_ROLES):
            for diagnosis in range(NUM_ROLES):
                stratum = mask & (mechanisms == mechanism) & (diagnoses == diagnosis)
                for cell in np.unique(cells[stratum]):
                    selected_draws = draws[stratum & (cells == cell)]
                    if tuple(sorted(selected_draws.tolist())) != tuple(
                        range(FROZEN_DRAWS_PER_CELL)
                    ):
                        raise ValueError(
                            f"{partition_name} cell draws must be exactly 0..7"
                        )

    routing = np.asarray(arrays["routing_weights"])
    expert_logits = np.asarray(arrays["expert_logits"])
    intact_logits = np.asarray(arrays["logits"])
    deleted_logits = np.asarray(arrays["deleted_logits"])
    if routing.shape != (count, NUM_ROLES):
        raise ValueError("routing_weights must have shape [N, 4]")
    if expert_logits.ndim != 3 or expert_logits.shape[:2] != (count, NUM_ROLES):
        raise ValueError("expert_logits must have shape [N, 4, C]")
    if intact_logits.shape != (count, expert_logits.shape[2]):
        raise ValueError("logits must have shape [N, C]")
    if deleted_logits.shape != expert_logits.shape:
        raise ValueError("deleted_logits must have shape [N, 4, C]")
    fixed_mass = arrays.get("fixed_mass_swap_logits")
    if correction_manifest is not None:
        if fixed_mass is None:
            raise ValueError(
                "C2 correction manifest requires fixed_mass_swap_logits in the trace"
            )
        if np.asarray(fixed_mass).dtype != np.dtype(np.float32):
            raise ValueError(
                "C2 correction manifest requires raw fixed_mass_swap_logits dtype float32"
            )
    if fixed_mass is not None:
        fixed_mass = np.asarray(fixed_mass)
        if fixed_mass.shape != (
            count,
            NUM_ROLES,
            NUM_ROLES,
            expert_logits.shape[2],
        ):
            raise ValueError(
                "fixed_mass_swap_logits must have shape [N, 4, 4, C]"
            )

    signatures, assignment = evaluate_identification(
        expert_features[identification_mask],
        routing[identification_mask],
        mechanisms[identification_mask],
        diagnoses[identification_mask],
        cells[identification_mask],
        designated,
        require_frozen_design=True,
    )
    if assignment["role_to_blinded_expert"] != assignment_seal[
        "role_to_blinded_expert"
    ]:
        raise ValueError("recomputed assignment disagrees with pre-intervention seal")
    intervention_summary, deletion_arrays = evaluate_interventions(
        expert_logits[intervention_mask],
        routing[intervention_mask],
        labels[intervention_mask],
        mechanisms[intervention_mask],
        diagnoses[intervention_mask],
        cells[intervention_mask],
        assignment["role_to_blinded_expert"],
        require_frozen_design=True,
        intact_logits_artifact=intact_logits[intervention_mask],
        deleted_logits_artifact=deleted_logits[intervention_mask],
        fixed_mass_swap_logits_artifact=(
            None if fixed_mass is None else fixed_mass[intervention_mask]
        ),
    )
    provenance = {
        "seed": trace_seed,
        "arm": trace_arm,
        "generator_manifest_sha256": _validate_sha256(
            arrays["generator_manifest_sha256"], "generator_manifest_sha256"
        ),
        "partition_manifest_sha256": _validate_sha256(
            arrays["partition_manifest_sha256"], "partition_manifest_sha256"
        ),
        "unified_trace_sha256": input_sha256,
        "assignment_seal_sha256": assignment_seal_sha256,
        "assignment_seal": assignment_seal,
        "collection_phase_order": list(COLLECTION_PHASE_ORDER),
        "assignment_seal_verified_before_intervention": True,
    }
    if correction_provenance is not None:
        provenance.update(correction_provenance)
    signatures["provenance"] = provenance
    assignment["provenance"] = provenance
    intervention_summary["provenance"] = provenance
    intervention_ids = sample_ids[intervention_mask]
    deletion_arrays = {
        **deletion_arrays,
        "sample_ids": intervention_ids,
        "mechanism_ids": mechanisms[intervention_mask],
        "diagnosis_labels": diagnoses[intervention_mask],
        "nuisance_cell_ids": cells[intervention_mask],
        "draw": draws[intervention_mask],
        "schema_id": np.asarray("p04.deletion-losses.v1"),
        "seed": np.asarray(provenance["seed"], dtype=np.int64),
        "arm": np.asarray(provenance["arm"]),
        "assignment_seal_sha256": np.asarray(assignment_seal_sha256),
    }
    if correction_provenance is not None:
        deletion_arrays.update(
            {
                "unified_trace_sha256": np.asarray(input_sha256),
                **{
                    name: np.asarray(value)
                    for name, value in correction_provenance.items()
                },
            }
        )
    return _write_outputs(
        output_dir,
        signatures,
        assignment,
        intervention_summary,
        deletion_arrays,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen P04 anonymous roles and interventions."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Preferred unified p04.mechanism-evaluator-input.v1 NPZ.",
    )
    parser.add_argument("--identification", type=Path)
    parser.add_argument("--intervention", type=Path)
    parser.add_argument(
        "--correction-manifest",
        type=Path,
        help="Optional registered p04.evaluation-correction.v1 JSON manifest.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.input is not None:
        if args.identification is not None or args.intervention is not None:
            raise ValueError(
                "use either --input or the --identification/--intervention pair"
            )
        paths = run_unified_evaluation(
            args.input,
            args.output_dir,
            correction_manifest=args.correction_manifest,
        )
    else:
        if args.correction_manifest is not None:
            raise ValueError("--correction-manifest is supported only with --input")
        if args.identification is None or args.intervention is None:
            raise ValueError(
                "provide --input or both --identification and --intervention"
            )
        paths = run_evaluation(args.identification, args.intervention, args.output_dir)
    print(json.dumps({key: str(value) for key, value in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ASSIGNMENT_SEAL_SCHEMA",
    "COLLECTION_PHASE_ORDER",
    "aggregate_equal_factorial",
    "build_preintervention_assignment_seal",
    "cosine_cost_matrix",
    "evaluate_identification",
    "evaluate_interventions",
    "exact_cosine_assignment",
    "exact_role_chance",
    "response_magnitudes",
    "run_evaluation",
    "run_unified_evaluation",
    "verify_preintervention_assignment_seal",
    "zscore_responses",
]
