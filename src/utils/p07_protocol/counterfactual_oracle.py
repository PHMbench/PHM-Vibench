"""Independent, fixed-path oracle for registered P07 E8 counterfactuals.

The oracle deliberately has no gate, score, or model-execution dependency.  It
authenticates an exact :mod:`intervention_registry` record, then composes the
public elementary semantics from :mod:`path_universe` stage by stage.

Dictionary removal needs one explicit boundary.  Removing a dictionary slot
normally causes the learned model to reselect among the remaining candidates;
that outcome cannot be inferred by a model-independent oracle.  Here a removed
slot on the frozen reference path is therefore *elided to identity*.  The audit
record names this fixed-path software-consistency semantics and is permanently
ineligible for evidential, causal, or physical-meaning claims.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Final, Literal, Sequence

import torch

from . import path_universe
from .intervention_registry import (
    INTERPRETATION_SCOPE,
    PROTOCOL_ID as INTERVENTION_REGISTRY_PROTOCOL_ID,
    DictionaryInterventionRecord,
    InterventionRegistry,
    PathInterventionRecord,
    validate_intervention_registry,
)


SCHEMA_VERSION: Final[int] = 2
PROTOCOL_ID: Final[str] = "P07-E8-COUNTERFACTUAL-ORACLE-v2"
ORACLE_SEMANTICS_VERSION: Final[str] = "fixed-path-public-operator-semantics-v1"
REMOVAL_SEMANTICS: Final[str] = "fixed_path_registered_slot_elision_to_identity"
CORRUPTION_MODE: Final[str] = "additive_gaussian_absolute"
CORRUPTION_SEED_SCOPE: Final[str] = (
    "sha256(registry_domain_seed,stage,registered_operator,"
    "root_sample_content_sha256)"
)
COMPLETION_STATE: Final[str] = "complete_all_input_samples_no_filtering"

RecordFamily = Literal["dictionary", "path"]
ExecutionKind = Literal[
    "fixed_path_dictionary_counterfactual",
    "direct_registered_path_counterfactual",
]
StageAction = Literal[
    "dictionary_replay",
    "dictionary_removal_to_identity",
    "dictionary_registered_replacement",
    "path_replay",
    "path_deletion_to_identity",
    "path_registered_replacement",
]
InterventionRecord = DictionaryInterventionRecord | PathInterventionRecord

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ALLOWED_ACTIONS: Final[frozenset[str]] = frozenset(
    {
        "dictionary_replay",
        "dictionary_removal_to_identity",
        "dictionary_registered_replacement",
        "path_replay",
        "path_deletion_to_identity",
        "path_registered_replacement",
    }
)
_DICTIONARY_CONDITIONS: Final[frozenset[str]] = frozenset(
    {
        "supported_base",
        "serialization_sham",
        "essential_operator_removal",
        "matched_absent_operator_removal_control",
        "wrong_dictionary_derangement",
        "operator_output_corruption",
        "ma5_expansion_secondary",
    }
)
_PATH_CONDITIONS: Final[frozenset[str]] = frozenset(
    {
        "selected_edge_deletion_to_identity",
        "selected_edge_registered_replacement",
        "unchanged_replay_sham",
        "non_selected_stage_control",
        "registered_equivalent_raw_path_control",
    }
)


@dataclass(frozen=True, slots=True)
class StageExecutionRecord:
    """Immutable trace for one independently executed stage."""

    stage: int
    registered_operator: str
    executed_operator: str
    action: StageAction
    corruption_mode: str | None = None
    corruption_magnitude: float | None = None
    corruption_seed: int | None = None
    corruption_derived_seeds: tuple[int, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "registered_operator": self.registered_operator,
            "executed_operator": self.executed_operator,
            "action": self.action,
            "corruption": (
                None
                if self.corruption_mode is None
                else {
                    "mode": self.corruption_mode,
                    "magnitude": self.corruption_magnitude,
                    "seed": self.corruption_seed,
                    "derived_seeds": list(self.corruption_derived_seeds),
                }
            ),
        }


@dataclass(frozen=True, slots=True)
class CounterfactualExecutionRecord:
    """Canonical, immutable provenance for one oracle tensor result."""

    schema_version: int
    protocol_id: str
    oracle_semantics_version: str
    removal_semantics: str
    registry_protocol_id: str
    registry_sha256: str
    truth_class_id: str
    truth_class_sha256: str
    selected_path: path_universe.RawPath
    selected_path_id: str
    selected_path_sha256: str
    selected_path_class_id: str
    selected_path_class_sha256: str
    selected_path_semantic_match: bool
    source_record_id: str
    source_record_sha256: str
    record_family: RecordFamily
    condition: str
    execution_kind: ExecutionKind
    original_path: path_universe.RawPath
    effective_path: path_universe.RawPath
    stages: tuple[StageExecutionRecord, ...]
    sample_keys: tuple[str, ...]
    input_sample_sha256: tuple[str, ...]
    output_sample_sha256: tuple[str, ...]
    input_batch_sha256: str
    output_batch_sha256: str
    tensor_shape: tuple[int, int, int]
    tensor_dtype: str
    tensor_device: str
    completion_state: str
    corruption_seed_domain: str
    corruption_seed_scope: str
    interpretation_scope: str
    software_consistency_only: bool
    evidence_eligible: bool
    causal_claim_eligible: bool
    physical_meaning_claimed: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "oracle_semantics_version": self.oracle_semantics_version,
            "removal_semantics": self.removal_semantics,
            "registry_protocol_id": self.registry_protocol_id,
            "registry_sha256": self.registry_sha256,
            "truth_binding": {
                "truth_class_id": self.truth_class_id,
                "truth_class_sha256": self.truth_class_sha256,
            },
            "export_binding": {
                "selected_path": list(self.selected_path),
                "selected_path_id": self.selected_path_id,
                "selected_path_sha256": self.selected_path_sha256,
                "selected_path_class_id": self.selected_path_class_id,
                "selected_path_class_sha256": self.selected_path_class_sha256,
                "selected_path_semantic_match": self.selected_path_semantic_match,
            },
            "source_record_id": self.source_record_id,
            "source_record_sha256": self.source_record_sha256,
            "record_family": self.record_family,
            "condition": self.condition,
            "execution_kind": self.execution_kind,
            "original_path": list(self.original_path),
            "effective_path": list(self.effective_path),
            "stages": [stage.to_payload() for stage in self.stages],
            "sample_keys": list(self.sample_keys),
            "input_sample_sha256": list(self.input_sample_sha256),
            "output_sample_sha256": list(self.output_sample_sha256),
            "input_batch_sha256": self.input_batch_sha256,
            "output_batch_sha256": self.output_batch_sha256,
            "tensor": {
                "shape": list(self.tensor_shape),
                "dtype": self.tensor_dtype,
                "device": self.tensor_device,
            },
            "completion_state": self.completion_state,
            "corruption_seed": {
                "domain": self.corruption_seed_domain,
                "scope": self.corruption_seed_scope,
            },
            "claim_boundary": {
                "software_consistency_only": self.software_consistency_only,
                "evidence_eligible": self.evidence_eligible,
                "causal_claim_eligible": self.causal_claim_eligible,
                "physical_meaning_claimed": self.physical_meaning_claimed,
                "scope": self.interpretation_scope,
            },
        }

    @property
    def execution_sha256(self) -> str:
        return path_universe.canonical_json_sha256(self.to_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_payload(), "execution_sha256": self.execution_sha256}

    def canonical_json(self) -> str:
        return path_universe.canonical_json_bytes(self.to_dict()).decode("utf-8")


@dataclass(frozen=True, slots=True)
class CounterfactualOracleResult:
    """An output tensor plus its immutable, content-bound audit record.

    PyTorch tensors are mutable even inside frozen dataclasses.  Call
    :func:`validate_counterfactual_result` before consuming a retained result;
    it fails if the tensor no longer matches the recorded hashes.
    """

    output: torch.Tensor = field(repr=False, compare=False, hash=False)
    record: CounterfactualExecutionRecord


def _require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hex string.")
    return value


def _raw_sample_sha256(x: torch.Tensor) -> tuple[str, ...]:
    hashes = []
    for sample in x.detach():
        # Flatten before the dtype view.  A size-one trailing dimension may be
        # reported contiguous while retaining a non-unit stride after public
        # operators such as D1, which PyTorch correctly rejects for dtype views.
        raw = bytes(
            sample.contiguous().reshape(-1).view(torch.uint8).cpu().tolist()
        )
        hashes.append(hashlib.sha256(raw).hexdigest())
    return tuple(hashes)


def _batch_sha256(x: torch.Tensor, sample_hashes: tuple[str, ...]) -> str:
    return path_universe.canonical_json_sha256(
        {
            "layout": "batch_length_channels",
            "shape": list(x.shape),
            "dtype": str(x.dtype).removeprefix("torch."),
            "sample_content_sha256": list(sample_hashes),
        }
    )


def _validate_input(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor.")
    # The public identity semantic supplies the authoritative BLC/dtype/finite
    # checks while retaining the exact tensor object and device.
    return path_universe.oracle_apply_operator("I", x)


def _resolve_sample_keys(
    x: torch.Tensor, sample_keys: Sequence[str] | None
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    content_hashes = _raw_sample_sha256(x)
    if sample_keys is None:
        return content_hashes, content_hashes
    if isinstance(sample_keys, (str, bytes)):
        raise TypeError("sample_keys must be a sequence of SHA-256 strings.")
    keys = tuple(sample_keys)
    if len(keys) != int(x.shape[0]):
        raise ValueError("sample_keys must match the batch dimension.")
    for index, key in enumerate(keys):
        _require_sha256(key, f"sample_keys[{index}]")
    if keys != content_hashes:
        raise ValueError(
            "sample_keys must equal the root input sample content SHA-256 values."
        )
    return keys, content_hashes


def _record_family(record: InterventionRecord) -> RecordFamily:
    if isinstance(record, DictionaryInterventionRecord):
        return "dictionary"
    if isinstance(record, PathInterventionRecord):
        return "path"
    raise TypeError(
        "intervention_record must be a DictionaryInterventionRecord or "
        "PathInterventionRecord."
    )


def _authenticate_record(
    registry: InterventionRegistry, intervention_record: InterventionRecord
) -> tuple[InterventionRegistry, InterventionRecord, RecordFamily]:
    authoritative_registry = validate_intervention_registry(registry)
    family = _record_family(intervention_record)
    records: tuple[InterventionRecord, ...]
    if family == "dictionary":
        records = authoritative_registry.dictionary_records
    else:
        records = authoritative_registry.path_records
    matches = tuple(
        record for record in records if record.record_id == intervention_record.record_id
    )
    if len(matches) != 1 or matches[0] != intervention_record:
        raise ValueError(
            "intervention_record is not the exact hash-bound member of registry."
        )
    _require_sha256(intervention_record.record_sha256, "source record hash")
    return authoritative_registry, matches[0], family


def _derived_corruption_seed(
    *, seed: int, stage: int, registered_operator: str, sample_key: str
) -> int:
    # ``seed`` is itself derived under path_universe.CORRUPTION_SEED_DOMAIN;
    # the frozen slot/sample derivation below intentionally matches the public
    # intervention contract while remaining stateless and batch-order invariant.
    material = f"{seed}:{stage}:{registered_operator}:{sample_key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


def _apply_frozen_corruption(
    x: torch.Tensor,
    corruption: Any,
    *,
    sample_keys: tuple[str, ...],
) -> tuple[torch.Tensor, tuple[int, ...]]:
    if corruption.mode != CORRUPTION_MODE:
        raise ValueError(f"Unsupported corruption mode {corruption.mode!r}.")
    if isinstance(corruption.magnitude, bool) or not isinstance(
        corruption.magnitude, (int, float)
    ):
        raise TypeError("Corruption magnitude must be numeric.")
    magnitude = float(corruption.magnitude)
    if not math.isfinite(magnitude) or magnitude <= 0.0:
        raise ValueError("Corruption magnitude must be positive and finite.")
    if isinstance(corruption.seed, bool) or not isinstance(corruption.seed, Integral):
        raise TypeError("Corruption seed must be an integer.")
    seed = int(corruption.seed)
    if seed < 0 or seed >= 2**63:
        raise ValueError("Corruption seed must be in [0, 2**63).")

    derived_seeds = tuple(
        _derived_corruption_seed(
            seed=seed,
            stage=int(corruption.stage),
            registered_operator=str(corruption.registered_operator),
            sample_key=sample_key,
        )
        for sample_key in sample_keys
    )
    noise_samples = []
    for sample, derived_seed in zip(x, derived_seeds):
        generator = torch.Generator(device=x.device)
        generator.manual_seed(derived_seed)
        noise_samples.append(
            torch.randn(
                sample.shape,
                dtype=x.dtype,
                device=x.device,
                generator=generator,
            )
        )
    output = x + magnitude * torch.stack(noise_samples, dim=0)
    return output, derived_seeds


def _require_transition(before: torch.Tensor, after: torch.Tensor, stage: int) -> None:
    if after.shape != before.shape:
        raise RuntimeError(
            f"Oracle stage {stage} changed shape from {tuple(before.shape)} "
            f"to {tuple(after.shape)}."
        )
    if after.dtype != before.dtype or after.device != before.device:
        raise RuntimeError(f"Oracle stage {stage} changed dtype or device.")
    if not bool(torch.isfinite(after).all()):
        raise ValueError(f"Oracle stage {stage} produced non-finite values.")


def _execute_path_record(
    x: torch.Tensor, record: PathInterventionRecord
) -> tuple[torch.Tensor, tuple[StageExecutionRecord, ...]]:
    path_universe.validate_raw_path(record.original_path)
    effective_path = path_universe.validate_raw_path(record.intervened_path)
    current = x
    stages = []
    for stage, (registered, executed) in enumerate(
        zip(record.original_path, effective_path)
    ):
        if executed == registered:
            action: StageAction = "path_replay"
        elif executed == "I":
            action = "path_deletion_to_identity"
        else:
            action = "path_registered_replacement"
        updated = path_universe.oracle_apply_operator(executed, current)
        _require_transition(current, updated, stage)
        stages.append(
            StageExecutionRecord(
                stage=stage,
                registered_operator=registered,
                executed_operator=executed,
                action=action,
            )
        )
        current = updated
    return current, tuple(stages)


def _intervention_maps(
    intervention: Any,
) -> tuple[
    set[tuple[int, str]],
    dict[tuple[int, str], str],
    dict[tuple[int, str], Any],
]:
    if intervention is None:
        return set(), {}, {}
    removed = set(intervention.removed)
    replacements: dict[tuple[int, str], str] = {}
    for stage, registered, executed in intervention.replacements:
        key = (stage, registered)
        if key in replacements:
            raise ValueError("Dictionary intervention repeats a replacement slot.")
        replacements[key] = executed
    corruptions: dict[tuple[int, str], Any] = {}
    for corruption in intervention.corruptions:
        key = (corruption.stage, corruption.registered_operator)
        if key in corruptions:
            raise ValueError("Dictionary intervention repeats a corruption slot.")
        corruptions[key] = corruption
    if removed.intersection(replacements) or removed.intersection(corruptions):
        raise ValueError("Dictionary removal overlaps a replacement or corruption slot.")
    return removed, replacements, corruptions


def _validate_all_stage_removal(record: DictionaryInterventionRecord) -> None:
    if record.condition not in {
        "essential_operator_removal",
        "matched_absent_operator_removal_control",
    }:
        return
    intervention = record.intervention
    if intervention is None:
        raise ValueError("All-stage removal record has no dictionary intervention.")
    operators = (
        record.target_operators
        if record.condition == "essential_operator_removal"
        else record.control_operators
    )
    expected = {
        (stage, operator)
        for stage in range(path_universe.K_STAGES)
        for operator in operators
    }
    if set(intervention.removed) != expected:
        raise ValueError("Removal record does not remove its operator at every stage.")


def _execute_dictionary_record(
    x: torch.Tensor,
    record: DictionaryInterventionRecord,
    *,
    sample_keys: tuple[str, ...],
) -> tuple[torch.Tensor, tuple[StageExecutionRecord, ...]]:
    path = path_universe.validate_raw_path(record.original_path)
    _validate_all_stage_removal(record)
    removed, replacements, corruptions = _intervention_maps(record.intervention)
    current = x
    stages = []
    for stage, registered in enumerate(path):
        key = (stage, registered)
        if key in removed:
            executed = "I"
            action: StageAction = "dictionary_removal_to_identity"
        elif key in replacements:
            executed = replacements[key]
            action = "dictionary_registered_replacement"
        else:
            executed = registered
            action = "dictionary_replay"

        updated = path_universe.oracle_apply_operator(executed, current)
        corruption = corruptions.get(key)
        derived_seeds: tuple[int, ...] = ()
        if corruption is not None:
            updated, derived_seeds = _apply_frozen_corruption(
                updated, corruption, sample_keys=sample_keys
            )
        _require_transition(current, updated, stage)
        stages.append(
            StageExecutionRecord(
                stage=stage,
                registered_operator=registered,
                executed_operator=executed,
                action=action,
                corruption_mode=None if corruption is None else corruption.mode,
                corruption_magnitude=(
                    None if corruption is None else float(corruption.magnitude)
                ),
                corruption_seed=None if corruption is None else int(corruption.seed),
                corruption_derived_seeds=derived_seeds,
            )
        )
        current = updated
    return current, tuple(stages)


def validate_counterfactual_record(
    record: CounterfactualExecutionRecord,
) -> CounterfactualExecutionRecord:
    """Fail closed on schema, hash, trace, sample, or claim-boundary drift."""

    if not isinstance(record, CounterfactualExecutionRecord):
        raise TypeError("record must be a CounterfactualExecutionRecord.")
    if (
        record.schema_version != SCHEMA_VERSION
        or record.protocol_id != PROTOCOL_ID
        or record.oracle_semantics_version != ORACLE_SEMANTICS_VERSION
        or record.removal_semantics != REMOVAL_SEMANTICS
        or record.registry_protocol_id != INTERVENTION_REGISTRY_PROTOCOL_ID
    ):
        raise ValueError("Counterfactual record schema or oracle semantics drifted.")
    _require_sha256(record.registry_sha256, "registry_sha256")
    _require_sha256(record.truth_class_sha256, "truth_class_sha256")
    _require_sha256(record.selected_path_sha256, "selected_path_sha256")
    _require_sha256(record.selected_path_class_sha256, "selected_path_class_sha256")
    _require_sha256(record.source_record_sha256, "source_record_sha256")
    _require_sha256(record.input_batch_sha256, "input_batch_sha256")
    _require_sha256(record.output_batch_sha256, "output_batch_sha256")
    if record.record_family not in {"dictionary", "path"}:
        raise ValueError("Counterfactual record family is invalid.")
    allowed_conditions = (
        _DICTIONARY_CONDITIONS
        if record.record_family == "dictionary"
        else _PATH_CONDITIONS
    )
    if record.condition not in allowed_conditions:
        raise ValueError("Counterfactual condition is not registered for its family.")
    expected_kind = (
        "fixed_path_dictionary_counterfactual"
        if record.record_family == "dictionary"
        else "direct_registered_path_counterfactual"
    )
    if record.execution_kind != expected_kind:
        raise ValueError("Counterfactual execution kind does not match its record family.")
    selected_path = path_universe.validate_raw_path(record.selected_path)
    original = path_universe.validate_raw_path(record.original_path)
    effective = path_universe.validate_raw_path(record.effective_path)
    if original != selected_path:
        raise ValueError("Counterfactual source path is not the registry-selected path.")
    registered_truth_class = next(
        (
            item
            for item in path_universe.enumerate_equivalence_classes()
            if item.class_id == record.truth_class_id
        ),
        None,
    )
    if (
        registered_truth_class is None
        or registered_truth_class.class_sha256 != record.truth_class_sha256
    ):
        raise ValueError("Counterfactual truth-class binding is not registered.")
    registered_path = next(
        (
            item
            for item in path_universe.enumerate_path_records()
            if item.raw_path_id == record.selected_path_id
        ),
        None,
    )
    registered_selected_class = next(
        (
            item
            for item in path_universe.enumerate_equivalence_classes()
            if item.class_id == record.selected_path_class_id
        ),
        None,
    )
    if (
        registered_path is None
        or registered_path.raw_path != selected_path
        or registered_path.raw_path_sha256 != record.selected_path_sha256
        or registered_selected_class is None
        or registered_selected_class.class_sha256
        != record.selected_path_class_sha256
        or registered_path.class_id != registered_selected_class.class_id
        or registered_path not in registered_selected_class.members
    ):
        raise ValueError(
            "Counterfactual selected path or selected semantic class is unregistered."
        )
    if record.selected_path_semantic_match != (
        registered_selected_class.class_id == registered_truth_class.class_id
    ):
        raise ValueError("Counterfactual recovery-status binding drifted.")
    if len(record.stages) != path_universe.K_STAGES:
        raise ValueError("Counterfactual record must contain exactly three stage traces.")
    if (
        len(record.tensor_shape) != 3
        or any(dimension <= 0 for dimension in record.tensor_shape)
        or record.tensor_shape[1] < 2
    ):
        raise ValueError("Counterfactual tensor shape is not a valid nonempty BLC shape.")
    batch_size = record.tensor_shape[0]
    if record.tensor_dtype not in {"float32", "float64"}:
        raise ValueError("Counterfactual tensor dtype is not float32 or float64.")
    if not record.tensor_device:
        raise ValueError("Counterfactual tensor device is empty.")
    if record.completion_state != COMPLETION_STATE:
        raise ValueError("Counterfactual completion state permits filtering or is incomplete.")
    sample_fields = (
        record.sample_keys,
        record.input_sample_sha256,
        record.output_sample_sha256,
    )
    if any(len(values) != batch_size for values in sample_fields):
        raise ValueError("Counterfactual sample hashes do not match the batch size.")
    for label, values in zip(
        ("sample_keys", "input_sample_sha256", "output_sample_sha256"),
        sample_fields,
    ):
        for index, value in enumerate(values):
            _require_sha256(value, f"{label}[{index}]")
    if record.sample_keys != record.input_sample_sha256:
        raise ValueError("Counterfactual sample keys are not root content hashes.")

    for stage, item in enumerate(record.stages):
        if item.stage != stage:
            raise ValueError("Counterfactual stage trace ordering is invalid.")
        if item.registered_operator != original[stage]:
            raise ValueError("Counterfactual registered path and stage trace disagree.")
        if item.executed_operator != effective[stage]:
            raise ValueError("Counterfactual effective path and stage trace disagree.")
        if item.action not in _ALLOWED_ACTIONS:
            raise ValueError("Counterfactual stage action is invalid.")
        if record.record_family == "dictionary" and not item.action.startswith(
            "dictionary_"
        ):
            raise ValueError("Dictionary record contains a path-stage action.")
        if record.record_family == "path" and not item.action.startswith("path_"):
            raise ValueError("Path record contains a dictionary-stage action.")
        if record.record_family == "dictionary":
            if item.action == "dictionary_removal_to_identity":
                expected_action = (
                    "dictionary_removal_to_identity"
                    if item.executed_operator == "I"
                    else "invalid"
                )
            elif item.executed_operator == item.registered_operator:
                expected_action = "dictionary_replay"
            else:
                expected_action = "dictionary_registered_replacement"
        elif item.executed_operator == item.registered_operator:
            expected_action = "path_replay"
        elif item.executed_operator == "I":
            expected_action = "path_deletion_to_identity"
        else:
            expected_action = "path_registered_replacement"
        if item.action != expected_action:
            raise ValueError("Counterfactual stage action disagrees with its operators.")
        if record.record_family == "path" and item.corruption_mode is not None:
            raise ValueError("Direct path counterfactual cannot carry corruption metadata.")
        if item.corruption_mode is None:
            if (
                item.corruption_magnitude is not None
                or item.corruption_seed is not None
                or item.corruption_derived_seeds
            ):
                raise ValueError("Uncorrupted stage carries corruption metadata.")
        else:
            if item.corruption_mode != CORRUPTION_MODE:
                raise ValueError("Counterfactual corruption mode drifted.")
            if (
                item.corruption_magnitude is None
                or not math.isfinite(item.corruption_magnitude)
                or item.corruption_magnitude <= 0.0
            ):
                raise ValueError("Counterfactual corruption magnitude is invalid.")
            if (
                isinstance(item.corruption_seed, bool)
                or not isinstance(item.corruption_seed, int)
                or item.corruption_seed < 0
                or item.corruption_seed >= 2**63
            ):
                raise ValueError("Counterfactual corruption seed is invalid.")
            expected_seeds = tuple(
                _derived_corruption_seed(
                    seed=item.corruption_seed,
                    stage=stage,
                    registered_operator=item.registered_operator,
                    sample_key=key,
                )
                for key in record.sample_keys
            )
            if item.corruption_derived_seeds != expected_seeds:
                raise ValueError("Counterfactual derived corruption seeds drifted.")

    actions = tuple(item.action for item in record.stages)
    corrupted = tuple(item.corruption_mode is not None for item in record.stages)
    if record.record_family == "dictionary":
        if record.condition in {
            "supported_base",
            "serialization_sham",
            "ma5_expansion_secondary",
        } and any(action != "dictionary_replay" for action in actions):
            raise ValueError("Replay/control dictionary condition changed the fixed path.")
        if record.condition in {
            "essential_operator_removal",
            "matched_absent_operator_removal_control",
        } and any(
            action
            not in {"dictionary_replay", "dictionary_removal_to_identity"}
            for action in actions
        ):
            raise ValueError("Dictionary removal condition used a non-removal action.")
        if record.condition == "wrong_dictionary_derangement" and any(
            action
            not in {"dictionary_replay", "dictionary_registered_replacement"}
            for action in actions
        ):
            raise ValueError("Wrong-dictionary condition used an invalid action.")
        if record.condition == "operator_output_corruption":
            expected_corruption = any(operator != "I" for operator in selected_path)
            if any(corrupted) != expected_corruption or any(
                action != "dictionary_replay" for action in actions
            ):
                raise ValueError("Corruption condition disagrees with exported-path slots.")
        elif any(corrupted):
            raise ValueError("Non-corruption dictionary condition carries noise metadata.")
    else:
        if record.condition == "unchanged_replay_sham" and any(
            action != "path_replay" for action in actions
        ):
            raise ValueError("Unchanged path sham changed the executed path.")
        if record.condition in {
            "selected_edge_deletion_to_identity",
            "non_selected_stage_control",
        } and "path_deletion_to_identity" not in actions:
            raise ValueError("Path deletion/control lacks its identity elision.")
        if record.condition == "selected_edge_registered_replacement" and (
            "path_registered_replacement" not in actions
        ):
            raise ValueError("Selected-edge replacement did not change an operator.")

    if (
        record.corruption_seed_domain != path_universe.CORRUPTION_SEED_DOMAIN
        or record.corruption_seed_scope != CORRUPTION_SEED_SCOPE
    ):
        raise ValueError("Counterfactual corruption seed contract drifted.")
    if record.interpretation_scope != INTERPRETATION_SCOPE:
        raise ValueError("Counterfactual interpretation scope drifted.")
    if (
        record.software_consistency_only is not True
        or record.evidence_eligible is not False
        or record.causal_claim_eligible is not False
        or record.physical_meaning_claimed is not False
    ):
        raise ValueError("Counterfactual claim boundary drifted.")
    _require_sha256(record.execution_sha256, "execution_sha256")
    return record


def validate_counterfactual_result(
    result: CounterfactualOracleResult,
    *,
    registry: InterventionRegistry | None = None,
    intervention_record: InterventionRecord | None = None,
) -> CounterfactualOracleResult:
    """Recompute output bindings and optionally reauthenticate source objects."""

    if not isinstance(result, CounterfactualOracleResult):
        raise TypeError("result must be a CounterfactualOracleResult.")
    record = validate_counterfactual_record(result.record)
    output = _validate_input(result.output)
    shape = tuple(int(value) for value in output.shape)
    if shape != record.tensor_shape:
        raise ValueError("Counterfactual output shape no longer matches its record.")
    if str(output.dtype).removeprefix("torch.") != record.tensor_dtype:
        raise ValueError("Counterfactual output dtype no longer matches its record.")
    if str(output.device) != record.tensor_device:
        raise ValueError("Counterfactual output device no longer matches its record.")
    output_sample_hashes = _raw_sample_sha256(output)
    if output_sample_hashes != record.output_sample_sha256 or (
        _batch_sha256(output, output_sample_hashes) != record.output_batch_sha256
    ):
        raise ValueError("Counterfactual output content no longer matches its hashes.")

    if registry is not None:
        validate_intervention_registry(registry)
        if (
            registry.protocol_id != record.registry_protocol_id
            or registry.manifest_sha256 != record.registry_sha256
            or registry.truth_class_id != record.truth_class_id
            or registry.truth_class_sha256 != record.truth_class_sha256
            or registry.selected_path != record.selected_path
            or registry.selected_path_id != record.selected_path_id
            or registry.selected_path_sha256 != record.selected_path_sha256
            or registry.selected_path_class_id != record.selected_path_class_id
            or registry.selected_path_class_sha256
            != record.selected_path_class_sha256
            or registry.selected_path_semantic_match
            != record.selected_path_semantic_match
        ):
            raise ValueError("Counterfactual result is bound to a different registry.")
        source_pool: tuple[InterventionRecord, ...] = (
            registry.dictionary_records
            if record.record_family == "dictionary"
            else registry.path_records
        )
        bound_sources = tuple(
            item for item in source_pool if item.record_id == record.source_record_id
        )
        if (
            len(bound_sources) != 1
            or bound_sources[0].record_sha256 != record.source_record_sha256
            or bound_sources[0].condition != record.condition
            or bound_sources[0].original_path != record.original_path
        ):
            raise ValueError("Counterfactual source-record binding is invalid.")
    if intervention_record is not None:
        family = _record_family(intervention_record)
        if (
            family != record.record_family
            or intervention_record.record_id != record.source_record_id
            or intervention_record.record_sha256 != record.source_record_sha256
            or intervention_record.condition != record.condition
            or intervention_record.original_path != record.original_path
        ):
            raise ValueError("Counterfactual result is bound to a different source record.")
        if registry is not None:
            _authenticate_record(registry, intervention_record)
    return result


def execute_counterfactual_oracle(
    x: torch.Tensor,
    registry: InterventionRegistry,
    intervention_record: InterventionRecord,
    *,
    sample_keys: Sequence[str] | None = None,
) -> CounterfactualOracleResult:
    """Execute one exact registered path/dictionary condition independently.

    ``sample_keys`` is an optional fail-closed assertion: every key must equal
    the SHA-256 of the corresponding root input sample's contiguous raw bytes.
    These keys make frozen corruptions stateless and batch-order invariant.
    """

    source = _validate_input(x)
    registry, intervention_record, family = _authenticate_record(
        registry, intervention_record
    )
    keys, input_sample_hashes = _resolve_sample_keys(source, sample_keys)
    input_batch_hash = _batch_sha256(source, input_sample_hashes)

    if family == "dictionary":
        if not isinstance(intervention_record, DictionaryInterventionRecord):
            raise AssertionError("Authenticated dictionary record changed type.")
        output, stages = _execute_dictionary_record(
            source, intervention_record, sample_keys=keys
        )
        execution_kind: ExecutionKind = "fixed_path_dictionary_counterfactual"
    else:
        if not isinstance(intervention_record, PathInterventionRecord):
            raise AssertionError("Authenticated path record changed type.")
        output, stages = _execute_path_record(source, intervention_record)
        execution_kind = "direct_registered_path_counterfactual"

    _require_transition(source, output, path_universe.K_STAGES - 1)
    output_sample_hashes = _raw_sample_sha256(output)
    effective_path = tuple(stage.executed_operator for stage in stages)
    record = CounterfactualExecutionRecord(
        schema_version=SCHEMA_VERSION,
        protocol_id=PROTOCOL_ID,
        oracle_semantics_version=ORACLE_SEMANTICS_VERSION,
        removal_semantics=REMOVAL_SEMANTICS,
        registry_protocol_id=registry.protocol_id,
        registry_sha256=registry.manifest_sha256,
        truth_class_id=registry.truth_class_id,
        truth_class_sha256=registry.truth_class_sha256,
        selected_path=registry.selected_path,
        selected_path_id=registry.selected_path_id,
        selected_path_sha256=registry.selected_path_sha256,
        selected_path_class_id=registry.selected_path_class_id,
        selected_path_class_sha256=registry.selected_path_class_sha256,
        selected_path_semantic_match=registry.selected_path_semantic_match,
        source_record_id=intervention_record.record_id,
        source_record_sha256=intervention_record.record_sha256,
        record_family=family,
        condition=intervention_record.condition,
        execution_kind=execution_kind,
        original_path=intervention_record.original_path,
        effective_path=path_universe.validate_raw_path(effective_path),
        stages=stages,
        sample_keys=keys,
        input_sample_sha256=input_sample_hashes,
        output_sample_sha256=output_sample_hashes,
        input_batch_sha256=input_batch_hash,
        output_batch_sha256=_batch_sha256(output, output_sample_hashes),
        tensor_shape=tuple(int(value) for value in output.shape),
        tensor_dtype=str(output.dtype).removeprefix("torch."),
        tensor_device=str(output.device),
        completion_state=COMPLETION_STATE,
        corruption_seed_domain=path_universe.CORRUPTION_SEED_DOMAIN,
        corruption_seed_scope=CORRUPTION_SEED_SCOPE,
        interpretation_scope=INTERPRETATION_SCOPE,
        software_consistency_only=True,
        evidence_eligible=False,
        causal_claim_eligible=False,
        physical_meaning_claimed=False,
    )
    result = CounterfactualOracleResult(output=output, record=record)
    return validate_counterfactual_result(
        result, registry=registry, intervention_record=intervention_record
    )


# Short alias for callers that already route through this module.
execute_counterfactual = execute_counterfactual_oracle


__all__ = [
    "CORRUPTION_MODE",
    "CORRUPTION_SEED_SCOPE",
    "COMPLETION_STATE",
    "ORACLE_SEMANTICS_VERSION",
    "PROTOCOL_ID",
    "REMOVAL_SEMANTICS",
    "SCHEMA_VERSION",
    "CounterfactualExecutionRecord",
    "CounterfactualOracleResult",
    "StageExecutionRecord",
    "execute_counterfactual",
    "execute_counterfactual_oracle",
    "validate_counterfactual_record",
    "validate_counterfactual_result",
]
