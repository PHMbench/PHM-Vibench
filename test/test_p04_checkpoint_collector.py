from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from scripts.p04.collect_checkpoint_outputs import (
    BLINDING_DOMAIN,
    SCHEMA,
    _blinding_permutation,
    collect_checkpoint_outputs,
)
from scripts.p04.evaluate_role_identification import (
    COLLECTION_PHASE_ORDER,
    build_preintervention_assignment_seal,
    run_unified_evaluation,
)
from src.configs.config_utils import load_config
from src.model_factory import build_model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, object]:
    data_root = tmp_path / "synthetic"
    raw_root = data_root / "raw" / "P04_Synthetic"
    raw_root.mkdir(parents=True)
    mechanisms = (
        "low_frequency",
        "harmonic",
        "impulsive_envelope",
        "aperiodic_residual",
    )
    partition_ids: dict[str, list[int]] = {
        "train": [],
        "optimization_validation": [],
        "identification": [],
        "intervention": [],
    }
    partition_groups: dict[str, list[str]] = {
        partition: [] for partition in partition_ids
    }
    partition_cells: dict[str, list[int]] = {
        partition: [] for partition in partition_ids
    }
    rng = np.random.default_rng(240401)
    records: list[dict[str, object]] = []
    samples: dict[int, np.ndarray] = {}
    next_sample_id = 904000000
    next_cell_id = 0
    for partition in partition_ids:
        frozen_partition = partition in {"identification", "intervention"}
        cells_per_stratum = 5 if frozen_partition else 1
        draws_per_cell = 8 if frozen_partition else 1
        for mechanism_id, mechanism in enumerate(mechanisms):
            for diagnosis in range(4):
                for _ in range(cells_per_stratum):
                    cell_id = next_cell_id
                    group = f"P04_SYN_CELL_{cell_id:04d}"
                    partition_groups[partition].append(group)
                    partition_cells[partition].append(cell_id)
                    next_cell_id += 1
                    for draw in range(draws_per_cell):
                        sample_id = next_sample_id
                        offset = sample_id - 904000000
                        next_sample_id += 1
                        sample = rng.normal(size=(512, 2)).astype(np.float32)
                        file_name = f"sample_{offset:06d}.npy"
                        np.save(raw_root / file_name, sample, allow_pickle=False)
                        samples[sample_id] = sample
                        partition_ids[partition].append(sample_id)
                        records.append(
                            {
                                "Id": sample_id,
                                "Dataset_id": 904,
                                "Domain_id": 0,
                                "Label": diagnosis,
                                "Name": "P04_Synthetic",
                                "File": file_name,
                                "Split_group": group,
                                "Split_stratum": (
                                    f"Y{diagnosis}:M{mechanism_id}"
                                ),
                                "Partition": partition,
                                "Mechanism": mechanism,
                                "Nuisance_cell": cell_id,
                                "Draw": draw,
                            }
                        )
    metadata_path = data_root / "metadata.csv"
    pd.DataFrame(records).to_csv(metadata_path, index=False)

    manifest = {
        "schema_version": 1,
        "strategy": "frozen_partitions",
        "group_key": "Split_group",
        "stratify_key": "Split_stratum",
        "metadata_file_sha256": _sha256(metadata_path),
        "partition_map": {
            "train": "train",
            "val": "optimization_validation",
            "test": "intervention",
        },
        "partitions": {
            partition: {
                "ids": ids,
                "groups": partition_groups[partition],
                "cell_ids": partition_cells[partition],
            }
            for partition, ids in partition_ids.items()
        },
    }
    manifest_path = data_root / "partition_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    generator_source = (
        Path(__file__).resolve().parents[1] / "scripts" / "p04" / "generate_synthetic.py"
    )
    generator_manifest = {
        "schema_id": "p04.synthetic-generator.v1",
        "schema_version": "1.0.0",
        "source": {
            "path": "scripts/p04/generate_synthetic.py",
            "sha256": _sha256(generator_source),
        },
        "content_hashes": {
            "metadata_sha256": _sha256(metadata_path),
            "partition_manifest_sha256": _sha256(manifest_path),
        },
    }
    generator_manifest_path = data_root / "generator_manifest.json"
    generator_manifest_path.write_text(
        json.dumps(generator_manifest, indent=2) + "\n", encoding="utf-8"
    )
    governed_files = sorted(path for path in data_root.rglob("*") if path.is_file())
    artifact_hash_ledger_path = data_root / "artifact_hashes.sha256"
    artifact_hash_ledger_path.write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(data_root).as_posix()}\n"
            for path in governed_files
        ),
        encoding="utf-8",
    )

    config = {
        "pipeline": "Pipeline_01_Fault_Diagnosis",
        "protocol": {"paper_id": "P04", "goal_id": "P04-G050", "arm": "FULL"},
        "environment": {
            "project": "p04_collector_fixture",
            "seed": 42,
            "iterations": 1,
            "output_dir": str(tmp_path / "unused_training_output"),
        },
        "data": {
            "data_dir": str(data_root),
            "metadata_file": "metadata.csv",
            "batch_size": 64,
            "num_workers": 0,
            "normalization": "none",
            "window_size": 512,
            "stride": 512,
            "num_window": 1,
            "dtype": "float32",
            "split": {
                "strategy": "grouped_metadata",
                "group_key": "Split_group",
                "stratify_key": "Split_stratum",
                "seed": 240401,
                "test_policy": "partition",
                "manifest_path": str(manifest_path),
                "manifest_mode": "read_only",
                "manifest_sha256": _sha256(manifest_path),
                "partition_map": {
                    "train": "train",
                    "val": "optimization_validation",
                    "test": "intervention",
                },
            },
        },
        "model": {
            "type": "MoE",
            "name": "M_04_RoleConstrainedMoE",
            "input_dim": 2,
            "num_classes": 4,
            "feature_dim": 8,
            "expert_hidden_channels": 4,
            "router_hidden_dim": 6,
            "dropout": 0.0,
            "routing_temperature": 1.0,
            "router_mode": "learned_prior",
            "expert_representation_mode": "role_constrained",
            "low_cutoff": 0.12,
            "envelope_band": [0.20, 0.80],
            "filter_transition": 0.03,
            "role_prior_strength": 0.50,
            "role_prior_max": 1.0,
            "role_prior_permutation": [0, 1, 2, 3],
            "role_prior_assignment": "aligned",
            "load_balance_weight": 0.01,
            "entropy_floor_weight": 0.01,
            "entropy_floor": 0.25,
        },
        "task": {
            "type": "Default_task",
            "name": "Default_task",
            "loss": "CE",
            "metrics": ["acc"],
            "optimizer": "adamw",
            "lr": 0.001,
            "weight_decay": 0.0001,
            "target_system_id": [904],
        },
        "trainer": {"name": "Default_trainer", "device": "cpu", "gpus": 1},
    }
    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    resolved = load_config(config_path)
    torch.manual_seed(17)
    model = build_model(resolved.model, metadata=pd.DataFrame(records)).eval()
    checkpoint_path = tmp_path / "model.ckpt"
    state_dict = {f"network.{name}": value for name, value in model.state_dict().items()}
    state_dict["metrics.unused_state"] = torch.tensor(1.0)
    torch.save({"state_dict": state_dict, "epoch": 3}, checkpoint_path)
    return {
        "config": config,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "metadata_path": metadata_path,
        "manifest_path": manifest_path,
        "generator_manifest_path": generator_manifest_path,
        "artifact_hash_ledger_path": artifact_hash_ledger_path,
        "partition_ids": partition_ids,
        "samples": samples,
        "model": model,
    }


def test_collects_blinded_single_forward_algebra_and_provenance(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "outputs" / "mechanism_input.npz"
    summary = collect_checkpoint_outputs(
        config_path=fixture["config_path"],
        checkpoint_path=fixture["checkpoint_path"],
        metadata_path=fixture["metadata_path"],
        partition_manifest_path=fixture["manifest_path"],
        output_path=output,
        arm="FULL",
        seed=42,
    )

    assert summary["schema"] == SCHEMA
    assert summary["samples"] == 1280
    assert summary["partition_counts"] == {
        "identification": 640,
        "intervention": 640,
    }
    with np.load(output, allow_pickle=False) as bundle:
        assert bundle["schema"].item() == SCHEMA
        assert bundle["schema_id"].item() == SCHEMA
        assert bundle["blinding_domain"].item() == BLINDING_DOMAIN
        assert bundle["sample_id"].tolist() == [
            *fixture["partition_ids"]["identification"],
            *fixture["partition_ids"]["intervention"],
        ]
        assert bundle["partition"].tolist() == [
            *(["identification"] * 640),
            *(["intervention"] * 640),
        ]
        assert bundle["logits"].shape == (1280, 4)
        assert bundle["routing_weights"].shape == (1280, 4)
        assert bundle["expert_features"].shape == (1280, 4, 8)
        assert bundle["expert_logits"].shape == (1280, 4, 4)
        assert bundle["deleted_logits"].shape == (1280, 4, 4)
        assert bundle["fixed_mass_swap_logits"].shape == (1280, 4, 4, 4)

        logits = bundle["logits"]
        routing = bundle["routing_weights"]
        expert_logits = bundle["expert_logits"]
        np.testing.assert_allclose(
            logits,
            np.sum(routing[..., None] * expert_logits, axis=1),
            rtol=1e-5,
            atol=1e-6,
        )
        for deleted_index in range(4):
            effective = routing.copy()
            effective[:, deleted_index] = 0.0
            effective /= (1.0 - routing[:, deleted_index])[:, None]
            expected_deleted = np.sum(effective[..., None] * expert_logits, axis=1)
            np.testing.assert_allclose(
                bundle["deleted_logits"][:, deleted_index],
                expected_deleted,
                rtol=1e-5,
                atol=1e-6,
            )
            for replacement_index in range(4):
                expected_swap = (
                    logits
                    - routing[:, deleted_index, None]
                    * expert_logits[:, deleted_index]
                    + routing[:, deleted_index, None]
                    * expert_logits[:, replacement_index]
                )
                np.testing.assert_allclose(
                    bundle["fixed_mass_swap_logits"][
                        :, deleted_index, replacement_index
                    ],
                    expected_swap,
                    rtol=1e-5,
                    atol=1e-6,
                )

        permutation = _blinding_permutation("FULL", 42, 4)
        np.testing.assert_array_equal(bundle["blinding_permutation"], permutation)
        designated = bundle["designated_role_to_expert"]
        np.testing.assert_array_equal(permutation[designated], np.arange(4))

        ordered_ids = bundle["sample_id"].tolist()
        model = fixture["model"]
        direct_logits_parts: list[np.ndarray] = []
        direct_routing_parts: list[np.ndarray] = []
        direct_expert_logits_parts: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(ordered_ids), 64):
                batch_ids = ordered_ids[start : start + 64]
                direct_input = torch.from_numpy(
                    np.stack(
                        [fixture["samples"][sample_id] for sample_id in batch_ids]
                    )
                )
                direct_logits, direct_diagnostics = model(
                    direct_input, return_diagnostics=True
                )
                direct_logits_parts.append(direct_logits.numpy())
                direct_routing_parts.append(
                    direct_diagnostics["routing_weights"].numpy()
                )
                direct_expert_logits_parts.append(
                    direct_diagnostics["expert_logits"].numpy()
                )
        np.testing.assert_allclose(
            bundle["logits"],
            np.concatenate(direct_logits_parts, axis=0),
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            bundle["routing_weights"],
            np.concatenate(direct_routing_parts, axis=0)[:, permutation],
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            bundle["expert_logits"],
            np.concatenate(direct_expert_logits_parts, axis=0)[:, permutation],
            rtol=1e-5,
            atol=1e-6,
        )

        identification = bundle["partition"] == "identification"
        expected_seal, expected_seal_sha256 = build_preintervention_assignment_seal(
            bundle["expert_features"][identification],
            bundle["mechanism"][identification],
            bundle["diagnosis"][identification],
            bundle["nuisance_cell"][identification],
            bundle["draw"][identification],
            bundle["sample_id"][identification],
            arm="FULL",
            seed=42,
            require_frozen_design=True,
        )
        expected_seal_json = json.dumps(
            expected_seal,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        assert bundle["assignment_seal_json"].item() == expected_seal_json
        assert bundle["assignment_seal_sha256"].item() == expected_seal_sha256
        assert summary["assignment_seal_sha256"] == expected_seal_sha256
        assert json.loads(bundle["collection_phase_order_json"].item()) == list(
            COLLECTION_PHASE_ORDER
        )
        assert bool(bundle["assignment_sealed_before_intervention_read"].item())

        provenance = json.loads(bundle["provenance_json"].item())
        assert provenance["blinding"]["rand_target_rule"].startswith(
            "canonical_constrained_representation_slots"
        )
        assert provenance["assignment_seal"] == {
            "content": expected_seal,
            "sha256": expected_seal_sha256,
            "canonical_json": expected_seal_json,
        }
        assert provenance["ordering"] == {
            "phases": list(COLLECTION_PHASE_ORDER),
            "observed_phase_events": list(COLLECTION_PHASE_ORDER),
            "assignment_sealed_before_intervention_read": True,
            "intervention_signal_files_read_before_seal": 0,
        }
        assert provenance["intervention"]["router_forward_count_per_batch"] == 1
        assert bundle["config_sha256"].item() == _sha256(fixture["config_path"])
        assert bundle["checkpoint_sha256"].item() == _sha256(
            fixture["checkpoint_path"]
        )
        assert bundle["manifest_sha256"].item() == _sha256(fixture["manifest_path"])
        assert bundle["partition_manifest_sha256"].item() == _sha256(
            fixture["manifest_path"]
        )
        assert bundle["generator_manifest_sha256"].item() == _sha256(
            fixture["generator_manifest_path"]
        )
        assert bundle["artifact_hash_ledger_sha256"].item() == _sha256(
            fixture["artifact_hash_ledger_path"]
        )
        assert bundle["metadata_sha256"].item() == _sha256(fixture["metadata_path"])
        assert all(len(value) == 64 for value in bundle["source_sha256"].tolist())

    evaluator_paths = run_unified_evaluation(output, tmp_path / "evaluation")
    metrics = json.loads(evaluator_paths["metrics"].read_text(encoding="utf-8"))
    assert metrics["provenance"]["assignment_seal_sha256"] == (
        summary["assignment_seal_sha256"]
    )
    assert metrics["provenance"][
        "assignment_seal_verified_before_intervention"
    ] is True

    with pytest.raises(FileExistsError, match="overwrite"):
        collect_checkpoint_outputs(
            config_path=fixture["config_path"],
            checkpoint_path=fixture["checkpoint_path"],
            metadata_path=fixture["metadata_path"],
            partition_manifest_path=fixture["manifest_path"],
            output_path=output,
            arm="FULL",
            seed=42,
        )


def test_rejects_hash_mismatch_and_non_strict_checkpoint(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    config = fixture["config"]
    config["data"]["split"]["manifest_sha256"] = "0" * 64
    mismatched_config = tmp_path / "mismatched.yaml"
    mismatched_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA-256"):
        collect_checkpoint_outputs(
            config_path=mismatched_config,
            checkpoint_path=fixture["checkpoint_path"],
            metadata_path=fixture["metadata_path"],
            partition_manifest_path=fixture["manifest_path"],
            output_path=tmp_path / "mismatch.npz",
            arm="FULL",
            seed=42,
        )

    bad_checkpoint = tmp_path / "bad.ckpt"
    model = fixture["model"]
    state = {f"network.{name}": value for name, value in model.state_dict().items()}
    state.pop(next(iter(state)))
    torch.save({"state_dict": state}, bad_checkpoint)
    with pytest.raises(ValueError, match=r"strict network\.\* checkpoint mapping"):
        collect_checkpoint_outputs(
            config_path=fixture["config_path"],
            checkpoint_path=bad_checkpoint,
            metadata_path=fixture["metadata_path"],
            partition_manifest_path=fixture["manifest_path"],
            output_path=tmp_path / "bad.npz",
            arm="FULL",
            seed=42,
        )

    metadata = pd.read_csv(fixture["metadata_path"])
    first_row = metadata.loc[metadata["Partition"] == "identification"].iloc[0]
    source_path = (
        Path(fixture["metadata_path"]).parent
        / "raw"
        / str(first_row["Name"])
        / str(first_row["File"])
    )
    changed = np.load(source_path, allow_pickle=False)
    changed[0, 0] += np.float32(0.5)
    np.save(source_path, changed, allow_pickle=False)
    with pytest.raises(ValueError, match="artifact hash ledger mismatch"):
        collect_checkpoint_outputs(
            config_path=fixture["config_path"],
            checkpoint_path=fixture["checkpoint_path"],
            metadata_path=fixture["metadata_path"],
            partition_manifest_path=fixture["manifest_path"],
            output_path=tmp_path / "tampered-source.npz",
            arm="FULL",
            seed=42,
        )
