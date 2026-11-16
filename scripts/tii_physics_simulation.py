"""Generate the frozen bearing PoC population and run its no-training E1 gate."""
from __future__ import annotations

import argparse
import csv
import shlex
import json
import math
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import scipy
from scipy.signal import hilbert, resample, resample_poly
import torch
import yaml

from src.data_factory.tii_physics_simulation import (bearing_frequencies, waveform, project,
    periodic_arrivals, impact_response, bearing_interventions)
from src.data_factory import build_data
from src.model_factory.ISFM.M_01_ISFM import Model
from src.task_factory.Components.tii_target_head import fit_head, make_episode


CONTRACT = None
OUTPUT = None
ARTIFACTS = None


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)




def prepare_source_interventions() -> None:
    """Prepare and check controls only on the already generated source training records."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Source intervention preparation requires explicit CPU execution")
    torch.set_num_threads(1)
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    rate = contract["waveform"]["sample_rate_hz"]
    with (OUTPUT / "records.csv").open(encoding="utf-8", newline="") as stream:
        records = [row for row in csv.DictReader(stream) if row["array"] == "source1_train"]
    with np.load(ARTIFACTS / "waveforms.npz") as archive:
        originals = archive["source1_train"].copy()
    expected = len(contract["waveform"]["classes"]) * contract["groups"]["source1_broad_train_per_class"]
    if len(records) != expected or len(originals) != expected:
        raise ValueError("The retained source-training inventory does not match the frozen construction")

    interventions: dict[str, list[np.ndarray]] = {}
    rows = []
    for record in records:
        seed, index = int(record["data_seed"]), int(record["array_index"])
        label = record["label_name"]
        original = originals[index]
        if not np.array_equal(waveform(contract, label, seed), original):
            raise ValueError(f"Saved source waveform does not match its recorded seed: {seed}")
        variants, counts = bearing_interventions(contract, label, seed, original)
        for name, signal in variants.items():
            interventions.setdefault(name, []).append(signal)
        base = project(original, contract, effective_rate=rate, increment=True)
        altered = project(variants["aperiodic_equal_energy"], contract, effective_rate=rate, increment=True)
        fault = impact_response(contract, periodic_arrivals(contract, label, seed),
                                contract["waveform"]["carrier_resonance_hz"])
        fault_increment = project(fault, contract, effective_rate=rate, increment=True)["incremental"].numpy().ravel()
        primary = project(variants["aperiodic_fault_energy"], contract, effective_rate=rate, increment=True)
        replacement = project(variants["aperiodic_fault_energy"] - (original - fault_increment), contract,
                              effective_rate=rate, increment=True)["incremental"]
        full_replacement = project(variants["aperiodic_full_fault_replacement"] - variants["background_only"], contract,
                                   effective_rate=rate, increment=True)
        physical_delta = project(variants["resonance_shift_physical"] - original, contract,
                                 effective_rate=rate, increment=True)
        control_delta = project(variants["resonance_shift_increment_only"] - original, contract,
                                effective_rate=rate, increment=True)
        negative_delta = project(variants["out_of_observation"] - variants["background_only"], contract,
                                 effective_rate=rate, increment=True)
        rms_before = float(torch.sqrt(torch.mean(base["incremental"] ** 2)))
        rms_after = float(torch.sqrt(torch.mean(altered["incremental"] ** 2)))
        fault_rms = float(np.sqrt(np.mean(fault_increment ** 2)))
        replacement_rms = float(torch.sqrt(torch.mean(replacement ** 2)))
        fault_energy_error = abs(replacement_rms - fault_rms) / fault_rms if fault_rms else replacement_rms
        full_replacement_rms = float(torch.sqrt(torch.mean(full_replacement["incremental"] ** 2)))
        full_replacement_error = abs(full_replacement_rms - fault_rms) / fault_rms if fault_rms else full_replacement_rms
        rows.append(dict(recording_id=seed, group=int(record["group"]), label=record["label"],
                         label_name=label, **counts,
                         fault_energy_relative_error=fault_energy_error,
                         full_replacement_fault_energy_relative_error=full_replacement_error,
                         full_replacement_common_vs_background_max_abs=float(torch.max(torch.abs(full_replacement["common"]))),
                         fault_energy_common_max_abs=float(torch.max(torch.abs(primary["common"] - base["common"]))),
                         fault_control_total_increment_relative_change=abs(float(torch.sqrt(torch.mean(primary["incremental"] ** 2))) - rms_before) / rms_before,
                         equal_energy_relative_error=abs(rms_after - rms_before) / rms_before,
                         equal_energy_common_max_abs=float(torch.max(torch.abs(altered["common"] - base["common"]))),
                         shifted_physical_common_rms=float(torch.sqrt(torch.mean(physical_delta["common"] ** 2))),
                         shifted_control_common_max_abs=float(torch.max(torch.abs(control_delta["common"]))),
                         outside_common_max_abs=float(torch.max(torch.abs(negative_delta["common"]))),
                         outside_increment_max_abs=float(torch.max(torch.abs(negative_delta["incremental"])))))
    maxima = {key: max(row[key] for row in rows) for key in
              ("fault_energy_relative_error", "full_replacement_fault_energy_relative_error",
               "full_replacement_common_vs_background_max_abs", "fault_energy_common_max_abs", "fault_control_total_increment_relative_change",
               "equal_energy_relative_error", "equal_energy_common_max_abs", "shifted_physical_common_rms",
               "shifted_control_common_max_abs", "outside_common_max_abs", "outside_increment_max_abs")}
    tolerance = 1e-12
    unconstrained = {"shifted_physical_common_rms", "fault_control_total_increment_relative_change"}
    if any(maxima[key] > tolerance for key in maxima if key not in unconstrained):
        raise ValueError(f"A source intervention violated its frozen coordinate or energy equality: {maxima}")
    directory = OUTPUT.parent / "intervention_source_v3"
    artifacts = ARTIFACTS.parent / "intervention_source_v3"
    directory.mkdir(parents=True, exist_ok=False)
    artifacts.mkdir(parents=True, exist_ok=False)
    with (directory / "contract.yaml").open("x", encoding="utf-8") as stream:
        stream.write(CONTRACT.read_text(encoding="utf-8"))
    write_csv(directory / "records.csv", rows)
    with (artifacts / "waveforms.npz").open("xb") as stream:
        np.savez_compressed(stream, **{name: np.stack(waves) for name, waves in interventions.items()})
    summary = dict(status="VERIFIED_SOURCE_CONSTRUCTION_ONLY", records=len(rows),
                   fault_records=sum(row["label_name"] != "normal" for row in rows),
                   target_waveforms_read=0, target_metadata_enumerated=True, model_fits=0, **maxima,
                   interpretation="No target interventions or model outcomes have been evaluated. "
                                  "The fault-energy control preserves background but may change total increment energy; "
                                  "the full-fault replacement changes common to background; "
                                  "the physical resonance shift may also change the common band.",
                   command=shlex.join([sys.executable, *sys.argv]),
                   versions={"python": platform.python_version(), "numpy": np.__version__,
                             "scipy": scipy.__version__, "torch": torch.__version__})
    with (directory / "summary.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("E1 uses CPU and requires CUDA_VISIBLE_DEVICES=''")
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=False)
    ARTIFACTS.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    frequencies = bearing_frequencies(contract)
    rotation = contract["geometry"]["shaft_frequency_hz"]
    kinematic_error = max(abs(frequencies["outer_race"] / rotation - 3.0530),
                          abs(frequencies["inner_race"] / rotation - 4.9469))
    if kinematic_error >= 1e-3:
        raise ValueError(f"reference bearing frequency factors disagree: {kinematic_error}")

    # Verification seeds are disjoint from experiment groups. The same three
    # seeds pair outer- and inner-race examples; they are not six independent runs.
    peak_rows: list[dict] = []
    orthogonalities = []
    settings = contract["e1"]
    rate = contract["waveform"]["sample_rate_hz"]
    for label in ("outer_race", "inner_race"):
        for seed in contract["groups"]["e1_verification_seeds"]:
            raw = waveform(contract, label, seed)
            projected = project(raw, contract, effective_rate=rate, increment=True)
            common = projected["common"].numpy().ravel()
            increment = projected["incremental"].numpy().ravel()
            denominator = np.linalg.norm(common) * np.linalg.norm(increment)
            orthogonalities.append(abs(float(np.dot(common, increment))) / denominator)
            envelope = np.abs(hilbert(increment))
            spectrum = np.abs(np.fft.rfft(envelope - envelope.mean()))
            bins = np.fft.rfftfreq(len(envelope), 1 / rate)
            search = (bins >= settings["envelope_search_hz"][0]) & (bins <= settings["envelope_search_hz"][1])
            estimated = float(bins[search][np.argmax(spectrum[search])])
            expected = frequencies[label]
            peak_rows.append(dict(label=label, seed=seed, expected_hz=expected,
                                  measured_envelope_peak_hz=estimated, absolute_error_hz=abs(estimated - expected)))
    max_peak_error = max(row["absolute_error_hz"] for row in peak_rows)
    max_orthogonality = max(orthogonalities)
    if max_peak_error > settings["envelope_peak_tolerance_hz"]:
        raise ValueError(f"E1 envelope peak outside frozen tolerance: {max_peak_error}")
    if max_orthogonality > settings["projector_normalized_inner_product_max"]:
        raise ValueError(f"E1 frequency projectors are not orthogonal: {max_orthogonality}")

    t = np.arange(rate, dtype=np.float64) / rate
    high_tone = np.sin(2 * np.pi * settings["antialias_input_sinusoid_hz"] * t)
    downsampled = resample_poly(high_tone, 1, 8, window=("kaiser", 5.0))
    alias_ratio = float(np.sqrt(np.mean(downsampled[16:-16] ** 2)) / np.sqrt(np.mean(high_tone ** 2)))
    unfiltered_ratio = float(np.sqrt(np.mean(high_tone[::8][16:-16] ** 2)) / np.sqrt(np.mean(high_tone ** 2)))
    if unfiltered_ratio < settings["unfiltered_decimation_rms_ratio_min"]:
        raise ValueError(f"E1 unfiltered decimation does not expose aliasing: {unfiltered_ratio}")
    if alias_ratio > settings["antialias_interior_rms_ratio_max"]:
        raise ValueError(f"E1 antialias rejection outside frozen tolerance: {alias_ratio}")

    groups = contract["groups"]
    roles = (("source1_train", "source1_broad_train_per_class", "source1", "train", rate),
             ("source1_validation", "source1_broad_validation_per_class", "source1", "validation", rate),
             ("source2_train", "source2_narrow_train_per_class", "source2", "train", 512),
             ("source2_validation", "source2_narrow_validation_per_class", "source2", "validation", 512),
             ("target", "target_broad_per_class", "target", "target", rate))
    records: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    for array_name, count_name, source, split, effective_rate in roles:
        waves = []
        for label_index, label in enumerate(contract["waveform"]["classes"]):
            for index in range(groups[count_name]):
                seed = groups["role_seed_bases"][array_name] + 10000 * label_index + index
                raw = waveform(contract, label, seed)
                observed = raw if effective_rate == rate else resample_poly(raw, 1, 8, window=("kaiser", 5.0))
                records.append(dict(recording_id=seed, group=seed, source=source, role=split,
                                    label=label_index, label_name=label, data_seed=seed,
                                    array=array_name, array_index=len(waves), effective_rate_hz=effective_rate))
                waves.append(observed)
        arrays[array_name] = np.stack(waves)
    target_records = [dict(recording_id=row["recording_id"], group=row["group"], label=row["label"])
                      for row in records if row["source"] == "target"]
    episode = make_episode(target_records, shots=groups["target_support_records_per_class"],
                           seed=groups["target_episode_seed"])
    target_roles = {row["recording_id"]: row["role"] for row in episode}
    for row in records:
        if row["source"] == "target":
            row["role"] = target_roles[row["recording_id"]]
    ids = [row["recording_id"] for row in records]
    group_sets = {role: {row["group"] for row in records if row["role"] == role}
                  for role in ("train", "validation", "support", "query")}
    overlap = max(len(group_sets[a] & group_sets[b]) for i, a in enumerate(group_sets)
                  for b in list(group_sets)[i + 1:])
    if len(set(ids)) != len(ids) or overlap > settings["split_group_overlap_max"]:
        raise ValueError("E1 trajectory or group identifiers overlap across splits")
    compositions = []
    for index in (0, 32, 64):
        increment = project(arrays["source1_train"][index], contract,
                            effective_rate=rate, increment=True)["incremental"].numpy().ravel()
        composed = project(increment, contract, effective_rate=rate, increment=True)["common"].numpy().ravel()
        compositions.append(float(np.linalg.norm(composed) / np.linalg.norm(increment)))
    max_composition = max(compositions)
    if max_composition > settings["projector_normalized_inner_product_max"]:
        raise ValueError(f"E1 common-after-increment projection failed: {max_composition}")
    write_csv(OUTPUT / "records.csv", records)
    write_csv(OUTPUT / "target_episode.csv", episode)
    write_csv(OUTPUT / "envelope_peaks.csv", peak_rows)
    np.savez_compressed(ARTIFACTS / "waveforms.npz", **arrays)
    result = dict(status="PASS", scope="constructed data; no model training", records=len(records),
                  array_shapes={key: list(value.shape) for key, value in arrays.items()},
                  outer_frequency_hz=frequencies["outer_race"], inner_frequency_hz=frequencies["inner_race"],
                  max_reference_factor_error=kinematic_error, max_envelope_peak_error_hz=max_peak_error,
                  antialias_interior_rms_ratio=alias_ratio, unfiltered_decimation_rms_ratio=unfiltered_ratio,
                  max_projector_normalized_inner_product=max_orthogonality,
                  max_projector_composition_norm_ratio=max_composition, projector_composition_examples=3,
                  max_cross_split_group_overlap=overlap, target_support=sum(r["role"] == "support" for r in records),
                  target_query=sum(r["role"] == "query" for r in records),
                  versions={"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "torch": torch.__version__},
                  command=shlex.join([sys.executable, *sys.argv]))
    (OUTPUT / "e1_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def native_source_factory(contract: dict, *, source_ids: tuple[int, ...] = (1, 2),
                          metadata_path: Path | None = None, inventory_path: Path | None = None):
    observation = contract["observation"]
    data = SimpleNamespace(factory_name="default", data_dir=str(ARTIFACTS / "native_sources"),
                           metadata_file=str(metadata_path or OUTPUT / "native_source_metadata.csv"),
                           record_inventory=str(inventory_path or OUTPUT / "native_source_inventory.csv"),
                           cache_dir=str(ARTIFACTS / "native_scratch"),
                           use_cache=True, evidence_kind="tensor_fixture", normalization="source_rms",
                           batch_size=32, source_batch_size=32, num_workers=0, seed=0, rounds=1,
                           duration_s=contract["waveform"]["duration_s"],
                           grid_rate_hz=observation["projection_grid_hz"],
                           num_patches=observation["num_patches"], patch_size=observation["samples_per_patch"],
                           output_unit="m/s^2", common_bands_hz=[observation["common_band_hz"]],
                           increment_bands_hz=[observation["increment_band_hz"]])
    task = SimpleNamespace(type="DG", name="tii_joint", target_system_id=list(source_ids),
                           source_system_ids=list(source_ids), sampling_seed=0)
    return build_data(data, task)


def prepare_inner_episode() -> None:
    """Freeze the source-only pseudo-target episode without fitting an encoder or head."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Inner-episode preparation requires explicit CPU execution")
    torch.set_num_threads(1)
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    selection = contract["e2_e4"]["source_only_inner_selection"]
    with (OUTPUT / "native_source_metadata.csv").open(encoding="utf-8", newline="") as stream:
        metadata = list(csv.DictReader(stream))
    with (OUTPUT / "native_source_inventory.csv").open(encoding="utf-8", newline="") as stream:
        inventory = list(csv.DictReader(stream))
    source1_metadata = [row for row in metadata if int(row["Dataset_id"]) == 1]
    source1_inventory = [row for row in inventory if int(row["dataset_id"]) == 1]
    source2_train = [row for row in inventory if int(row["dataset_id"]) == 2 and row["role"] == "source_train"]
    labels = {int(row["Id"]): int(row["Label"]) for row in metadata if int(row["Dataset_id"]) == 2}
    records = [dict(recording_id=int(row["recording_id"]), group=int(row["group"]), label=labels[int(row["Id"])])
               for row in source2_train]
    episode = make_episode(records, shots=selection["episode_shots_per_class"], seed=selection["episode_seed"])
    directory = OUTPUT.parent / "source_inner_episode"
    directory.mkdir(parents=True, exist_ok=False)
    write_csv(directory / "source1_metadata.csv", source1_metadata)
    write_csv(directory / "source1_inventory.csv", source1_inventory)
    write_csv(directory / "source2_episode.csv", episode)
    with (directory / "contract.yaml").open("x", encoding="utf-8") as stream:
        stream.write(CONTRACT.read_text(encoding="utf-8"))
    factory = native_source_factory(contract, source_ids=(1,), metadata_path=directory / "source1_metadata.csv",
                                    inventory_path=directory / "source1_inventory.csv")
    if set(factory.train_dataset.sources) != {1}:
        raise ValueError("The pseudo-target must be excluded from inner source RMS and encoder data")
    by_role = {role: sum(row["role"] == role for row in episode) for role in ("support", "query")}
    by_class = {str(label): {role: sum(row["label"] == label and row["role"] == role for row in episode)
                             for role in ("support", "query")}
                for label in sorted({row["label"] for row in episode})}
    summary = dict(status="VERIFIED_SOURCE_PREPARATION_ONLY", inner_encoder_sources=[1], pseudo_target_source=2,
                   source1_train_records=len(factory.train_dataset.sources[1]["recording_id"]),
                   source1_validation_records=len(source1_inventory) - len(factory.train_dataset.sources[1]["recording_id"]),
                   pseudo_target_inventory="source2_train_only", pseudo_target_episode=by_role,
                   pseudo_target_per_class=by_class, inner_source_rms=factory.source_rms,
                   excluded_from_inner_rms=["source2_train", "source2_validation", "outer_target"],
                   encoder_fits=0, head_fits=0, selected_alpha=None, target_waveforms_read=0,
                   command=shlex.join([sys.executable, *sys.argv]),
                   versions={"python": platform.python_version(), "numpy": np.__version__,
                             "scipy": scipy.__version__, "torch": torch.__version__})
    with (directory / "summary.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def export_native_sources() -> None:
    """Publish generated source tensors in PHMFactory's existing H5 format.

    The only reader, windowing and source sampler used below are PHMFactory's.
    This operation checks the source-data path without fitting an encoder.
    """
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    assessment = json.loads((OUTPUT / "e1_metrics.json").read_text(encoding="utf-8"))
    if assessment["status"] != "PASS":
        raise ValueError("The corrected E1 gate must pass before native-data integration")
    directory = ARTIFACTS / "native_sources"
    directory.mkdir(exist_ok=False)
    with (OUTPUT / "records.csv").open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["source"] in ("source1", "source2")]
    metadata = []
    inventory = []
    with np.load(ARTIFACTS / "waveforms.npz") as waves:
        for source in (1, 2):
            name = f"SyntheticBearingSource{source}"
            source_rows = [row for row in rows if row["source"] == f"source{source}"]
            with h5py.File(directory / f"{name}.h5", "x") as h5:
                h5.attrs["evidence_kind"] = "constructed_bearing_not_industrial_data"
                for row in source_rows:
                    record_id = int(row["recording_id"])
                    signal = waves[row["array"]][int(row["array_index"])]
                    h5.create_dataset(str(record_id), data=signal[:, None, None])
                    rate = int(row["effective_rate_hz"])
                    metadata.append(dict(Id=record_id, Name=name, Dataset_id=source,
                                         Label=int(row["label"]), Sample_rate=rate, Channel=1,
                                         File=f"synthetic_{record_id}", Domain_id=int(row["group"])))
                    inventory.append(dict(Id=record_id, dataset_id=source, recording_id=record_id,
                                          group=int(row["group"]),
                                          role="source_train" if row["role"] == "train" else "source_val",
                                          native_rate_hz=contract["waveform"]["sample_rate_hz"],
                                          effective_rate_hz=rate, channel=0, input_unit="m/s^2", unit_scale=1.0,
                                          common_support="fully_usable",
                                          increment_support="fully_usable" if source == 1 else "unavailable",
                                          support_basis="tensor_fixture",
                                          support_evidence="independently generated synthetic signal; narrow source digitally antialiased"))
    write_csv(OUTPUT / "native_source_metadata.csv", metadata)
    write_csv(OUTPUT / "native_source_inventory.csv", inventory)
    factory = native_source_factory(contract)
    train = next(iter(factory.get_dataloader("train")))
    validation = list(factory.get_dataloader("val"))
    summary = dict(status="PASS", scope="constructed source DataFactory path only; zero model fits",
                   source_records=len(rows), source_window_inventory=len(factory.window_inventory),
                   source_rms=factory.source_rms,
                   train_batch_shapes={str(source): list(batch["x"].shape) for source, batch in train.items()},
                   validation_records=sum(len(batch["y"]) for batch in validation),
                   h5_format="PHMFactory tii_make_fixture compatible", data_owner="parent ignored artifact directory",
                   command=shlex.join([sys.executable, *sys.argv]))
    (OUTPUT / "native_datafactory_check.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def export_native_target() -> None:
    """Store the constructed target separately, without fitting a statistic."""
    contract = yaml.safe_load(CONTRACT.read_text(encoding='utf-8'))
    if json.loads((OUTPUT / 'e1_metrics.json').read_text(encoding='utf-8'))['status'] != 'PASS':
        raise ValueError('the complete constructed-data gate must pass')
    directory = ARTIFACTS / 'native_target'
    directory.mkdir(exist_ok=False)
    with (OUTPUT / 'records.csv').open(encoding='utf-8', newline='') as stream:
        rows = [row for row in csv.DictReader(stream) if row['source'] == 'target']
    metadata, inventory = [], []
    name = 'SyntheticBearingTarget'
    with np.load(ARTIFACTS / 'waveforms.npz') as waves, h5py.File(directory / f'{name}.h5', 'x') as h5:
        h5.attrs['evidence_kind'] = 'constructed_bearing_not_industrial_data'
        for row in rows:
            record_id = int(row['recording_id'])
            signal = waves[row['array']][int(row['array_index'])]
            h5.create_dataset(str(record_id), data=signal[:, None, None])
            rate = int(row['effective_rate_hz'])
            metadata.append(dict(Id=record_id, Name=name, Dataset_id=3, Label=int(row['label']),
                                 Sample_rate=rate, Channel=1, File=f'synthetic_{record_id}',
                                 Domain_id=int(row['group'])))
            inventory.append(dict(Id=record_id, dataset_id=3, recording_id=record_id,
                group=int(row['group']), role='target', native_rate_hz=contract['waveform']['sample_rate_hz'],
                effective_rate_hz=rate, channel=0, input_unit=contract['waveform']['output_unit'], unit_scale=1.0,
                common_support='fully_usable', increment_support='fully_usable',
                support_basis='tensor_fixture', support_evidence='independently generated synthetic signal'))
    write_csv(OUTPUT / 'native_target_metadata.csv', metadata)
    write_csv(OUTPUT / 'native_target_inventory.csv', inventory)
    print(json.dumps(dict(records=len(rows), dataset_id=3, fitted_statistics=0,
                          command=shlex.join([sys.executable, *sys.argv])), indent=2))


def export_target_interventions(destination: Path, selections: Path, selection_spec: Path) -> None:
    """Write paired query interventions after the source-only decision is fixed."""
    from src.task_factory.Components.tii_target_head import select_source_regularization
    spec = json.loads(selection_spec.read_text(encoding='utf-8'))
    rows = json.loads(selections.read_text(encoding='utf-8'))
    select_source_regularization(rows, encoder_sources=spec['encoder_sources'],
        target_dataset=spec['target_dataset'], source_tasks=spec['source_tasks'], seeds=spec['seeds'],
        selection_arms=spec['selection_arms'], source_pool=spec['source_pool'])
    checkpoints = [Path(path) for path in spec['checkpoints']]
    if not checkpoints or any(not path.is_file() for path in checkpoints):
        raise ValueError('all preselected encoder checkpoints must exist before target interventions')
    repo = Path(__file__).resolve().parents[1]
    destination = destination.resolve()
    if destination == repo or repo in destination.parents:
        raise ValueError('target intervention artifacts must be outside the implementation repository')
    contract = yaml.safe_load(CONTRACT.read_text(encoding='utf-8'))
    if json.loads((OUTPUT / 'e1_metrics.json').read_text(encoding='utf-8'))['status'] != 'PASS':
        raise ValueError('the constructed-data gate must pass before interventions')
    with (OUTPUT / 'records.csv').open(encoding='utf-8', newline='') as stream:
        records = [r for r in csv.DictReader(stream) if r['source'] == 'target']
    episode = make_episode([dict(recording_id=int(r['recording_id']), group=int(r['group']),
                                 label=int(r['label'])) for r in records])
    role_by_id = {r['recording_id']: r['role'] for r in episode}
    if any(r['role'] != role_by_id[int(r['recording_id'])] for r in records):
        raise ValueError('target roles disagree with the fixed record episode')
    destination.mkdir(parents=True, exist_ok=False)
    summary = []
    with np.load(ARTIFACTS / 'waveforms.npz') as archive:
        source = archive['target']
        for record in records:
            index, seed = int(record['array_index']), int(record['data_seed'])
            original = source[index]
            if not np.array_equal(waveform(contract, record['label_name'], seed), original):
                raise ValueError('saved target record does not match its frozen construction')
            variants, counts = bearing_interventions(contract, record['label_name'], seed, original)
            if record['role'] == 'support':
                variants = {name: original for name in variants}
            original_p = project(original, contract, effective_rate=contract['waveform']['sample_rate_hz'],
                                 increment=True)['incremental'].numpy()
            for name, altered in variants.items():
                directory = destination / name
                directory.mkdir(exist_ok=True)
                with h5py.File(directory / 'SyntheticBearingTarget.h5', 'a') as h5:
                    h5.create_dataset(record['recording_id'], data=altered[:, None, None])
                changed_p = project(altered, contract, effective_rate=contract['waveform']['sample_rate_hz'],
                                    increment=True)['incremental'].numpy()
                summary.append(dict(recording_id=int(record['recording_id']), label=int(record['label']),
                    role=record['role'], intervention=name, original_increment_rms=float(np.sqrt(np.mean(original_p**2))),
                    altered_increment_rms=float(np.sqrt(np.mean(changed_p**2))),
                    periodic_arrivals=counts['periodic_arrivals'], poisson_arrivals=counts['poisson_arrivals']))
    write_csv(destination / 'intervention_inventory.csv', summary)
    print(json.dumps(dict(records=len(records), variants=sorted({r['intervention'] for r in summary}),
                          unchanged_support=True, checkpoint_count=len(checkpoints),
                          command=shlex.join([sys.executable, *sys.argv])), indent=2))


def check_native_models() -> None:
    """Exercise both existing matched encoders on the actual DataFactory batch."""
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    factory = native_source_factory(contract)
    batches = next(iter(factory.get_dataloader("train")))
    results = {}
    states = []
    for arm in ("support", "ordinary"):
        torch.manual_seed(0)
        args = SimpleNamespace(type="ISFM", name="M_01_ISFM", embedding="SupportConditionedTokenizer",
                               token_organization=arm, backbone="B_04_Dlinear", task_head="H_01_Linear_cla",
                               output_dim=8, d_model=16, patch_size_L=64, patch_size_C=1,
                               num_patches=64, source_rms=factory.source_rms)
        model = Model(args, factory.target_metadata).eval()
        states.append(model.state_dict())
        outputs = {}
        with torch.inference_mode():
            for source, batch in batches.items():
                logits = model(batch["x"], file_id=batch["file_id"], task_id="classification",
                               incremental=batch["incremental"], availability=batch["availability"])
                if tuple(logits.shape) != (32, 3) or not torch.isfinite(logits).all():
                    raise ValueError(f"{arm}, source {source}: invalid native logits")
                outputs[str(source)] = list(logits.shape)
        results[arm] = dict(logit_shapes=outputs, parameters=sum(p.numel() for p in model.parameters()))
    matched = states[0].keys() == states[1].keys() and all(torch.equal(states[0][key], states[1][key])
                                                            for key in states[0])
    if not matched:
        raise ValueError("S/U native model initializations must match before a paired comparison")
    summary = dict(status="PASS", scope="S/U source DataFactory to native model forward only; zero fits",
                   paired_initialization_equal=matched, models=results,
                   missing_comparisons=["ordinary_single_branch", "native_HSE", "FISHER", "envelope_linear"],
                   command=shlex.join([sys.executable, *sys.argv]))
    with (OUTPUT / "native_model_check.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def check_native_hse() -> None:
    """Exercise the pinned HSE variant on the same admitted source signals."""
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    factory = native_source_factory(contract)
    batches = next(iter(factory.get_dataloader("train")))
    torch.manual_seed(0)
    args = SimpleNamespace(type="ISFM", name="M_01_ISFM", embedding="E_01_HSE",
                           backbone="B_04_Dlinear", task_head="H_01_Linear_cla",
                           output_dim=8, d_model=16, patch_size_L=64, patch_size_C=1,
                           num_patches=64, source_rms=factory.source_rms)
    model = Model(args, factory.target_metadata).eval()
    results = {}
    with torch.inference_mode():
        for source, batch in batches.items():
            union = (batch["x"] + batch["incremental"]).reshape(len(batch["x"]), -1).numpy()
            # HSE's time coordinate uses the effective sampling rate in source
            # metadata. The narrow Fourier projection is already band-limited;
            # return it to the 512 Hz grid without changing its admitted content.
            if source == 2:
                native = resample(union, contract["observation"]["narrow_sample_rate_hz"], axis=1)
                restored = resample(native, contract["observation"]["projection_grid_hz"], axis=1)
                roundtrip_error = float(np.max(np.abs(restored - union)))
                if not np.allclose(restored, union, rtol=1e-5, atol=1e-6):
                    raise ValueError(f"HSE narrow-grid conversion changed admitted content: {roundtrip_error}")
            else:
                native, roundtrip_error = union, 0.0
            x = torch.from_numpy((native / factory.source_rms).astype(np.float32))[:, :, None]
            logits = model(x, file_id=batch["file_id"], task_id="classification")
            if tuple(logits.shape) != (32, 3) or not torch.isfinite(logits).all():
                raise ValueError(f"native HSE, source {source}: invalid logits")
            results[str(source)] = dict(input_shape=list(x.shape), logit_shape=list(logits.shape),
                                        effective_rate_hz=int(contract["observation"][
                                            "broad_sample_rate_hz" if source == 1 else "narrow_sample_rate_hz"]),
                                        admitted_roundtrip_max_abs_error=roundtrip_error)
    summary = dict(status="PASS", scope="pinned HSE source DataFactory to native model forward; zero fits",
                   parameters=sum(p.numel() for p in model.parameters()), sources=results,
                   interpretation="SiLU and deterministic evaluation starts are the pinned implementation; original article states Leaky ReLU",
                   command=shlex.join([sys.executable, *sys.argv]))
    with (OUTPUT / "native_hse_forward.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def check_native_ordinary() -> None:
    """Exercise the existing single-branch patch/DLinear reference."""
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    factory = native_source_factory(contract)
    batches = next(iter(factory.get_dataloader("train")))
    torch.manual_seed(0)
    args = SimpleNamespace(type="ISFM", name="M_01_ISFM", embedding="E_03_Patch",
                           backbone="B_04_Dlinear", task_head="H_01_Linear_cla",
                           output_dim=8, d_model=30, activation="gelu", input_dim=1,
                           window_size=4096, patch_size_L=64, patch_size_C=1,
                           num_patches=64, source_rms=factory.source_rms)
    model = Model(args, factory.target_metadata).eval()
    outputs = {}
    with torch.inference_mode():
        for source, batch in batches.items():
            x = ((batch["x"] + batch["incremental"]) / factory.source_rms).reshape(32, 4096, 1)
            logits = model(x, file_id=batch["file_id"], task_id="classification")
            if tuple(logits.shape) != (32, 3) or not torch.isfinite(logits).all():
                raise ValueError(f"single-branch reference, source {source}: invalid logits")
            outputs[str(source)] = list(logits.shape)
    summary = dict(status="PASS", scope="single-branch DataFactory to native model forward; zero fits",
                   embedding="E_03_Patch", backbone="B_04_Dlinear", hidden_width=30,
                   parameters=sum(p.numel() for p in model.parameters()), logit_shapes=outputs,
                   interpretation="candidate ordinary reference; no training or source-only tuning performed",
                   command=shlex.join([sys.executable, *sys.argv]))
    with (OUTPUT / "native_ordinary_forward.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def check_envelope_linear() -> None:
    """Exercise an envelope-spectrum feature and the existing linear head."""
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    factory = native_source_factory(contract)
    batches = next(iter(factory.get_dataloader("train")))
    rate = contract["observation"]["projection_grid_hz"]
    band = contract["observation"]["common_band_hz"]
    features = {}
    for source, batch in batches.items():
        union = ((batch["x"] + batch["incremental"]) / factory.source_rms).reshape(32, -1).numpy()
        squared_envelope = np.abs(hilbert(union, axis=1)) ** 2
        spectrum = np.abs(np.fft.rfft(squared_envelope, axis=1)) / union.shape[1]
        frequencies = np.fft.rfftfreq(union.shape[1], d=1 / rate)
        features[source] = spectrum[:, (frequencies >= band[0]) & (frequencies < band[1])]
    source = batches[1]
    labels = source["y"].numpy()
    head = fit_head(features[1], labels, source["group"], 0.01)
    outputs = {}
    for source_id in (1, 2):
        logits = head.logits(features[source_id])
        if logits.shape != (32, 3) or not np.isfinite(logits).all():
            raise ValueError(f"envelope-linear reference, source {source_id}: invalid logits")
        outputs[str(source_id)] = dict(feature_shape=list(features[source_id].shape), logit_shape=list(logits.shape))
    with (ARTIFACTS / "envelope_smoke_head.npz").open("xb") as stream:
        np.savez(stream, weight=head.weight, bias=head.bias)
    summary = dict(status="PASS", scope="squared-envelope spectrum plus native linear head on source batches only",
                   input="same source-RMS-normalized admitted union", spectrum_band_hz=band,
                   head_alpha=0.01, head_iterations=head.iterations, head_gradient_inf=head.gradient_inf,
                   source_head_fits=1, encoder_fits=0, target_records_used=0, sources=outputs,
                   interpretation="software smoke only; source-only alpha selection and target comparison not run",
                   command=shlex.join([sys.executable, *sys.argv]))
    with (OUTPUT / "envelope_linear_check.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True, type=Path)
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--artifact_dir', required=True, type=Path)
    actions = parser.add_mutually_exclusive_group()
    for option in ('prepare-source-interventions', 'prepare-inner-episode', 'export-native-sources', 'export-native-target',
                   'export-target-interventions', 'check-native-models', 'check-native-hse', 'check-native-ordinary', 'check-envelope-linear'):
        actions.add_argument('--' + option, action='store_true')
    parser.add_argument('--intervention_dir', type=Path)
    parser.add_argument('--source_selection', type=Path)
    parser.add_argument('--selection_spec', type=Path)
    args = parser.parse_args()
    global CONTRACT, OUTPUT, ARTIFACTS
    CONTRACT, OUTPUT, ARTIFACTS = (args.contract.resolve(), args.output_dir.resolve(), args.artifact_dir.resolve())
    repo = Path(__file__).resolve().parents[1]
    if any(path == repo or repo in path.parents for path in (OUTPUT, ARTIFACTS)):
        raise ValueError('experiment output and artifacts must be outside the implementation repository')
    if args.prepare_source_interventions:
        prepare_source_interventions()
    elif args.prepare_inner_episode:
        prepare_inner_episode()
    elif args.export_native_sources:
        export_native_sources()
    elif args.export_native_target:
        export_native_target()
    elif args.export_target_interventions:
        if not all((args.intervention_dir, args.source_selection, args.selection_spec)):
            parser.error('target interventions require --intervention_dir, --source_selection and --selection_spec')
        export_target_interventions(args.intervention_dir, args.source_selection, args.selection_spec)
    elif args.check_native_models:
        check_native_models()
    elif args.check_native_hse:
        check_native_hse()
    elif args.check_native_ordinary:
        check_native_ordinary()
    elif args.check_envelope_linear:
        check_envelope_linear()
    else:
        main()


if __name__ == '__main__':
    cli()
