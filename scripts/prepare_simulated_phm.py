"""Prepare analytical simulation data, never calibrated industrial evidence.

The legacy three-class profile is unchanged. The explicitly selected bearing4
profile is an idealized no-slip impact/resonance reference, not a reproduction
of a calibrated bearing model. Signal loading stays in the existing readers.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


BEARING4_CLASSES = ("healthy", "inner_race", "outer_race", "rolling_element")
BEARING4_RPM = ((900., 900.), (1200., 1200.), (1500., 1500.),
                (1800., 1800.), (900., 1800.), (1800., 900.))
BEARING4_FS = 12800
BEARING4_LENGTH = 16384


def prepare(output: Path, seed: int = 2026, target_scale: float = 1.0,
            target_label_shift: int = 0) -> Path:
    """Preserved legacy profile; its DC offsets are not bearing fault physics."""
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw" / "Dummy_Data"
    raw.mkdir(parents=True)
    rows = []
    sample_rate, length = 2048, 2048
    time = np.arange(length) / sample_rate
    for label in range(3):
        for unit in range(3):
            for domain in range(3):
                file_id = 900001 + label * 9 + unit * 3 + domain
                rng = np.random.default_rng(seed + file_id)
                rate = (24 + 7 * label) * (1 + 0.15 * domain)
                phase = (time * rate + 0.07 * unit) % 1
                impact = np.exp(-20 * phase) * np.sin(2 * np.pi * 350 * time)
                wave = (0.45 * np.sin(2 * np.pi * rate * time)
                        + (0.15 + 0.15 * label) * impact)
                ch1 = (label - 1) * 0.8 + (1 + 0.1 * domain) * wave
                ch2 = 0.6 * ch1 + 0.15 * np.cos(2 * np.pi * rate * time + 0.2)
                signal = np.column_stack((ch1, ch2)) + rng.normal(0, 0.03, (length, 2))
                if domain == 2:
                    signal *= target_scale
                filename = f"sim_c{label}_u{unit}_d{domain}.csv"
                np.savetxt(raw / filename, signal.astype(np.float32), delimiter=",",
                           header="ch1,ch2", comments="", fmt="%.8g")
                observed_label = (label + target_label_shift) % 3 if domain == 2 else label
                rows.append({
                    "Id": file_id, "Dataset_id": 900, "Name": "Dummy_Data",
                    "Type": "simulated_signal", "File": filename, "Visiable": True,
                    "Label": observed_label, "Label_Description": f"simulated_class_{observed_label}",
                    "Domain_id": domain, "Sample_Rate": sample_rate, "Length": length,
                    "Channels": 2, "asset_id": f"sim_c{label}_u{unit}", "run_id": filename,
                    "Data_Origin": "simulation", "Simulation_Seed": seed,
                    "Target_Scale": target_scale if domain == 2 else 1.0,
                    "Target_Label_Shift": target_label_shift if domain == 2 else 0,
                    "Fault_Diagnosis": True, "Anomaly_Detection": False,
                    "Remaining_Life": False, "Digital_Twin_Prediction": False,
                    "Description": "Analytical harmonics/decaying pulses plus calibration offset; no real PHM evidence",
                })
    metadata = output / "metadata_simulated.csv"
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return metadata


def bearing4_signal(unit: int, condition: int, label: int, seed: int = 2026):
    """Return two sensor views of one full trajectory plus evaluator metadata.

    Nine balls, d/D=0.2, alpha=0, stationary outer race, ideal rolling.
    Rolling-element impulses use 2*BSF (two ideal contacts per ball revolution).
    Constant-load assumptions exclude degradation, nonlinear contacts and slip.
    References: doi:10.1016/0022-460X(84)90595-9;
    doi:10.1016/j.ymssp.2010.07.017. Specific numbers are project assumptions.
    """
    if unit not in range(12) or condition not in range(6) or label not in range(4):
        raise ValueError("bearing4 requires unit 0..11, condition 0..5, label 0..3")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    fs, length = BEARING4_FS, BEARING4_LENGTH
    warmup = fs // 10
    machine = np.random.default_rng(np.random.SeedSequence([seed, unit, 0]))
    resonance = float(machine.uniform(2700., 3300.))
    decay = float(machine.uniform(700., 900.))
    rng = np.random.default_rng(np.random.SeedSequence([seed, unit, condition, label, 1]))
    start, end = BEARING4_RPM[condition]
    rpm = np.r_[np.full(warmup, start), np.linspace(start, end, length)]
    turns = np.r_[0., np.cumsum((rpm[:-1] + rpm[1:]) / (120. * fs))]
    phase = float(rng.uniform(0., 1.))
    excitation = np.zeros(len(turns))
    orders = (0., 5.4, 3.6, 4.8)
    event_count = 0
    if label:
        cycles = orders[label] * turns + phase
        levels = np.arange(np.floor(cycles[0]) + 1., np.floor(cycles[-1]) + 1.)
        right = np.searchsorted(cycles, levels)
        fraction = (levels - cycles[right - 1]) / (cycles[right] - cycles[right - 1])
        at_turns = turns[right - 1] + fraction * (turns[right] - turns[right - 1])
        modulation_order = 1. if label == 1 else (0.4 if label == 3 else 0.)
        modulation = (1. + .5 * np.cos(2 * np.pi * modulation_order * at_turns)
                      if modulation_order else np.ones(len(levels)))
        amplitude = float(rng.uniform(.8, 1.2)) * modulation
        # Fractional-sample deposition avoids class-dependent rounding conventions.
        np.add.at(excitation, right - 1, amplitude * (1. - fraction))
        np.add.at(excitation, right, amplitude * fraction)
        event_count = int(np.count_nonzero(right >= warmup))
    lag = np.arange(int(.025 * fs)) / fs
    response = np.exp(-decay * lag) * np.sin(2 * np.pi * resonance * lag)
    vibration = np.convolve(excitation, response, mode="full")[:len(turns)]
    vibration += .10 * np.sin(2 * np.pi * turns + phase)
    vibration += .05 * np.sin(4 * np.pi * turns + phase)
    filtered = np.convolve(vibration, np.array([.25, .50, .25]), mode="full")[:len(turns)]
    # Noise scales and sensor filters never depend on the fault label.
    first = vibration + rng.normal(0., .05, len(turns))
    second = .7 * filtered + rng.normal(0., .08, len(turns))
    signal = np.column_stack((first[warmup:], second[warmup:])).astype(np.float32)
    return signal, {"Resonance_Hz": resonance, "Decay_Per_Second": decay,
                    "RPM_Start": start, "RPM_End": end,
                    "Fault_Order": orders[label], "Impulse_Count": event_count}


def prepare_bearing4(output: Path, seed: int = 2026, pilot: bool = False) -> Path:
    """Write full records before any learner-side windowing or preprocessing."""
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw" / "CSV_Signal"
    raw.mkdir(parents=True)
    units = (0, 6, 8, 10) if pilot else tuple(range(12))
    conditions = (0, 3, 4) if pilot else tuple(range(6))
    rows = []
    for unit in units:
        split = ("train" if unit < 6 else "validation" if unit < 8
                 else "adaptation" if unit < 10 else "evaluation")
        for condition in conditions:
            for label, name in enumerate(BEARING4_CLASSES):
                signal, details = bearing4_signal(unit, condition, label, seed)
                file_id = 99040000 + (unit * 6 + condition) * 4 + label
                filename = f"record_{file_id}.csv"
                np.savetxt(raw / filename, signal, delimiter=",", fmt="%.8g",
                           header="sensor_a,sensor_b", comments="")
                rows.append({"Id": file_id, "Dataset_id": 9904, "Name": "CSV_Signal",
                             "Type": "bearing_simulation", "File": filename,
                             "Visiable": True, "Label": label, "Label_Description": name,
                             "Domain_id": condition, "Partition": split,
                             "asset_id": f"sim_unit_{unit:02d}", "run_id": file_id,
                             "Data_Origin": "simulation", "Simulation_Profile": "bearing4_v1",
                             "Simulation_Seed": seed, "Sample_Rate": BEARING4_FS,
                             "Length": BEARING4_LENGTH, "Channels": 2,
                             "Amplitude_Unit": "arbitrary", "Fault_Diagnosis": True,
                             "Anomaly_Detection": False, "Remaining_Life": False,
                             "Digital_Twin_Prediction": False, **details})
    metadata = output / "metadata_bearing4.csv"
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    specification = {
        "profile": "bearing4_v1", "seed": seed, "pilot": pilot,
        "classes": BEARING4_CLASSES, "rpm_profiles": BEARING4_RPM,
        "sample_rate_hz": BEARING4_FS, "record_length": BEARING4_LENGTH,
        "unit_ids": units, "condition_ids": conditions, "records": len(rows),
        "source_conditions": [0, 1, 2], "target_conditions": [3, 4, 5],
        "geometry": {"balls": 9, "diameter_ratio": .2, "contact_angle_rad": 0.},
        "fault_orders": {"BPFI": 5.4, "BPFO": 3.6, "BSF": 2.4, "FTF": .4},
        "scope": "idealized uncalibrated simulation; no real industrial effectiveness",
        "reader": {"name": "CSV_Signal", "csv_signal_columns": ["sensor_a"]},
        "evaluator_only": ["Label", "asset_id", "run_id", "Fault_Order", "Impulse_Count",
                           "Resonance_Hz", "Decay_Per_Second", "Simulation_Seed"],
    }
    (output / "simulation_spec.json").write_text(json.dumps(specification, indent=2), encoding="utf-8")
    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--profile", choices=("legacy", "bearing4"), default="legacy")
    parser.add_argument("--pilot", action="store_true", help="fixed subset of bearing4, not full evidence")
    parser.add_argument("--target-scale", type=float, default=1.0)
    parser.add_argument("--target-label-shift", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args(argv)
    if not np.isfinite(args.target_scale) or args.target_scale <= 0:
        parser.error("--target-scale must be finite and positive")
    if args.profile == "bearing4":
        if args.target_scale != 1.0 or args.target_label_shift:
            parser.error("legacy target interventions are not part of bearing4_v1")
        metadata = prepare_bearing4(args.output, args.seed, args.pilot)
    else:
        if args.pilot:
            parser.error("--pilot requires --profile bearing4")
        metadata = prepare(args.output, args.seed, args.target_scale, args.target_label_shift)
    print(f"simulation_metadata={metadata}")
    with metadata.open(encoding="utf-8") as handle:
        print(f"simulation_records={sum(1 for _ in csv.DictReader(handle))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
