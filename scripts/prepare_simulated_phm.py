"""Write explicitly simulated signals for classification, DG and CFM demos.

Three signal classes, three acquisition conditions, and three simulated units per
class. This analytical signal generator is not a calibrated bearing simulator.
CSV loading and all learning remain in the existing PHMFactory owners.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def prepare(output: Path, seed: int = 2026, target_scale: float = 1.0,
            target_label_shift: int = 0) -> Path:
    """Create a new data root; target interventions never modify source records."""
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
                # A declared DC calibration offset keeps the mean-linear demo informative.
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--target-scale", type=float, default=1.0)
    parser.add_argument("--target-label-shift", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args(argv)
    if not np.isfinite(args.target_scale) or args.target_scale <= 0:
        parser.error("--target-scale must be finite and positive")
    metadata = prepare(args.output, args.seed, args.target_scale, args.target_label_shift)
    print(f"simulation_metadata={metadata}")
    print("simulation_records=27")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
