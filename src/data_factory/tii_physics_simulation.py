"""Constructed bearing impacts and paired physical interventions."""
from __future__ import annotations

import math
import numpy as np
import torch
from .tii_physical import project_window


def bearing_frequencies(contract: dict) -> dict[str, float]:
    geometry = contract["geometry"]
    ratio = (geometry["ball_diameter_in"] / geometry["pitch_diameter_in"]
             * math.cos(math.radians(geometry["contact_angle_degrees"])))
    rotations = geometry["shaft_frequency_hz"]
    count = geometry["rolling_elements"]
    return {"outer_race": rotations * count * (1 - ratio) / 2,
            "inner_race": rotations * count * (1 + ratio) / 2}

def waveform(contract: dict, label: str, seed: int) -> np.ndarray:
    settings = contract["waveform"]
    rate = settings["sample_rate_hz"]
    count = round(rate * settings["duration_s"])
    times = np.arange(count, dtype=np.float64) / rate
    rng = np.random.default_rng(seed)
    rotations = contract["geometry"]["shaft_frequency_hz"]
    signal = np.zeros(count, dtype=np.float64)
    for amplitude, multiple in zip(settings["background_harmonic_amplitudes"],
                                   settings["background_harmonic_multiples"]):
        signal += amplitude * np.sin(2 * np.pi * multiple * rotations * times + rng.uniform(0, 2 * np.pi))
    if label != "normal":
        period = 1 / bearing_frequencies(contract)[label]
        carrier = settings["carrier_resonance_hz"]
        damping = settings["damping_ratio"]
        decay = 2 * np.pi * damping * carrier
        damped_frequency = carrier * math.sqrt(1 - damping ** 2)
        support = settings["impulse_truncation_time_constants"] / decay
        phase = rng.uniform(0, period)
        first = math.floor((-support - phase) / period)
        last = math.ceil((settings["duration_s"] - phase) / period)
        indices = np.arange(first, last + 1)
        arrival_times = (phase + indices * period
                         + rng.uniform(-settings["impact_jitter_fraction_of_period"] * period,
                                       settings["impact_jitter_fraction_of_period"] * period,
                                       len(indices)))
        for arrival in arrival_times:
            start = max(0, int(math.ceil(arrival * rate)))
            stop = min(count, int(math.ceil((arrival + support) * rate)))
            if start >= stop:
                continue
            delay = times[start:stop] - arrival
            signal[start:stop] += (settings["impact_amplitude"] * np.exp(-decay * delay)
                                   * np.sin(2 * np.pi * damped_frequency * delay))
    signal += rng.normal(0, settings["measurement_noise_std"], count)
    return signal

def project(signal: np.ndarray, contract: dict, *, effective_rate: int, increment: bool) -> dict:
    observation = contract["observation"]
    return project_window(
        torch.from_numpy(signal[:, None]), channel=0,
        native_rate_hz=effective_rate, effective_rate_hz=effective_rate,
        grid_rate_hz=observation["projection_grid_hz"], duration_s=contract["waveform"]["duration_s"],
        num_patches=observation["num_patches"], patch_size=observation["samples_per_patch"],
        common_bands_hz=[tuple(observation["common_band_hz"])],
        increment_bands_hz=[tuple(observation["increment_band_hz"])],
        common_support="fully_usable", increment_support="fully_usable" if increment else "unavailable",
        input_unit=contract["waveform"]["output_unit"], output_unit=contract["waveform"]["output_unit"], unit_scale=1,
        support_basis="tensor_fixture", support_evidence="caller-specified constructed bearing simulation",
    )

def periodic_arrivals(contract: dict, label: str, seed: int) -> np.ndarray:
    """Recover the exact seeded impact times used in the retained waveform."""
    if label == "normal":
        return np.empty(0, dtype=np.float64)
    settings = contract["waveform"]
    rng = np.random.default_rng(seed)
    for _ in zip(settings["background_harmonic_amplitudes"], settings["background_harmonic_multiples"]):
        rng.uniform(0, 2 * np.pi)
    period = 1 / bearing_frequencies(contract)[label]
    decay = 2 * np.pi * settings["damping_ratio"] * settings["carrier_resonance_hz"]
    support = settings["impulse_truncation_time_constants"] / decay
    phase = rng.uniform(0, period)
    first = math.floor((-support - phase) / period)
    last = math.ceil((settings["duration_s"] - phase) / period)
    indices = np.arange(first, last + 1)
    return (phase + indices * period
            + rng.uniform(-settings["impact_jitter_fraction_of_period"] * period,
                          settings["impact_jitter_fraction_of_period"] * period, len(indices)))

def impact_response(contract: dict, arrivals: np.ndarray, carrier_hz: float) -> np.ndarray:
    settings = contract["waveform"]
    rate = settings["sample_rate_hz"]
    count = round(rate * settings["duration_s"])
    times = np.arange(count, dtype=np.float64) / rate
    damping = settings["damping_ratio"]
    decay = 2 * np.pi * damping * carrier_hz
    damped_frequency = carrier_hz * math.sqrt(1 - damping ** 2)
    support = settings["impulse_truncation_time_constants"] / decay
    signal = np.zeros(count, dtype=np.float64)
    for arrival in arrivals:
        start = max(0, int(math.ceil(arrival * rate)))
        stop = min(count, int(math.ceil((arrival + support) * rate)))
        if start < stop:
            delay = times[start:stop] - arrival
            signal[start:stop] += (settings["impact_amplitude"] * np.exp(-decay * delay)
                                   * np.sin(2 * np.pi * damped_frequency * delay))
    return signal

def bearing_interventions(contract: dict, label: str, seed: int, original: np.ndarray) -> tuple[dict, dict]:
    """Construct paired interventions; this function fits no model or parameter."""
    settings = contract["waveform"]
    controls = contract["e2_e4"]["target_interventions"]
    rate = settings["sample_rate_hz"]
    arrivals = periodic_arrivals(contract, label, seed)
    fault = impact_response(contract, arrivals, settings["carrier_resonance_hz"])
    background = original - fault

    shifted_fault = impact_response(contract, arrivals,
                                    settings["carrier_resonance_hz"] * controls["resonance_shift_factor"])
    shifted = background + shifted_fault
    difference = shifted - original
    shift_increment = project(difference, contract, effective_rate=rate, increment=True)["incremental"].numpy().ravel()
    shifted_increment_only = original + shift_increment

    outside_fault = impact_response(contract, arrivals, controls["out_of_observation_resonance_hz"])
    outside_contract = dict(contract, observation=dict(contract["observation"],
                            common_band_hz=controls["out_of_observation_band_hz"]))
    outside = project(outside_fault, outside_contract, effective_rate=rate, increment=False)["common"].numpy().ravel()
    negative = background + outside

    poisson_count = 0
    aperiodic = original.copy()
    fault_matched = original.copy()
    full_fault_replacement = original.copy()
    if label != "normal":
        rng = np.random.default_rng(seed + 1000000)
        decay = 2 * np.pi * settings["damping_ratio"] * settings["carrier_resonance_hz"]
        instant = -settings["impulse_truncation_time_constants"] / decay
        poisson_times = []
        while True:
            instant += rng.exponential(1 / controls["poisson_rate_hz"])
            if instant >= settings["duration_s"]:
                break
            poisson_times.append(instant)
        poisson_count = len(poisson_times)
        noise = impact_response(contract, np.asarray(poisson_times), settings["carrier_resonance_hz"])
        poisson_increment = project(noise, contract, effective_rate=rate, increment=True)["incremental"].numpy().ravel()
        original_increment = project(original, contract, effective_rate=rate, increment=True)["incremental"].numpy().ravel()
        fault_increment = project(fault, contract, effective_rate=rate, increment=True)["incremental"].numpy().ravel()
        original_rms = float(np.sqrt(np.mean(original_increment ** 2)))
        fault_rms = float(np.sqrt(np.mean(fault_increment ** 2)))
        poisson_rms = float(np.sqrt(np.mean(poisson_increment ** 2)))
        if poisson_rms == 0:
            raise ValueError(f"Poisson response has no admitted increment: {seed}")
        aperiodic += original_rms / poisson_rms * poisson_increment - original_increment
        fault_matched += fault_rms / poisson_rms * poisson_increment - fault_increment
        full_fault_replacement = background + fault_rms / poisson_rms * poisson_increment

    variants = {"aperiodic_equal_energy": aperiodic, "aperiodic_fault_energy": fault_matched,
                "aperiodic_full_fault_replacement": full_fault_replacement,
                "resonance_shift_physical": shifted,
                "resonance_shift_increment_only": shifted_increment_only,
                "background_only": background, "out_of_observation": negative}
    for factor in controls.get('gain_factors', []):
        factor = float(factor)
        if not math.isfinite(factor) or factor <= 0 or factor == 1:
            raise ValueError('gain factors must be finite, positive and differ from one')
        key = f'gain_{factor:g}'.replace('.', 'p')
        if key in variants:
            raise ValueError('duplicate gain factor')
        variants[key] = original * factor
    return variants, {"periodic_arrivals": len(arrivals), "poisson_arrivals": poisson_count}
