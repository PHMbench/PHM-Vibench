"""
Fault Diagnosis Specific Evaluation Metrics

Provides domain-specific evaluation metrics for explainability methods
in the context of mechanical fault diagnosis.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
from scipy import signal
from scipy.stats import pearsonr, spearmanr
import warnings


class FaultDiagnosisMetrics:
    """
    Collection of fault diagnosis specific metrics for evaluating
    the quality and usefulness of explanations.
    """

    def __init__(self, sampling_rate: float = 1024.0):
        """
        Initialize the metrics calculator.

        Args:
            sampling_rate: Sampling rate of the vibration signals in Hz
        """
        self.sampling_rate = sampling_rate

        # Common fault characteristic frequencies (will be overridden by specific cases)
        self.fault_frequencies = {
            'bearing_bpfo': [],  # Ball Pass Frequency Outer Race
            'bearing_bpfi': [],  # Ball Pass Frequency Inner Race
            'bearing_bsf': [],   # Ball Spin Frequency
            'bearing_ftf': [],   # Fundamental Train Frequency
            'gear_mesh': [],     # Gear Mesh Frequency
            'shaft_1x': [],      # 1x shaft frequency
            'shaft_2x': [],      # 2x shaft frequency
        }

    def set_fault_frequencies(self, frequencies: Dict[str, List[float]]):
        """
        Set fault characteristic frequencies for evaluation.

        Args:
            frequencies: Dictionary mapping fault types to their characteristic frequencies
        """
        self.fault_frequencies.update(frequencies)

    def physical_consistency(self,
                           explanation: Any,
                           known_fault_type: str,
                           tolerance: float = 0.1) -> Dict[str, float]:
        """
        Measure physical consistency of explanations with fault theory.

        This metric evaluates whether the explanation highlights regions
        that are consistent with the physical characteristics of the
        known fault type.

        Args:
            explanation: Explanation object with attribution data
            known_fault_type: The actual fault type in the signal
            tolerance: Frequency tolerance for matching (fraction of frequency)

        Returns:
            Dictionary containing consistency scores
        """
        consistency_scores = {}

        # Get attribution data
        if hasattr(explanation, 'get_data'):
            attributions = explanation.get_data('attributions')
            original_signal = explanation.get_data('original_signal')
        else:
            # Assume explanation is a dictionary
            attributions = explanation.get('attributions')
            original_signal = explanation.get('original_signal')

        if attributions is None or original_signal is None:
            return {'error': 'Missing attribution or signal data'}

        # Convert to numpy if needed
        if isinstance(attributions, torch.Tensor):
            attributions = attributions.detach().cpu().numpy()
        if isinstance(original_signal, torch.Tensor):
            original_signal = original_signal.detach().cpu().numpy()

        # Flatten for analysis
        if len(attributions.shape) > 1:
            attributions = attributions.flatten()
        if len(original_signal.shape) > 1:
            original_signal = original_signal.flatten()

        # Frequency domain analysis
        freq_consistency = self._frequency_domain_consistency(
            original_signal, attributions, known_fault_type, tolerance
        )
        consistency_scores['frequency_consistency'] = freq_consistency

        # Time domain consistency
        time_consistency = self._time_domain_consistency(
            original_signal, attributions, known_fault_type
        )
        consistency_scores['time_consistency'] = time_consistency

        # Energy distribution consistency
        energy_consistency = self._energy_consistency(
            original_signal, attributions, known_fault_type
        )
        consistency_scores['energy_consistency'] = energy_consistency

        # Overall consistency (weighted average)
        weights = {'frequency': 0.4, 'time': 0.3, 'energy': 0.3}
        consistency_scores['overall'] = (
            weights['frequency'] * freq_consistency +
            weights['time'] * time_consistency +
            weights['energy'] * energy_consistency
        )

        return consistency_scores

    def fault_discriminability(self,
                             explanations_by_class: Dict[str, List[Any]]) -> Dict[str, float]:
        """
        Measure how well explanations can distinguish between different fault types.

        This metric evaluates whether explanations for different fault types
        are sufficiently different from each other.

        Args:
            explanations_by_class: Dictionary mapping fault types to list of explanations

        Returns:
            Dictionary containing discriminability scores
        """
        if len(explanations_by_class) < 2:
            return {'error': 'Need at least 2 fault types for discriminability'}

        discriminability_scores = {}

        # Compute average attribution pattern for each class
        class_patterns = {}
        for fault_type, explanations in explanations_by_class.items():
            patterns = []
            for explanation in explanations:
                if hasattr(explanation, 'get_data'):
                    attributions = explanation.get_data('attributions')
                else:
                    attributions = explanation.get('attributions')

                if attributions is not None:
                    if isinstance(attributions, torch.Tensor):
                        attributions = attributions.detach().cpu().numpy()
                    patterns.append(attributions.flatten())

            if patterns:
                class_patterns[fault_type] = np.mean(patterns, axis=0)

        # Compute pairwise discriminability
        discriminability_values = []
        fault_types = list(class_patterns.keys())

        for i, fault1 in enumerate(fault_types):
            for j, fault2 in enumerate(fault_types):
                if i < j:  # Avoid duplicates
                    pattern1 = class_patterns[fault1]
                    pattern2 = class_patterns[fault2]

                    # Compute distance metrics
                    euclidean_dist = np.linalg.norm(pattern1 - pattern2)
                    cosine_dist = 1 - np.dot(pattern1, pattern2) / (
                        np.linalg.norm(pattern1) * np.linalg.norm(pattern2) + 1e-8
                    )

                    discriminability_values.append({
                        'fault_pair': f'{fault1}_vs_{fault2}',
                        'euclidean_distance': euclidean_dist,
                        'cosine_distance': cosine_dist
                    })

        # Summarize discriminability
        if discriminability_values:
            euclidean_scores = [d['euclidean_distance'] for d in discriminability_values]
            cosine_scores = [d['cosine_distance'] for d in discriminability_values]

            discriminability_scores['mean_euclidean_distance'] = np.mean(euclidean_scores)
            discriminability_scores['mean_cosine_distance'] = np.mean(cosine_scores)
            discriminability_scores['std_euclidean_distance'] = np.std(euclidean_scores)
            discriminability_scores['std_cosine_distance'] = np.std(cosine_scores)

            # Overall discriminability (higher is better)
            # Normalize to [0, 1] range approximately
            norm_euclidean = np.clip(np.mean(euclidean_scores) / 10.0, 0, 1)
            norm_cosine = np.clip(np.mean(cosine_scores), 0, 1)
            discriminability_scores['overall'] = (norm_euclidean + norm_cosine) / 2

            discriminability_scores['pairwise_details'] = discriminability_values

        return discriminability_scores

    def temporal_localization(self,
                            explanation: Any,
                            fault_events: List[Tuple[int, int]],
                            tolerance_window: int = 50) -> Dict[str, float]:
        """
        Measure how well explanations localize temporal fault events.

        This metric evaluates whether explanations highlight the correct
        time regions where fault events occur.

        Args:
            explanation: Explanation object with attribution data
            fault_events: List of (start_time, end_time) tuples for fault events
            tolerance_window: Tolerance window in samples for event detection

        Returns:
            Dictionary containing localization scores
        """
        localization_scores = {}

        # Get attribution data
        if hasattr(explanation, 'get_data'):
            attributions = explanation.get_data('attributions')
        else:
            attributions = explanation.get('attributions')

        if attributions is None:
            return {'error': 'Missing attribution data'}

        # Convert to numpy
        if isinstance(attributions, torch.Tensor):
            attributions = attributions.detach().cpu().numpy()

        # Flatten if needed
        if len(attributions.shape) > 1:
            attributions = attributions.flatten()

        # Create binary mask for fault events
        signal_length = len(attributions)
        fault_mask = np.zeros(signal_length)

        for start, end in fault_events:
            start = max(0, min(start, signal_length - 1))
            end = max(0, min(end, signal_length - 1))
            fault_mask[start:end + 1] = 1

        # Create attribution mask (top X% of attributions)
        attribution_threshold = np.percentile(np.abs(attributions), 90)
        attribution_mask = (np.abs(attributions) >= attribution_threshold).astype(float)

        # Compute localization metrics
        # True positive rate: how much of fault events are correctly highlighted
        true_positive_rate = np.sum(attribution_mask * fault_mask) / (np.sum(fault_mask) + 1e-8)

        # Precision: how much of highlighted regions are actually fault events
        precision = np.sum(attribution_mask * fault_mask) / (np.sum(attribution_mask) + 1e-8)

        # F1 score
        f1_score = 2 * precision * true_positive_rate / (precision + true_positive_rate + 1e-8)

        localization_scores['true_positive_rate'] = true_positive_rate
        localization_scores['precision'] = precision
        localization_scores['f1_score'] = f1_score

        # IoU (Intersection over Union)
        intersection = np.sum(attribution_mask * fault_mask)
        union = np.sum(attribution_mask) + np.sum(fault_mask) - intersection
        iou = intersection / (union + 1e-8)
        localization_scores['iou'] = iou

        # Overall localization score
        localization_scores['overall'] = f1_score

        return localization_scores

    def frequency_alignment(self,
                          explanation: Any,
                          target_frequencies: List[float],
                          tolerance: float = 0.05) -> Dict[str, float]:
        """
        Measure alignment of explanations with target frequencies.

        This metric evaluates whether the frequency content of attributions
        aligns with expected fault characteristic frequencies.

        Args:
            explanation: Explanation object with attribution data
            target_frequencies: List of target frequencies in Hz
            tolerance: Frequency tolerance for matching

        Returns:
            Dictionary containing alignment scores
        """
        alignment_scores = {}

        # Get attribution data
        if hasattr(explanation, 'get_data'):
            attributions = explanation.get_data('attributions')
            original_signal = explanation.get_data('original_signal')
        else:
            attributions = explanation.get('attributions')
            original_signal = explanation.get('original_signal')

        if attributions is None:
            return {'error': 'Missing attribution data'}

        # Convert to numpy
        if isinstance(attributions, torch.Tensor):
            attributions = attributions.detach().cpu().numpy()
        if isinstance(original_signal, torch.Tensor):
            original_signal = original_signal.detach().cpu().numpy()

        # Flatten if needed
        if len(attributions.shape) > 1:
            attributions = attributions.flatten()
        if len(original_signal.shape) > 1:
            original_signal = original_signal.flatten()

        # Compute FFT of attributions
        attribution_fft = np.fft.fft(attributions)
        attribution_freq = np.fft.fftfreq(len(attributions), 1/self.sampling_rate)
        attribution_power = np.abs(attribution_fft)

        # Only consider positive frequencies
        pos_mask = attribution_freq > 0
        pos_freq = attribution_freq[pos_mask]
        pos_power = attribution_power[pos_mask]

        # Find dominant frequencies in attribution
        if len(pos_power) > 0:
            # Find peaks
            peaks, _ = signal.find_peaks(pos_power, height=np.max(pos_power) * 0.1)
            dominant_freqs = pos_freq[peaks]

            # Compute alignment with target frequencies
            aligned_freqs = []
            for target_freq in target_frequencies:
                for dom_freq in dominant_freqs:
                    if abs(dom_freq - target_freq) / target_freq <= tolerance:
                        aligned_freqs.append((target_freq, dom_freq))
                        break

            # Alignment scores
            alignment_ratio = len(aligned_freqs) / (len(target_frequencies) + 1e-8)
            power_alignment = self._compute_power_alignment(pos_freq, pos_power, target_frequencies, tolerance)

            alignment_scores['frequency_alignment_ratio'] = alignment_ratio
            alignment_scores['power_alignment'] = power_alignment
            alignment_scores['aligned_frequencies'] = aligned_freqs
            alignment_scores['dominant_frequencies'] = dominant_freqs.tolist()

            # Overall alignment
            alignment_scores['overall'] = (alignment_ratio + power_alignment) / 2

        return alignment_scores

    def _frequency_domain_consistency(self,
                                    signal: np.ndarray,
                                    attributions: np.ndarray,
                                    fault_type: str,
                                    tolerance: float) -> float:
        """Compute frequency domain consistency score."""
        if fault_type not in self.fault_frequencies or not self.fault_frequencies[fault_type]:
            return 0.0  # No reference frequencies available

        # Compute FFTs
        signal_fft = np.fft.fft(signal)
        attr_fft = np.fft.fft(attributions)

        freq = np.fft.fftfreq(len(signal), 1/self.sampling_rate)
        pos_mask = freq > 0
        pos_freq = freq[pos_mask]
        pos_signal_power = np.abs(signal_fft[pos_mask])
        pos_attr_power = np.abs(attr_fft[pos_mask])

        # Check alignment with fault frequencies
        target_freqs = self.fault_frequencies[fault_type]
        aligned_power = 0
        total_power = np.sum(pos_attr_power) + 1e-8

        for target_freq in target_freqs:
            # Find frequency range around target
            freq_mask = np.abs(pos_freq - target_freq) / target_freq <= tolerance
            aligned_power += np.sum(pos_attr_power[freq_mask])

        return aligned_power / total_power

    def _time_domain_consistency(self,
                               signal: np.ndarray,
                               attributions: np.ndarray,
                               fault_type: str) -> float:
        """Compute time domain consistency score."""
        # For bearing faults, we expect high attributions at impact locations
        # Look for correlation between signal envelope and attributions

        # Compute signal envelope
        analytic_signal = signal.hilbert(signal)
        envelope = np.abs(analytic_signal)

        # Compute correlation between envelope and attributions
        correlation, _ = pearsonr(envelope, np.abs(attributions))

        return max(0, correlation)  # Return positive correlation

    def _energy_consistency(self,
                          signal: np.ndarray,
                          attributions: np.ndarray,
                          fault_type: str) -> float:
        """Compute energy distribution consistency score."""
        # Fault events typically have higher energy
        # Check if high-attribution regions correspond to high-energy regions

        # Compute signal energy in sliding windows
        window_size = min(100, len(signal) // 10)
        signal_energy = []
        attr_energy = []

        for i in range(0, len(signal) - window_size + 1, window_size // 2):
            window = signal[i:i + window_size]
            attr_window = attributions[i:i + window_size]

            signal_energy.append(np.sum(window ** 2))
            attr_energy.append(np.sum(np.abs(attr_window)))

        # Compute correlation
        if len(signal_energy) > 1:
            correlation, _ = pearsonr(signal_energy, attr_energy)
            return max(0, correlation)
        else:
            return 0.0

    def _compute_power_alignment(self,
                               frequencies: np.ndarray,
                               power: np.ndarray,
                               target_freqs: List[float],
                               tolerance: float) -> float:
        """Compute power alignment with target frequencies."""
        aligned_power = 0
        total_power = np.sum(power) + 1e-8

        for target_freq in target_freqs:
            # Find frequency range around target
            freq_mask = np.abs(frequencies - target_freq) / target_freq <= tolerance
            aligned_power += np.sum(power[freq_mask])

        return aligned_power / total_power


# Convenience function for quick metric computation
def evaluate_explanation(explanation: Any,
                        signal: np.ndarray,
                        fault_type: str,
                        sampling_rate: float = 1024.0,
                        fault_events: Optional[List[Tuple[int, int]]] = None,
                        target_frequencies: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Evaluate explanation using multiple fault diagnosis specific metrics.

    Args:
        explanation: Explanation object or dictionary
        signal: Original vibration signal
        fault_type: Type of fault in the signal
        sampling_rate: Sampling rate in Hz
        fault_events: List of (start, end) tuples for fault events
        target_frequencies: Expected fault frequencies in Hz

    Returns:
        Dictionary containing evaluation scores
    """
    metrics = FaultDiagnosisMetrics(sampling_rate)

    # Set fault frequencies if provided
    if target_frequencies:
        metrics.set_fault_frequencies({fault_type: target_frequencies})

    results = {}

    # Physical consistency
    physical_consistency = metrics.physical_consistency(explanation, fault_type)
    results.update(physical_consistency)

    # Temporal localization (if fault events provided)
    if fault_events:
        temporal_localization = metrics.temporal_localization(explanation, fault_events)
        results.update(temporal_localization)

    # Frequency alignment (if target frequencies provided)
    if target_frequencies:
        frequency_alignment = metrics.frequency_alignment(explanation, target_frequencies)
        results.update(frequency_alignment)

    return results