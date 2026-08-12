"""
SignalData: Unified Signal Container for Explainable Fault Diagnosis

This module provides a standardized data structure for handling signals and features
across different models and explanation methods in the Explainable FD Toolkit.
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import torch
from pathlib import Path
import json


class SignalData:
    """
    统一信号与特征的数据容器

    This class provides a standardized interface for handling raw signals,
    processed features, and metadata in fault diagnosis tasks.
    """

    def __init__(
        self,
        raw_signal: Union[np.ndarray, torch.Tensor],
        sampling_rate: int,
        metadata: Optional[Dict[str, Any]] = None,
        processed_features: Optional[Union[np.ndarray, torch.Tensor]] = None,
        time_stamps: Optional[Union[np.ndarray, torch.Tensor]] = None,
        channel_names: Optional[List[str]] = None,
        label: Optional[Union[int, str]] = None,
    ):
        """
        Initialize SignalData container.

        Args:
            raw_signal: Raw signal data [T] or [C, T] or [batch, C, T]
            sampling_rate: Sampling rate in Hz
            metadata: Additional metadata dictionary
            processed_features: Processed features for explanation (e.g., FFT features)
            time_stamps: Time stamps for the signal samples
            channel_names: Names of signal channels (e.g., ['acc_x', 'acc_y', 'acc_z'])
            label: Fault label or class name
        """
        self.raw_signal = self._to_numpy(raw_signal)
        self.sampling_rate = sampling_rate
        self.metadata = metadata or {}
        self.processed_features = self._to_numpy(processed_features) if processed_features is not None else None
        self.time_stamps = self._to_numpy(time_stamps) if time_stamps is not None else None
        self.channel_names = channel_names
        self.label = label

        # Auto-generate time stamps if not provided
        if self.time_stamps is None:
            self.time_stamps = np.arange(self.raw_signal.shape[-1]) / self.sampling_rate

        # Validate signal shape
        self._validate_signal_shape()

    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor, None]) -> Optional[np.ndarray]:
        """Convert tensor to numpy array if needed."""
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    def _validate_signal_shape(self) -> None:
        """Validate signal data shape."""
        if len(self.raw_signal.shape) < 1:
            raise ValueError("Raw signal must have at least 1 dimension")

        if self.time_stamps is not None and self.time_stamps.shape[-1] != self.raw_signal.shape[-1]:
            raise ValueError("Time stamps length must match signal length")

    def get_shape(self) -> Tuple[int, ...]:
        """Get the shape of the raw signal."""
        return self.raw_signal.shape

    def get_num_channels(self) -> int:
        """Get number of channels in the signal."""
        if len(self.raw_signal.shape) == 1:
            return 1
        elif len(self.raw_signal.shape) == 2:
            # Could be [C, T] or [batch, T]
            return self.raw_signal.shape[0] if self.channel_names is not None else 1
        elif len(self.raw_signal.shape) == 3:
            return self.raw_signal.shape[1]  # [batch, C, T]
        else:
            return self.raw_signal.shape[1]

    def get_length(self) -> int:
        """Get the length of the signal in samples."""
        return self.raw_signal.shape[-1]

    def get_duration(self) -> float:
        """Get the duration of the signal in seconds."""
        return self.get_length() / self.sampling_rate

    def get_channel_data(self, channel_idx: Optional[int] = None) -> np.ndarray:
        """
        Get data for a specific channel.

        Args:
            channel_idx: Channel index, if None, returns the first channel

        Returns:
            Channel data as numpy array
        """
        if len(self.raw_signal.shape) == 1:
            return self.raw_signal
        elif len(self.raw_signal.shape) == 2:
            if self.channel_names is not None:
                # Assume [C, T] format
                return self.raw_signal[channel_idx if channel_idx is not None else 0]
            else:
                # Assume [batch, T] format, return first sample
                return self.raw_signal[0]
        elif len(self.raw_signal.shape) == 3:
            # [batch, C, T] format
            return self.raw_signal[0, channel_idx if channel_idx is not None else 0]
        else:
            raise ValueError(f"Unsupported signal shape: {self.raw_signal.shape}")

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata entry."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self.metadata.get(key, default)

    def set_processed_features(self, features: Union[np.ndarray, torch.Tensor]) -> None:
        """Set processed features for explanation."""
        self.processed_features = self._to_numpy(features)

    def get_processed_features(self) -> Optional[np.ndarray]:
        """Get processed features."""
        return self.processed_features

    def to_dict(self) -> Dict[str, Any]:
        """Convert SignalData to dictionary for serialization."""
        return {
            'raw_signal': self.raw_signal.tolist(),
            'sampling_rate': self.sampling_rate,
            'metadata': self.metadata,
            'processed_features': self.processed_features.tolist() if self.processed_features is not None else None,
            'time_stamps': self.time_stamps.tolist() if self.time_stamps is not None else None,
            'channel_names': self.channel_names,
            'label': self.label,
            'shape': self.get_shape(),
            'num_channels': self.get_num_channels(),
            'duration': self.get_duration()
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Save SignalData to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignalData':
        """Create SignalData from dictionary."""
        raw_signal = np.array(data['raw_signal'])
        processed_features = np.array(data['processed_features']) if data['processed_features'] is not None else None
        time_stamps = np.array(data['time_stamps']) if data['time_stamps'] is not None else None

        return cls(
            raw_signal=raw_signal,
            sampling_rate=data['sampling_rate'],
            metadata=data['metadata'],
            processed_features=processed_features,
            time_stamps=time_stamps,
            channel_names=data['channel_names'],
            label=data['label']
        )

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'SignalData':
        """Load SignalData from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def copy(self) -> 'SignalData':
        """Create a copy of the SignalData object."""
        return SignalData(
            raw_signal=self.raw_signal.copy(),
            sampling_rate=self.sampling_rate,
            metadata=self.metadata.copy(),
            processed_features=self.processed_features.copy() if self.processed_features is not None else None,
            time_stamps=self.time_stamps.copy() if self.time_stamps is not None else None,
            channel_names=self.channel_names.copy() if self.channel_names is not None else None,
            label=self.label
        )

    def get_time_window(self, start_time: float, end_time: float) -> 'SignalData':
        """
        Extract a time window from the signal.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            New SignalData object with the time window
        """
        start_idx = int(start_time * self.sampling_rate)
        end_idx = int(end_time * self.sampling_rate)

        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(self.get_length(), end_idx)

        if start_idx >= end_idx:
            raise ValueError("Invalid time window: start_time >= end_time or out of bounds")

        # Extract signal window
        if len(self.raw_signal.shape) == 1:
            window_signal = self.raw_signal[start_idx:end_idx]
        elif len(self.raw_signal.shape) == 2:
            window_signal = self.raw_signal[:, start_idx:end_idx]
        else:
            window_signal = self.raw_signal[:, :, start_idx:end_idx]

        # Extract time stamps window
        window_time_stamps = self.time_stamps[start_idx:end_idx] if self.time_stamps is not None else None

        return SignalData(
            raw_signal=window_signal,
            sampling_rate=self.sampling_rate,
            metadata=self.metadata.copy(),
            processed_features=None,  # Features need to be recomputed for window
            time_stamps=window_time_stamps,
            channel_names=self.channel_names,
            label=self.label
        )

    def __repr__(self) -> str:
        """String representation of SignalData."""
        shape_info = f"shape={self.get_shape()}"
        duration_info = f"duration={self.get_duration():.2f}s"
        channels_info = f"channels={self.get_num_channels()}"
        label_info = f"label={self.label}" if self.label is not None else "no_label"

        return f"SignalData({shape_info}, {duration_info}, {channels_info}, {label_info})"