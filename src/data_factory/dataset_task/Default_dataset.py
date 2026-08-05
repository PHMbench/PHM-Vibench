"""Default window dataset used by maintained PHMFactory data paths."""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset


_WITHIN_ID_TEST_TASK_TYPES = frozenset({"FS", "GFS", "pretrain", "Pretrain"})


class Default_dataset(Dataset):  # THU_006or018_basic
    """Convert one raw time-series file into deterministic train/val/test windows.

    Public callers keep using the historical constructor. ``mode="test"`` means
    the full file for tasks whose test IDs are distinct from training IDs. For
    FS/GFS/pretrain tasks, where the current ID search intentionally reuses the
    same files, ``mode="test"`` selects the held-out tail of the window sequence.
    """
    def __init__(self, data, metadata, args_data, args_task, mode="train"):
        if not data:
            raise ValueError("dataset input must contain one file ID")

        self.key = list(data.keys())[0]
        self.data = data[self.key]
        self.args_data = args_data
        self.args_task = args_task
        split_cfg = getattr(args_data, "split", None)
        self.split_strategy = getattr(split_cfg, "strategy", "legacy_windows")

        requested_mode = str(mode).lower()
        if requested_mode == "valid":
            requested_mode = "val"
        if (
            requested_mode == "test"
            and getattr(args_task, "type", None) in _WITHIN_ID_TEST_TASK_TYPES
        ):
            requested_mode = "test_holdout"
        if requested_mode not in {"train", "val", "test", "test_holdout"}:
            raise ValueError(
                "mode must be one of train, val/valid, test; "
                f"got {mode!r}"
            )
        self.mode = requested_mode

        self.window_size = int(args_data.window_size)
        self.stride = int(args_data.stride)
        self.num_window = int(args_data.num_window)
        self.window_sampling_strategy = getattr(
            args_data,
            "window_sampling_strategy",
            "evenly_spaced",
        )
        self.window_sampling_seed = int(
            getattr(args_data, "window_sampling_seed", 0)
        )

        self.train_ratio = float(getattr(args_data, "train_ratio", 0.8))
        self.val_ratio = float(
            getattr(args_data, "val_ratio", max(0.0, 1.0 - self.train_ratio))
        )
        self.test_ratio = float(
            getattr(
                args_data,
                "test_ratio",
                max(0.0, 1.0 - self.train_ratio - self.val_ratio),
            )
        )
        self._validate_split_ratios()

        self.processed_data = []
        self.prepare_data(metadata)

    def _validate_split_ratios(self) -> None:
        ratios = {
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
        }
        invalid = {
            name: value
            for name, value in ratios.items()
            if not 0 <= value <= 1
        }
        if invalid:
            raise ValueError(f"data split ratios must be within [0, 1]: {invalid}")
        total = sum(ratios.values())
        if total > 1.0 + 1e-8:
            raise ValueError(
                "data split ratios must sum to at most 1.0; "
                f"got train+val+test={total:.6f}"
            )

    def prepare_data(self, metadata=None):
        """Create windows, normalize them, and select the requested split."""
        self._process_single_data(self.data)
        if (
            self.mode in {"train", "val", "test_holdout"}
            and self.split_strategy == "legacy_windows"
        ):
            self._split_data_for_mode()

        self.total_samples = len(self.processed_data)
        self.label = metadata[self.key]["Label"]

    def _sequential_sampling(self, sample_data, data_length):
        """Create windows from left to right."""
        num_samples = max(
            0,
            (data_length - self.window_size) // self.stride + 1,
        )
        num_samples = min(num_samples, self.num_window)

        for index in range(num_samples):
            start_idx = index * self.stride
            end_idx = start_idx + self.window_size
            self.processed_data.append(sample_data[start_idx:end_idx])

    def _random_sampling(self, sample_data, data_length):
        """Create a deterministic random window set shared by every split mode."""
        if data_length == self.window_size:
            self.processed_data.append(sample_data)
            return

        possible_starts = np.arange(data_length - self.window_size + 1)
        if len(possible_starts) <= self.num_window:
            selected_starts = possible_starts
        else:
            rng = np.random.default_rng(self.window_sampling_seed)
            selected_starts = rng.choice(
                possible_starts,
                size=self.num_window,
                replace=False,
            )
            selected_starts.sort()

        for start_idx in selected_starts:
            end_idx = start_idx + self.window_size
            self.processed_data.append(sample_data[start_idx:end_idx])

    def _evenly_spaced_sampling(self, sample_data, data_length):
        """Create windows distributed across the complete file."""
        if self.num_window == 0:
            return
        if data_length == self.window_size:
            self.processed_data.append(sample_data)
        elif self.num_window == 1:
            start_idx = (data_length - self.window_size) // 2
            self.processed_data.append(
                sample_data[start_idx : start_idx + self.window_size]
            )
        else:
            effective_length = data_length - self.window_size
            if effective_length < 0:
                return
            step = effective_length / (self.num_window - 1)
            for index in range(self.num_window):
                start_idx = min(
                    int(round(index * step)),
                    data_length - self.window_size,
                )
                self.processed_data.append(
                    sample_data[start_idx : start_idx + self.window_size]
                )

    def _process_single_data(self, sample_data):
        """Validate one raw array and apply the configured windowing strategy."""
        if not isinstance(sample_data, np.ndarray):
            sample_data = np.asarray(sample_data)

        dtype = getattr(self.args_data, "dtype", None)
        if dtype == "float32":
            sample_data = sample_data.astype(np.float32)
        elif dtype == "float64":
            sample_data = sample_data.astype(np.float64)

        if sample_data.ndim == 3:
            sample_data = sample_data.reshape(sample_data.shape[0], -1)
        if sample_data.ndim not in {1, 2}:
            raise ValueError(
                f"data ID {self.key!r} must be rank 1 or 2 after reader output; "
                f"got shape {sample_data.shape}"
            )

        data_length = len(sample_data)
        if data_length < self.window_size:
            raise ValueError(
                f"data ID {self.key!r} has length {data_length}, shorter than "
                f"data.window_size={self.window_size}. Reduce window_size or "
                "provide a longer signal."
            )

        if self.window_sampling_strategy == "sequential":
            self._sequential_sampling(sample_data, data_length)
        elif self.window_sampling_strategy == "random":
            self._random_sampling(sample_data, data_length)
        elif self.window_sampling_strategy == "evenly_spaced":
            self._evenly_spaced_sampling(sample_data, data_length)
        else:
            raise ValueError(
                "Unknown window_sampling_strategy: "
                f"{self.window_sampling_strategy}"
            )

        if not self.processed_data:
            raise ValueError(
                f"data ID {self.key!r} produced zero windows. Check "
                "window_size, stride, num_window, and window_sampling_strategy."
            )

        normalized_windows = []
        for window in self.processed_data:
            normalized = self._normalize_window(window)
            normalized_windows.append(self._maybe_add_noise(normalized))
        self.processed_data = normalized_windows

    def _normalize_window(self, window: np.ndarray) -> np.ndarray:
        """Normalize one window using the configured per-window method."""
        normalization = getattr(
            self.args_data,
            "normalization",
            "standardization",
        )
        if normalization == "minmax":
            min_vals = np.min(window, axis=0)
            max_vals = np.max(window, axis=0)
            denominator = max_vals - min_vals
            denominator[denominator == 0] = 1
            return (window - min_vals) / denominator
        if normalization == "standardization":
            mean_vals = np.mean(window, axis=0)
            std_vals = np.std(window, axis=0)
            return (window - mean_vals) / (std_vals + 1e-8)
        if normalization == "none":
            return window
        raise ValueError(f"Unknown normalization method: {normalization}")

    def _maybe_add_noise(self, window: np.ndarray) -> np.ndarray:
        """Add AWGN only when ``data.noise_snr`` is explicitly configured."""
        noise_snr = getattr(self.args_data, "noise_snr", None)
        if noise_snr is None:
            return window
        try:
            snr_db = float(noise_snr)
            signal_power = np.mean(window.astype(np.float64) ** 2)
            if signal_power <= 0:
                return window
            snr_linear = 10.0 ** (snr_db / 10.0)
            noise_power = signal_power / snr_linear
            noise_std = np.sqrt(noise_power)
            noise = np.random.normal(
                loc=0.0,
                scale=noise_std,
                size=window.shape,
            ).astype(window.dtype)
            return window + noise
        except Exception:
            return window

    def _split_data_for_mode(self):
        """Select a non-overlapping split from one deterministic window list."""
        if not self.processed_data:
            return

        total_samples = len(self.processed_data)
        train_end = int(round(self.train_ratio * total_samples))
        val_end = int(
            round((self.train_ratio + self.val_ratio) * total_samples)
        )
        test_end = int(
            round(
                (
                    self.train_ratio
                    + self.val_ratio
                    + self.test_ratio
                )
                * total_samples
            )
        )
        train_end = min(max(train_end, 0), total_samples)
        val_end = min(max(val_end, train_end), total_samples)
        test_end = min(max(test_end, val_end), total_samples)

        if self.mode == "train":
            self.processed_data = self.processed_data[:train_end]
        elif self.mode == "val":
            self.processed_data = self.processed_data[train_end:val_end]
        elif self.mode == "test_holdout":
            self.processed_data = self.processed_data[val_end:test_end]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"索引 {idx} 超出范围")

        return {
            "x": self.processed_data[idx],
            "y": self.label,
        }


class classification_dataset(Default_dataset):
    def __init__(self, data, metadata, args_data, args_task, mode="train"):
        super().__init__(data, metadata, args_data, args_task, mode)


class RUL_dataset(Default_dataset):
    def __init__(self, data, metadata, args_data, args_task, mode="train"):
        super().__init__(data, metadata, args_data, args_task, mode)


class Anomaly_dataset(Default_dataset):
    def __init__(self, data, metadata, args_data, args_task, mode="train"):
        super().__init__(data, metadata, args_data, args_task, mode)


class DigitalTwin_dataset(Default_dataset):
    def __init__(self, data, metadata, args_data, args_task, mode="train"):
        super().__init__(data, metadata, args_data, args_task, mode)


class FM_dataset(Default_dataset):
    def __init__(self, data, metadata, args_data, args_task, mode="train"):
        super().__init__(data, metadata, args_data, args_task, mode)
