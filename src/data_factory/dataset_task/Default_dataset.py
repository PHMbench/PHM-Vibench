"""Default window dataset used by maintained PHMFactory data paths."""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
from torch.utils.data import Dataset


_WITHIN_ID_TEST_TASK_TYPES = frozenset({"FS", "GFS", "pretrain", "Pretrain"})


class Default_dataset(Dataset):  # THU_006or018_basic
    """Convert one raw time-series file into deterministic train/val/test windows.

    Public callers keep using the historical constructor. ``mode="test"`` means
    the full file for tasks whose test IDs are distinct from training IDs. For
    FS/GFS/pretrain tasks, where the current ID search intentionally reuses the
    same files, ``mode="test"`` selects the held-out tail of the window sequence.

    ``window_intervals`` records the raw ``[start, end)`` interval for every
    retained window. It is used to report whether different splits share raw
    samples; it does not change the sampling or split algorithm.
    """

    def __init__(self, data, metadata, args_data, args_task, mode="train"):
        if not data:
            raise ValueError("dataset input must contain one file ID")

        self.key = list(data.keys())[0]
        self.data = data[self.key]
        self.args_data = args_data
        self.args_task = args_task
        split_config = getattr(args_data, "split", None)
        if isinstance(split_config, Mapping):
            split_strategy = split_config.get("strategy", "legacy_windows")
        else:
            split_strategy = getattr(split_config, "strategy", "legacy_windows")
        self.split_strategy = str(split_strategy)
        if self.split_strategy not in {"legacy_windows", "grouped_metadata"}:
            raise ValueError(
                "data.split.strategy must be 'legacy_windows' or "
                f"'grouped_metadata', got {self.split_strategy!r}"
            )

        requested_mode = str(mode).lower()
        if requested_mode == "valid":
            requested_mode = "val"
        if (
            requested_mode == "test"
            and self.split_strategy == "legacy_windows"
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
        self.num_window = int(args_data.num_window)
        self.window_sampling_strategy = str(
            getattr(args_data, "window_sampling_strategy", "evenly_spaced")
        ).lower()
        raw_stride = getattr(args_data, "stride", None)
        self.stride = None if raw_stride is None else int(raw_stride)
        self.window_sampling_seed = int(
            getattr(args_data, "window_sampling_seed", 0)
        )
        self._validate_window_config(raw_stride)

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
        self.unused_ratio = float(getattr(args_data, "unused_ratio", 0.0))
        self._validate_split_ratios()

        legacy_noise = getattr(args_data, "noise_snr", None)
        if legacy_noise is not None:
            raise ValueError(
                "data.noise_snr is ambiguous because it affects every split. "
                "Use data.train_noise_snr for training augmentation and/or "
                "data.evaluation_noise_snr for an explicit evaluation corruption."
            )
        self.train_noise_snr = self._optional_finite_float(
            getattr(args_data, "train_noise_snr", None),
            "train_noise_snr",
        )
        self.evaluation_noise_snr = self._optional_finite_float(
            getattr(args_data, "evaluation_noise_snr", None),
            "evaluation_noise_snr",
        )
        self.evaluation_noise_seed = int(
            getattr(args_data, "evaluation_noise_seed", 0)
        )
        self._evaluation_noise_rng = np.random.default_rng(
            self.evaluation_noise_seed
        )

        self.processed_data = []
        self.window_intervals: list[tuple[int, int]] = []
        self.prepare_data(metadata)

    def _validate_window_config(self, raw_stride) -> None:
        if self.window_size <= 0:
            raise ValueError(
                f"data.window_size must be positive, got {self.window_size}."
            )
        if self.num_window <= 0:
            raise ValueError(
                f"data.num_window must be positive, got {self.num_window}."
            )
        allowed = {"sequential", "random", "evenly_spaced"}
        if self.window_sampling_strategy not in allowed:
            raise ValueError(
                "Unknown window_sampling_strategy "
                f"{self.window_sampling_strategy!r}; choose one of "
                f"{', '.join(sorted(allowed))}."
            )
        if self.window_sampling_strategy == "sequential":
            if self.stride is None or self.stride <= 0:
                raise ValueError(
                    "data.stride must be a positive integer when "
                    "window_sampling_strategy='sequential'."
                )
        elif raw_stride is not None:
            raise ValueError(
                "data.stride is only consumed by "
                "window_sampling_strategy='sequential'. Remove stride or select "
                "the sequential strategy."
            )

    @staticmethod
    def _optional_finite_float(value, name: str):
        if value is None:
            return None
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"data.{name} must be finite, got {value!r}.")
        return result

    def _validate_split_ratios(self) -> None:
        ratios = {
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "unused_ratio": self.unused_ratio,
        }
        invalid = {
            name: value
            for name, value in ratios.items()
            if not math.isfinite(value) or not 0 <= value <= 1
        }
        if invalid:
            raise ValueError(f"data split ratios must be within [0, 1]: {invalid}")
        total = sum(ratios.values())
        if not math.isclose(total, 1.0, abs_tol=1e-8):
            raise ValueError(
                "data train_ratio + val_ratio + test_ratio + unused_ratio "
                f"must equal 1.0; got {total:.6f}. Declare unused_ratio "
                "explicitly when reserving data."
            )

    def prepare_data(self, metadata=None):
        """Create windows, normalize them, and select the requested split."""
        self._process_single_data(self.data)
        if (
            self.split_strategy == "legacy_windows"
            and self.mode in {"train", "val", "test_holdout"}
        ):
            self._split_data_for_mode()

        self.total_samples = len(self.processed_data)
        self.label = metadata[self.key]["Label"]

    def _append_window(
        self,
        sample_data: np.ndarray,
        start_idx: int,
        end_idx: int,
    ) -> None:
        """Append one window and its exact raw-sample interval."""
        self.processed_data.append(sample_data[start_idx:end_idx])
        self.window_intervals.append((int(start_idx), int(end_idx)))

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
            self._append_window(sample_data, start_idx, end_idx)

    def _random_sampling(self, sample_data, data_length):
        """Create a deterministic random window set shared by every split mode."""
        if data_length == self.window_size:
            self._append_window(sample_data, 0, self.window_size)
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
            self._append_window(sample_data, int(start_idx), int(end_idx))

    def _evenly_spaced_sampling(self, sample_data, data_length):
        """Create windows distributed across the complete file."""
        if data_length == self.window_size:
            self._append_window(sample_data, 0, self.window_size)
        elif self.num_window == 1:
            start_idx = (data_length - self.window_size) // 2
            self._append_window(
                sample_data,
                start_idx,
                start_idx + self.window_size,
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
                self._append_window(
                    sample_data,
                    start_idx,
                    start_idx + self.window_size,
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
        try:
            finite_input = np.isfinite(sample_data).all()
        except TypeError as exc:
            raise ValueError(
                f"data ID {self.key!r} must contain numeric signal values; "
                f"got dtype {sample_data.dtype}."
            ) from exc
        if not finite_input:
            raise FloatingPointError(
                f"data ID {self.key!r} contains NaN or Inf values after reading."
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
        else:
            self._evenly_spaced_sampling(sample_data, data_length)

        if not self.processed_data:
            raise ValueError(
                f"data ID {self.key!r} produced zero windows. Check "
                "window_size, stride, num_window, and window_sampling_strategy."
            )

        transformed_windows = []
        for window in self.processed_data:
            normalized = self._normalize_window(window)
            transformed = self._maybe_add_noise(normalized)
            if not np.isfinite(transformed).all():
                raise FloatingPointError(
                    f"data ID {self.key!r} produced NaN or Inf during "
                    f"{self.mode} preprocessing."
                )
            transformed_windows.append(transformed)
        self.processed_data = transformed_windows

    def _normalize_window(self, window: np.ndarray) -> np.ndarray:
        """Normalize one window using an explicitly per-window method."""
        normalization = str(
            getattr(
                self.args_data,
                "normalization",
                "per_window_standardization",
            )
        ).lower()
        if normalization in {"minmax", "per_window_minmax"}:
            min_vals = np.min(window, axis=0)
            max_vals = np.max(window, axis=0)
            denominator = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
            result = (window - min_vals) / denominator
        elif normalization in {
            "standardization",
            "per_window_standardization",
        }:
            mean_vals = np.mean(window, axis=0)
            std_vals = np.std(window, axis=0)
            result = (window - mean_vals) / (std_vals + 1e-8)
        elif normalization == "none":
            result = window
        else:
            raise ValueError(
                "Unknown normalization method "
                f"{normalization!r}; use per_window_standardization, "
                "per_window_minmax, or none."
            )

        if not np.isfinite(result).all():
            raise FloatingPointError(
                f"data ID {self.key!r} produced NaN or Inf during "
                f"normalization={normalization!r}."
            )
        return result

    def _maybe_add_noise(self, window: np.ndarray) -> np.ndarray:
        """Apply split-specific AWGN only when explicitly configured."""
        if self.mode == "train":
            snr_db = self.train_noise_snr
            rng = np.random
        else:
            snr_db = self.evaluation_noise_snr
            rng = self._evaluation_noise_rng
        if snr_db is None:
            return window

        signal_power = float(np.mean(window.astype(np.float64) ** 2))
        if not math.isfinite(signal_power) or signal_power <= 0:
            raise ValueError(
                f"Cannot apply {snr_db} dB AWGN to data ID {self.key!r}: "
                "signal power must be finite and positive."
            )
        snr_linear = 10.0 ** (snr_db / 10.0)
        if not math.isfinite(snr_linear) or snr_linear <= 0:
            raise ValueError(f"Invalid AWGN SNR value: {snr_db!r} dB.")
        noise_std = math.sqrt(signal_power / snr_linear)
        noise = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=window.shape,
        ).astype(window.dtype)
        return window + noise

    def _split_data_for_mode(self):
        """Select a non-overlapping window-list split and retain its intervals."""
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
            selection = slice(None, train_end)
        elif self.mode == "val":
            selection = slice(train_end, val_end)
        else:
            selection = slice(val_end, test_end)

        self.processed_data = self.processed_data[selection]
        self.window_intervals = self.window_intervals[selection]

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
