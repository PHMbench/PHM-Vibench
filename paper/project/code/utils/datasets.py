"""
Data Loading Utilities for 1D-2D Fusion Demo
Minimal implementation that wraps main repository datasets
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from pathlib import Path

import h5py
import pandas as pd


PHM_DATASET_BY_TASK = {
    "CWRU": (1, "RM_001_CWRU"),
    "XJTU": (2, "RM_002_XJTU"),
    "THU_006": (6, "RM_006_THU"),
    "THU_018": (14, "RM_018_THU24"),
}


class OneD2DDataset(Dataset):
    """
    Wrapper dataset that provides 1D signals and creates 2D spectrograms on the fly
    """

    def __init__(self, base_dataset, spectrogram_size=(128, 128)):
        """
        Initialize 1D-2D dataset

        Args:
            base_dataset: Base dataset that returns (signal, label)
            spectrogram_size: Target size for spectrogram creation
        """
        self.base_dataset = base_dataset
        self.spectrogram_size = spectrogram_size

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get 1D signal and label from base dataset
        signal, label = self.base_dataset[idx]

        # Ensure signal is a tensor
        if not isinstance(signal, torch.Tensor):
            signal = torch.tensor(signal, dtype=torch.float32)

        # Convert label to tensor if needed
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)

        return signal, label


class DummyDataset(Dataset):
    """
    Minimal dummy dataset for testing when main repository datasets are not available
    """

    def __init__(self, num_samples=1000, seq_len=4096, num_classes=10):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_classes = num_classes

        # Generate synthetic signals with different patterns for each class
        self.signals = []
        self.labels = []

        for i in range(num_samples):
            label = i % num_classes

            # Create different signal patterns for different classes
            if label % 3 == 0:  # Sinusoidal with noise
                t = np.linspace(0, 1, seq_len)
                signal = np.sin(2 * np.pi * (label + 1) * t) + 0.1 * np.random.randn(seq_len)
            elif label % 3 == 1:  # Chirp signal
                t = np.linspace(0, 1, seq_len)
                signal = np.sin(2 * np.pi * (label + 1) * t * (1 + t)) + 0.1 * np.random.randn(seq_len)
            else:  # Impulse + noise
                signal = np.random.randn(seq_len) * 0.1
                impulse_pos = (label * seq_len) // num_classes
                impulse_width = min(50, seq_len - impulse_pos)
                signal[impulse_pos:impulse_pos + impulse_width] += np.random.randn(impulse_width)

            self.signals.append(torch.tensor(signal, dtype=torch.float32))
            self.labels.append(torch.tensor(label, dtype=torch.long))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


class PHMVibenchWindowDataset(Dataset):
    """Small deterministic HDF5 window dataset for the paper-local demo."""

    def __init__(
        self,
        data_dir,
        dataset_task,
        seq_len=4096,
        num_classes=10,
        max_records=20,
        windows_per_record=2,
    ):
        self.data_dir = Path(data_dir)
        self.dataset_task = dataset_task
        self.dataset_id, file_stem = _dataset_info_from_task(dataset_task)
        self.seq_len = int(seq_len)
        self.num_classes = int(num_classes)
        self.max_records = int(max_records)
        self.windows_per_record = int(windows_per_record)
        self.h5_path = self.data_dir / f"{file_stem}.h5"
        self.metadata_path = self.data_dir / "metadata.xlsx"
        self.samples = self._build_index()

    def _build_index(self):
        if not self.h5_path.exists():
            raise FileNotFoundError(f"missing HDF5 file: {self.h5_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"missing metadata file: {self.metadata_path}")

        metadata = pd.read_excel(self.metadata_path)
        metadata = metadata[metadata["Dataset_id"] == self.dataset_id].copy()
        metadata = metadata[metadata["Label"].notna()]
        metadata["Id"] = metadata["Id"].astype(str)
        metadata["Label"] = metadata["Label"].astype(int)

        with h5py.File(self.h5_path, "r") as h5_file:
            available = set(h5_file.keys())
            metadata = metadata[metadata["Id"].isin(available)]
            labels = sorted(metadata["Label"].unique().tolist())[: self.num_classes]
            label_map = {label: idx for idx, label in enumerate(labels)}
            metadata = metadata[metadata["Label"].isin(label_map)]
            metadata = metadata.head(self.max_records)

            samples = []
            for row in metadata.itertuples(index=False):
                key = str(row.Id)
                label = label_map[int(row.Label)]
                length = int(h5_file[key].shape[0])
                if length < self.seq_len:
                    continue
                max_start = length - self.seq_len
                starts = (
                    [0]
                    if self.windows_per_record <= 1
                    else np.linspace(0, max_start, self.windows_per_record, dtype=int).tolist()
                )
                for start in starts:
                    samples.append((key, int(start), label))

        if not samples:
            raise ValueError(
                f"no valid windows for dataset_task={self.dataset_task}, "
                f"seq_len={self.seq_len}, num_classes={self.num_classes}"
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, start, label = self.samples[idx]
        with h5py.File(self.h5_path, "r") as h5_file:
            signal = np.asarray(h5_file[key][start : start + self.seq_len], dtype=np.float32)

        signal = np.squeeze(signal)
        if signal.ndim == 2:
            signal = signal[:, 0]
        elif signal.ndim > 2:
            signal = signal.reshape(signal.shape[0], -1)[:, 0]

        signal = signal.astype(np.float32, copy=False)
        std = float(signal.std())
        if std > 0:
            signal = (signal - float(signal.mean())) / std

        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def _dataset_info_from_task(dataset_task):
    normalized = str(dataset_task).upper()
    for marker, dataset_info in PHM_DATASET_BY_TASK.items():
        if marker in normalized:
            return dataset_info
    raise ValueError(f"unsupported dataset_task for PHM-Vibench demo: {dataset_task}")


def get_1d2d_dataloaders(config):
    """
    Get dataloaders for 1D-2D fusion training

    Args:
        config: Configuration dictionary containing:
            - data_dir: Path to data directory
            - dataset_task: Dataset task name
            - batch_size: Batch size for dataloaders
            - num_workers: Number of worker processes
            - pin_memory: Whether to pin memory

    Returns:
        train_loader, val_loader, test_loader: PyTorch DataLoaders
    """
    # Extract config values with defaults
    data_dir = config.get('data_dir', '/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/data/THU_018')
    dataset_task = config.get('dataset_task', 'THU_018_basic')
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)
    pin_memory = config.get('pin_memory', True)
    train_ratio = config.get('train_ratio', 0.7)
    val_ratio = config.get('val_ratio', 0.2)
    use_dummy = config.get('use_dummy', False)
    input_dim = config.get('input_dim', 4096)
    num_classes = config.get('num_classes', 10)
    max_records = config.get('max_records', 20)
    windows_per_record = config.get('windows_per_record', 2)

    if use_dummy:
        print("Using dummy dataset for testing...")
        full_dataset = DummyDataset(num_samples=1000, seq_len=input_dim, num_classes=num_classes)
    else:
        try:
            full_dataset = PHMVibenchWindowDataset(
                data_dir=data_dir,
                dataset_task=dataset_task,
                seq_len=input_dim,
                num_classes=num_classes,
                max_records=max_records,
                windows_per_record=windows_per_record,
            )
            print(
                "Loaded PHM-Vibench HDF5 windows: "
                f"{len(full_dataset)} samples from {dataset_task}"
            )
        except Exception as e:
            print(f"Could not load PHM-Vibench HDF5 windows: {e}")
            print("Using dummy dataset for testing...")
            full_dataset = DummyDataset(num_samples=1000, seq_len=input_dim, num_classes=num_classes)

    # Split into train/val/test
    total_size = len(full_dataset)
    train_size = max(1, int(train_ratio * total_size))
    val_size = max(1, int(val_ratio * total_size))
    test_size = total_size - train_size - val_size
    if test_size <= 0:
        test_size = 1
        train_size = max(1, train_size - 1)

    generator = torch.Generator().manual_seed(config.get('seed', 0))
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Wrap datasets with 1D-2D functionality
    train_1d2d = OneD2DDataset(train_dataset)
    val_1d2d = OneD2DDataset(val_dataset)
    test_1d2d = OneD2DDataset(test_dataset)

    # Create dataloaders
    train_loader = DataLoader(
        train_1d2d,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_1d2d,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_1d2d,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader


def test_dataloaders():
    """Test function for dataloaders"""
    # Simple config for testing
    test_config = {
        'data_dir': '/tmp/test_data',  # This will trigger dummy dataset
        'dataset_task': 'THU_018_basic',
        'batch_size': 16,
        'num_workers': 0,  # Set to 0 for testing
        'pin_memory': False
    }

    try:
        train_loader, val_loader, test_loader = get_1d2d_dataloaders(test_config)

        print(f"Train loader batches: {len(train_loader)}")
        print(f"Val loader batches: {len(val_loader)}")
        print(f"Test loader batches: {len(test_loader)}")

        # Test one batch
        for signals, labels in train_loader:
            print(f"Batch signals shape: {signals.shape}")
            print(f"Batch labels shape: {labels.shape}")
            print(f"Signal dtype: {signals.dtype}")
            print(f"Label dtype: {labels.dtype}")
            print(f"Label range: {labels.min().item()} - {labels.max().item()}")
            break

        print("Dataloader test: PASSED")
        return True

    except Exception as e:
        print(f"Dataloader test: FAILED - {e}")
        return False


if __name__ == "__main__":
    test_dataloaders()
