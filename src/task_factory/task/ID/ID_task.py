"""
Enhanced ID Task Module for PHM-Vibench Framework

This module provides a comprehensive and extensible base class for ID-based tasks
that process time-series data with flexible windowing, batching, and preprocessing
capabilities. It serves as the foundation for various machine learning workflows
including pre-training, classification, and anomaly detection.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
import logging

from ...Default_task import Default_task
from ...utils.data_processing import process_sample
from ... import register_task

logger = logging.getLogger(__name__)


class BaseIDTask(Default_task, ABC):
    """
    Abstract base class for ID-based tasks with extensible architecture.

    This class provides a comprehensive framework for processing time-series data
    from ID datasets with configurable windowing, preprocessing, and batching
    strategies. It's designed to be extended for specific task requirements.

    Key Features:
    - Flexible windowing strategies (sequential, random, evenly_spaced)
    - Configurable preprocessing pipelines
    - Extensible batch preparation for variable-length sequences
    - Memory-efficient processing for large datasets
    - Support for multi-channel time-series data

    Architecture:
    - create_windows(): Transform (l, c) → (w, window_l, c)
    - process_sample(): Individual sample preprocessing
    - prepare_batch(): Extensible batching with uniform output (b, w, window_l, c)
    """

    def __init__(
        self,
        network: torch.nn.Module,
        args_data: Any,
        args_model: Any,
        args_task: Any,
        args_trainer: Any,
        args_environment: Any,
        metadata: Any
    ):
        """
        Initialize the ID task with enhanced configuration validation.

        Args:
            network: Neural network model
            args_data: Data processing configuration
            args_model: Model configuration
            args_task: Task-specific configuration
            args_trainer: Training configuration
            args_environment: Environment configuration
            metadata: Dataset metadata
        """
        super().__init__(network, args_data, args_model, args_task, args_trainer, args_environment, metadata)

        # Validate required data processing parameters
        self._validate_data_config()

        # Initialize processing statistics
        self.processing_stats = {
            'total_samples_processed': 0,
            'total_windows_created': 0,
            'failed_samples': 0,
            'average_windows_per_sample': 0.0
        }

        logger.info(f"Initialized {self.__class__.__name__} with windowing strategy: "
                   f"{getattr(args_data, 'window_sampling_strategy', 'evenly_spaced')}")

    def _validate_data_config(self) -> None:
        """Validate required data processing configuration parameters."""
        required_params = ['window_size', 'stride', 'num_window']
        missing_params = []

        for param in required_params:
            if not hasattr(self.args_data, param):
                missing_params.append(param)

        if missing_params:
            raise ValueError(f"Missing required data processing parameters: {missing_params}")

        # Validate parameter values
        if self.args_data.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.args_data.window_size}")
        if self.args_data.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.args_data.stride}")
        if self.args_data.num_window <= 0:
            raise ValueError(f"num_window must be positive, got {self.args_data.num_window}")

    def create_windows(
        self,
        data: np.ndarray,
        window_size: Optional[int] = None,
        stride: Optional[int] = None,
        num_window: Optional[int] = None,
        strategy: Optional[str] = None
    ) -> List[np.ndarray]:
        """
        Transform single sample from (l, c) to windowed format (w, window_l, c).

        This method provides flexible windowing strategies for time-series data,
        handling variable sequence lengths gracefully with appropriate padding
        or truncation strategies.

        Args:
            data: Input time-series data with shape (l, c) where:
                  l = sequence length, c = number of channels
            window_size: Length of each window (overrides args_data.window_size)
            stride: Step size between windows (overrides args_data.stride)
            num_window: Number of windows to create (overrides args_data.num_window)
            strategy: Windowing strategy (overrides args_data.window_sampling_strategy)
                     Options: 'sequential', 'random', 'evenly_spaced'

        Returns:
            List of windows, each with shape (window_l, c)
            Empty list if input is shorter than window size

        Raises:
            ValueError: If data dimensions are invalid
        """
        if data.ndim < 1 or data.ndim > 2:
            raise ValueError(f"Data must be 1D or 2D, got shape {data.shape}")

        # Ensure data is 2D (l, c)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Use provided parameters or fall back to configuration
        win_size = window_size or self.args_data.window_size
        step = stride or self.args_data.stride
        num_win = num_window or self.args_data.num_window
        strat = strategy or getattr(self.args_data, 'window_sampling_strategy', 'evenly_spaced')

        data_length = len(data)
        windows: List[np.ndarray] = []

        # Handle case where data is shorter than window size
        if data_length < win_size:
            logger.debug(f"Data length {data_length} < window size {win_size}, returning empty list")
            return windows

        # Apply windowing strategy
        if strat == 'sequential':
            windows = self._create_sequential_windows(data, win_size, step, num_win)
        elif strat == 'random':
            windows = self._create_random_windows(data, win_size, num_win)
        elif strat == 'evenly_spaced':
            windows = self._create_evenly_spaced_windows(data, win_size, num_win)
        else:
            raise ValueError(f"Unknown windowing strategy: {strat}")

        # Update statistics
        self.processing_stats['total_windows_created'] += len(windows)

        return windows

    def _create_sequential_windows(
        self,
        data: np.ndarray,
        win_size: int,
        stride: int,
        num_window: int
    ) -> List[np.ndarray]:
        """Create sequential windows with specified stride."""
        windows = []
        data_length = len(data)
        max_windows = max(0, (data_length - win_size) // stride + 1)
        actual_num = min(max_windows, num_window)

        for i in range(actual_num):
            start = i * stride
            windows.append(data[start:start + win_size].copy())

        return windows

    def _create_random_windows(
        self,
        data: np.ndarray,
        win_size: int,
        num_window: int
    ) -> List[np.ndarray]:
        """Create randomly positioned windows."""
        windows = []
        data_length = len(data)
        possible_starts = np.arange(data_length - win_size + 1)

        if len(possible_starts) <= num_window:
            # Use all possible positions
            selected_starts = possible_starts
        else:
            # Randomly sample positions without replacement
            selected_starts = np.random.choice(
                possible_starts, size=num_window, replace=False
            )

        for start in selected_starts:
            windows.append(data[start:start + win_size].copy())

        return windows

    def _create_evenly_spaced_windows(
        self,
        data: np.ndarray,
        win_size: int,
        num_window: int
    ) -> List[np.ndarray]:
        """Create evenly spaced windows across the sequence."""
        windows = []
        data_length = len(data)

        if num_window <= 1:
            # Single window at center
            start = max(0, (data_length - win_size) // 2)
            windows.append(data[start:start + win_size].copy())
        else:
            # Multiple evenly spaced windows
            effective_length = data_length - win_size
            step = 0 if num_window == 1 else effective_length / (num_window - 1)

            for i in range(num_window):
                start = int(round(i * step))
                start = min(start, data_length - win_size)
                windows.append(data[start:start + win_size].copy())

        return windows

    def process_sample(
        self,
        data: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Handle individual sample processing pipeline.

        This method applies normalization, filtering, and other preprocessing
        steps while maintaining data integrity and handling edge cases.

        Args:
            data: Raw time-series data with shape (l, c)
            metadata: Optional metadata for sample-specific processing

        Returns:
            Processed sample with consistent format and shape (l, c)

        Raises:
            ValueError: If data format is invalid
        """
        if data.size == 0:
            raise ValueError("Cannot process empty data array")

        # Ensure data is numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Use existing process_sample utility with enhancements
        processed_data = process_sample(data, self.args_data)

        # Apply any additional task-specific processing
        processed_data = self._apply_task_specific_processing(processed_data, metadata)

        # Update statistics
        self.processing_stats['total_samples_processed'] += 1

        return processed_data

    def _apply_task_specific_processing(
        self,
        data: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None  # pylint: disable=unused-argument
    ) -> np.ndarray:
        """
        Apply task-specific preprocessing steps.

        This method can be overridden by derived classes to implement
        custom preprocessing logic based on task requirements.

        Args:
            data: Preprocessed data array
            metadata: Sample metadata for context-aware processing (unused in base implementation)

        Returns:
            Data array with task-specific processing applied
        """
        # Base implementation - no additional processing
        # Override in derived classes for custom behavior
        # metadata parameter is available for derived classes but unused here
        return data

    @abstractmethod
    def prepare_batch(
        self,
        batch_data: List[Tuple[str, np.ndarray, Dict[str, Any]]]
    ) -> Dict[str, torch.Tensor]:
        """
        **Key extensible method for task-specific batching.**

        This method handles variable window counts across different IDs and
        provides uniform batch tensors. It's designed as the main extension
        point for derived classes to implement task-specific batching strategies.

        Args:
            batch_data: List of (id, data_array, metadata) tuples where:
                       - id: Sample identifier (str)
                       - data_array: Raw time-series with shape (l, c)
                       - metadata: Sample metadata dictionary

        Returns:
            Dictionary containing uniform batch tensors:
                - 'x': Tensor with shape (b, w, window_l, c) where:
                       b = batch size
                       w = number of windows (uniform across batch)
                       window_l = window length
                       c = number of channels
                - 'y': Target tensor with appropriate shape for task
                - 'file_id': List of sample IDs
                - Additional task-specific tensors as needed

        Note:
            This method must handle variable window counts (w) across different
            IDs through padding, truncation, or sampling strategies to ensure
            uniform batch dimensions.
        """
        pass

    def _shared_step(
        self,
        batch: Dict[str, Any],
        stage: str,
        task_id: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Override to preprocess raw batches on-the-fly.

        This method intercepts raw batches from ID_dataset and applies
        the complete processing pipeline before passing to the parent
        implementation.

        Args:
            batch: Raw batch from dataloader
            stage: Training stage ('train', 'val', 'test')
            task_id: Whether to include task ID in processing

        Returns:
            Dictionary containing loss and metrics
        """
        # Check if batch needs preprocessing
        if 'x' not in batch:
            # Extract raw data and apply processing pipeline
            batch = self._preprocess_raw_batch(batch)

        return super()._shared_step(batch, stage, task_id)

    def _preprocess_raw_batch(self, raw_batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Preprocess raw batch from ID_dataset using the processing pipeline.

        Args:
            raw_batch: Raw batch containing 'id' and 'metadata' lists

        Returns:
            Processed batch ready for model input
        """
        # Extract batch components
        ids = raw_batch.get('id', [])
        metadata_list = raw_batch.get('metadata', [])

        # Load actual data for each ID
        batch_data = []
        for sample_id, metadata in zip(ids, metadata_list):
            try:
                # Get data from the data factory (lazy loading)
                data_array = self._get_data_for_id(sample_id)
                if data_array is not None:
                    batch_data.append((sample_id, data_array, metadata))
                else:
                    logger.warning(f"No data found for ID: {sample_id}")
            except Exception as e:
                logger.error(f"Failed to load data for ID {sample_id}: {e}")
                self.processing_stats['failed_samples'] += 1
                continue

        # Apply the extensible batch preparation
        if batch_data:
            processed_batch = self.prepare_batch(batch_data)

            # Update processing statistics
            if self.processing_stats['total_samples_processed'] > 0:
                self.processing_stats['average_windows_per_sample'] = (
                    self.processing_stats['total_windows_created'] /
                    self.processing_stats['total_samples_processed']
                )

            return processed_batch
        else:
            # Return empty batch if no valid data
            return {
                'x': torch.empty(0),
                'y': torch.empty(0, dtype=torch.long),
                'file_id': []
            }

    def _get_data_for_id(self, sample_id: str) -> Optional[np.ndarray]:
        """
        Get actual data for a specific ID from the data factory.

        This method provides access to the lazy-loaded data through
        the data factory's H5DataDict interface.

        Args:
            sample_id: The ID to retrieve data for

        Returns:
            Data array for the specified ID, or None if not found
        """
        try:
            # Access data through trainer's data factory
            if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'data'):
                return self.trainer.datamodule.data[sample_id]
            else:
                logger.warning("Cannot access data factory from trainer")
                return None
        except (KeyError, AttributeError) as e:
            logger.debug(f"Data not found for ID {sample_id}: {e}")
            return None

    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive processing statistics.

        Returns:
            Dictionary containing processing statistics and performance metrics
        """
        return self.processing_stats.copy()

    def reset_processing_statistics(self) -> None:
        """Reset processing statistics counters."""
        self.processing_stats = {
            'total_samples_processed': 0,
            'total_windows_created': 0,
            'failed_samples': 0,
            'average_windows_per_sample': 0.0
        }


@register_task("Default_task", "ID_task")
class task(BaseIDTask):
    """
    Concrete implementation of ID task for general-purpose time-series processing.

    This class provides a default implementation of the abstract BaseIDTask,
    suitable for most common use cases including classification, regression,
    and pre-training tasks.
    """

    def prepare_batch(
        self,
        batch_data: List[Tuple[str, np.ndarray, Dict[str, Any]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Default implementation of batch preparation for ID-based processing.

        This implementation handles variable window counts by taking the first
        window from each sample, providing a simple but effective batching
        strategy for most tasks.

        Args:
            batch_data: List of (id, data_array, metadata) tuples

        Returns:
            Dictionary containing:
                - 'x': Tensor with shape (b, window_l, c)
                - 'y': Target tensor with shape (b,)
                - 'file_id': List of sample IDs
        """
        xs, ys, fids = [], [], []

        for sample_id, data_array, metadata in batch_data:
            try:
                # Process the sample
                processed_data = self.process_sample(data_array, metadata)

                # Create windows
                windows = self.create_windows(processed_data)

                if not windows:
                    logger.debug(f"No windows created for sample {sample_id}")
                    continue

                # Take the first window (can be extended for multiple windows)
                window = windows[0]

                # Convert to tensor and add to batch
                xs.append(torch.tensor(window, dtype=torch.float32))
                ys.append(metadata.get('Label', 0))  # Default label if not found
                fids.append(sample_id)

            except Exception as e:
                logger.error(f"Failed to process sample {sample_id}: {e}")
                self.processing_stats['failed_samples'] += 1
                continue

        if not xs:
            # Return empty batch if no valid samples
            return {
                'x': torch.empty(0, self.args_data.window_size, 1),
                'y': torch.empty(0, dtype=torch.long),
                'file_id': []
            }

        return {
            'x': torch.stack(xs),
            'y': torch.tensor(ys, dtype=torch.long),
            'file_id': fids,
        }


class MultiWindowIDTask(BaseIDTask):
    """
    Extended ID task that handles multiple windows per sample.

    This class demonstrates how to extend BaseIDTask for more complex
    batching strategies that preserve multiple windows per sample.
    """

    def prepare_batch(
        self,
        batch_data: List[Tuple[str, np.ndarray, Dict[str, Any]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Batch preparation that handles multiple windows per sample.

        This implementation creates uniform batches by padding or truncating
        to ensure all samples have the same number of windows.

        Args:
            batch_data: List of (id, data_array, metadata) tuples

        Returns:
            Dictionary containing:
                - 'x': Tensor with shape (b, w, window_l, c)
                - 'y': Target tensor with shape (b,)
                - 'file_id': List of sample IDs
                - 'window_mask': Boolean mask indicating valid windows
        """
        all_windows, ys, fids, window_counts = [], [], [], []

        for sample_id, data_array, metadata in batch_data:
            try:
                # Process the sample
                processed_data = self.process_sample(data_array, metadata)

                # Create windows
                windows = self.create_windows(processed_data)

                if not windows:
                    logger.debug(f"No windows created for sample {sample_id}")
                    continue

                # Convert windows to tensors
                window_tensors = [torch.tensor(w, dtype=torch.float32) for w in windows]

                all_windows.append(window_tensors)
                window_counts.append(len(window_tensors))
                ys.append(metadata.get('Label', 0))
                fids.append(sample_id)

            except Exception as e:
                logger.error(f"Failed to process sample {sample_id}: {e}")
                self.processing_stats['failed_samples'] += 1
                continue

        if not all_windows:
            # Return empty batch
            return {
                'x': torch.empty(0, self.args_data.num_window, self.args_data.window_size, 1),
                'y': torch.empty(0, dtype=torch.long),
                'file_id': [],
                'window_mask': torch.empty(0, self.args_data.num_window, dtype=torch.bool)
            }

        # Determine target number of windows (use configured num_window)
        target_windows = self.args_data.num_window
        batch_size = len(all_windows)

        # Get window and channel dimensions from first valid sample
        window_length = all_windows[0][0].shape[0]
        num_channels = all_windows[0][0].shape[1] if all_windows[0][0].ndim > 1 else 1

        # Initialize batch tensor and mask
        batch_tensor = torch.zeros(batch_size, target_windows, window_length, num_channels)
        window_mask = torch.zeros(batch_size, target_windows, dtype=torch.bool)

        # Fill batch tensor with padding/truncation
        for i, windows in enumerate(all_windows):
            actual_windows = min(len(windows), target_windows)

            for j in range(actual_windows):
                batch_tensor[i, j] = windows[j]
                window_mask[i, j] = True

        return {
            'x': batch_tensor,
            'y': torch.tensor(ys, dtype=torch.long),
            'file_id': fids,
            'window_mask': window_mask
        }


if __name__ == '__main__':
    """Test module functionality."""
    print("Testing ID_task module...")

    # Create sample data for testing
    sample_data = np.random.randn(1000, 2)  # 1000 time steps, 2 channels

    # Mock configuration
    class MockArgs:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    args_data = MockArgs(
        window_size=128,
        stride=64,
        num_window=5,
        window_sampling_strategy='evenly_spaced',
        normalization='standardization',
        dtype='float32'
    )

    # Test windowing functionality (without full task initialization)
    print("Testing windowing strategies...")

    # Test sequential windowing
    from ...utils.data_processing import create_windows
    windows_seq = create_windows(sample_data, args_data)
    print(f"Sequential windows: {len(windows_seq)} windows created")

    # Test different strategies
    args_data.window_sampling_strategy = 'random'
    windows_rand = create_windows(sample_data, args_data)
    print(f"Random windows: {len(windows_rand)} windows created")

    args_data.window_sampling_strategy = 'evenly_spaced'
    windows_even = create_windows(sample_data, args_data)
    print(f"Evenly spaced windows: {len(windows_even)} windows created")

    print("ID_task module tests completed successfully!")
    print("Note: Full task testing requires complete framework initialization.")

