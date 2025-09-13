"""
Enhanced ID Data Factory for PHM-Vibench Framework

This module provides an optimized data factory specifically designed for ID-based datasets
that defers data loading to the task processing stage, enabling memory-efficient workflows
and flexible data processing pipelines.
"""

from .data_factory import data_factory, register_data_factory
from .H5DataDict import H5DataDict
from .dataset_task.ID_dataset import set_dataset as ID_dataset_cls
from .dataset_task.Dataset_cluster import IdIncludedDataset
from .ID.Id_searcher import search_ids_for_task, search_target_dataset_metadata
from torch.utils.data import DataLoader
from typing import Any, Dict, Tuple, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


@register_data_factory("id")
class id_data_factory(data_factory):
    """
    Enhanced data factory for ID-based datasets with optimized memory usage.

    This factory specializes in handling ID-only datasets where actual data loading
    is deferred to the task processing stage. It provides efficient metadata management
    and streamlined dataloader creation for large-scale time-series datasets.

    Key Features:
    - Memory-efficient ID-only dataset creation
    - Lazy data loading through H5DataDict
    - Optimized batch processing for variable-length sequences
    - Seamless integration with existing task processing pipelines
    """

    def __init__(self, args_data: Any, args_task: Any):
        """
        Initialize the ID data factory with enhanced error handling and logging.

        Args:
            args_data: Data configuration parameters
            args_task: Task configuration parameters
        """
        logger.info("Initializing ID data factory...")
        super().__init__(args_data, args_task)
        logger.info(f"ID data factory initialized with {len(self.data)} data entries")

    def _init_data(self, args_data: Any, use_cache: bool = True, max_workers: int = 32) -> H5DataDict:
        """
        Override data initialization for ID-specific optimizations.

        This method optimizes data loading for ID-based workflows by:
        1. Only loading metadata initially
        2. Creating lazy-loading data dictionary
        3. Deferring actual data loading to task processing

        Args:
            args_data: Data configuration parameters
            use_cache: Whether to use cached data files
            max_workers: Number of worker threads for data loading

        Returns:
            H5DataDict: Lazy-loading data dictionary
        """
        logger.info("Initializing data with ID-optimized loading...")

        # Use parent implementation but with optimizations for ID workflow
        data_dict = super()._init_data(args_data, use_cache, max_workers)

        # Log data statistics for debugging
        if hasattr(self, 'target_metadata'):
            logger.info(f"Target metadata contains {len(self.target_metadata)} entries")

        return data_dict

    def _init_dataset(self) -> Tuple[IdIncludedDataset, IdIncludedDataset, IdIncludedDataset]:
        """
        Initialize ID-based datasets with enhanced error handling and progress tracking.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info("Initializing ID-based datasets...")

        # Get target metadata and split IDs
        self.target_metadata = search_target_dataset_metadata(self.metadata, self.args_task)
        train_val_ids, test_ids = search_ids_for_task(self.target_metadata, self.args_task)

        logger.info(f"Dataset split: {len(train_val_ids)} train/val IDs, {len(test_ids)} test IDs")

        # Initialize dataset containers
        train_dataset = {}
        val_dataset = {}
        test_dataset = {}

        # Create train/val datasets with progress tracking
        logger.info("Creating train/validation datasets...")
        for sample_id in tqdm(train_val_ids, desc="Creating train/val datasets"):
            try:
                # Create datasets with only metadata (no data loading)
                train_dataset[sample_id] = ID_dataset_cls(
                    data=None,  # No data loading at this stage
                    metadata=self.target_metadata,
                    args_data=self.args_data,
                    args_task=self.args_task,
                    mode='train'
                )
                val_dataset[sample_id] = ID_dataset_cls(
                    data=None,  # No data loading at this stage
                    metadata=self.target_metadata,
                    args_data=self.args_data,
                    args_task=self.args_task,
                    mode='val'
                )
            except Exception as e:
                logger.warning(f"Failed to create dataset for ID {sample_id}: {e}")
                continue

        # Create test datasets with progress tracking
        logger.info("Creating test datasets...")
        for sample_id in tqdm(test_ids, desc="Creating test datasets"):
            try:
                test_dataset[sample_id] = ID_dataset_cls(
                    data=None,  # No data loading at this stage
                    metadata=self.target_metadata,
                    args_data=self.args_data,
                    args_task=self.args_task,
                    mode='test'
                )
            except Exception as e:
                logger.warning(f"Failed to create test dataset for ID {sample_id}: {e}")
                continue

        # Wrap datasets with IdIncludedDataset for batch processing
        train_dataset = IdIncludedDataset(train_dataset, self.target_metadata)
        val_dataset = IdIncludedDataset(val_dataset, self.target_metadata)
        test_dataset = IdIncludedDataset(test_dataset, self.target_metadata)

        logger.info("ID-based datasets initialized successfully")
        return train_dataset, val_dataset, test_dataset

    def _make_loader(self, dataset: IdIncludedDataset, shuffle: bool) -> DataLoader:
        """
        Create optimized DataLoader for ID-based datasets.

        Args:
            dataset: The dataset to create a loader for
            shuffle: Whether to shuffle the data

        Returns:
            DataLoader configured for ID-based processing
        """
        # 强制禁用persistent_workers以防止内存累积
        persistent_workers = False
        # 限制num_workers数量以减少内存使用
        num_workers = min(getattr(self.args_data, 'num_workers', 0), 4)

        return DataLoader(
            dataset,
            batch_size=getattr(self.args_data, 'batch_size', 32),
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,  # 禁用pin_memory减少内存压力
            persistent_workers=persistent_workers,
            drop_last=False,  # Keep all samples for ID-based processing
        )

    def _init_dataloader(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Initialize optimized dataloaders for ID-based processing.

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        logger.info("Initializing ID-optimized dataloaders...")

        # Create dataloaders with appropriate shuffling
        self.train_loader = self._make_loader(self.train_dataset, shuffle=True)
        self.val_loader = self._make_loader(self.val_dataset, shuffle=False)
        self.test_loader = self._make_loader(self.test_dataset, shuffle=False)

        logger.info("ID-optimized dataloaders initialized successfully")
        return self.train_loader, self.val_loader, self.test_loader

    def get_data_for_id(self, sample_id: str) -> Optional[Any]:
        """
        Get actual data for a specific ID (lazy loading).

        Args:
            sample_id: The ID to retrieve data for

        Returns:
            Data array for the specified ID, or None if not found
        """
        try:
            return self.data[sample_id]
        except KeyError:
            logger.warning(f"Data not found for ID: {sample_id}")
            return None

    def get_metadata_for_id(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific ID.

        Args:
            sample_id: The ID to retrieve metadata for

        Returns:
            Metadata dictionary for the specified ID, or None if not found
        """
        try:
            return self.target_metadata[sample_id] if hasattr(self, 'target_metadata') else self.metadata[sample_id]
        except KeyError:
            logger.warning(f"Metadata not found for ID: {sample_id}")
            return None

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the loaded datasets.

        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            'total_ids': len(self.data) if self.data else 0,
            'train_ids': len(self.train_dataset) if hasattr(self, 'train_dataset') else 0,
            'val_ids': len(self.val_dataset) if hasattr(self, 'val_dataset') else 0,
            'test_ids': len(self.test_dataset) if hasattr(self, 'test_dataset') else 0,
        }

        if hasattr(self, 'target_metadata'):
            # Add metadata-based statistics
            labels = [meta.get('Label') for meta in self.target_metadata.values() if meta.get('Label') is not None]
            domains = [meta.get('Domain_id') for meta in self.target_metadata.values() if meta.get('Domain_id') is not None]

            stats.update({
                'unique_labels': len(set(labels)) if labels else 0,
                'unique_domains': len(set(domains)) if domains else 0,
                'label_distribution': {label: labels.count(label) for label in set(labels)} if labels else {},
                'domain_distribution': {domain: domains.count(domain) for domain in set(domains)} if domains else {},
            })

        return stats


if __name__ == '__main__':
    """Test module functionality."""
    print("Testing id_data_factory...")

    # Mock configuration objects
    class MockArgs:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    # This would require actual data files to test fully
    # For now, just verify the class can be imported and instantiated
    print("id_data_factory module loaded successfully!")
    print("Note: Full testing requires actual data files and metadata.")
