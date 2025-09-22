"""
ID Dataset Module for PHM-Vibench Framework

This module provides ID-based dataset classes that handle metadata and ID management
without loading actual data, following the factory pattern for deferred data processing.
"""

from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional
import logging
from ..data_utils import  MetadataAccessor


class ID_dataset(Dataset):
    """
    Dataset class that manages data IDs and metadata without loading actual data.

    This class follows the PHM-Vibench factory pattern where data loading is deferred
    to the task processing stage, enabling more flexible and memory-efficient workflows.

    Attributes:
        metadata (Dict[str, Dict[str, Any]]): Metadata dictionary with ID as key
        ids (List[str]): List of available data IDs
        args_data: Data processing configuration parameters
        args_task: Task-specific configuration parameters
        mode (str): Dataset mode ('train', 'val', 'test')
    """

    def __init__(
        self,
        metadata: Dict[str, Dict[str, Any]],
        args_data: Any,
        args_task: Any,
        mode: str = "train"
    ):
        """
        Initialize ID dataset with metadata and configuration.

        Args:
            metadata: Dictionary mapping ID to metadata information
                Format: {ID: {field: value}} where each ID maps to a metadata dict
            args_data: Data processing parameters (Namespace or dict-like object)
            args_task: Task parameters (Namespace or dict-like object)
            mode: Dataset mode, one of ['train', 'val', 'test']

        Raises:
            ValueError: If metadata is empty or mode is invalid
            TypeError: If metadata is not a dictionary
        """
        if not isinstance(metadata, (dict, MetadataAccessor)):
            raise TypeError(f"metadata must be a dictionary, got {type(metadata)}")

        if not metadata:
            raise ValueError("metadata cannot be empty")

        valid_modes = {'train', 'val', 'test', 'valid'}  # 'valid' for backward compatibility
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")

        self.metadata = metadata
        self.ids = list(self.metadata.keys())
        self.args_data = args_data
        self.args_task = args_task
        self.mode = mode


    def __len__(self) -> int:
        """Return the number of available IDs."""
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get ID and metadata for the specified index.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing:
                - 'id': The data ID (str)
                - 'metadata': Associated metadata dictionary

        Raises:
            IndexError: If index is out of range
        """
        if not 0 <= idx < len(self.ids):
            raise IndexError(f"Index {idx} out of range [0, {len(self.ids)})")

        sample_id = self.ids[idx]
        return {
            "id": sample_id,
            "metadata": self.metadata[sample_id]
        }

    def get_ids(self) -> List[str]:
        """Return a copy of the ID list."""
        return self.ids.copy()

    def get_metadata_for_id(self, sample_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific ID.

        Args:
            sample_id: The ID to look up

        Returns:
            Metadata dictionary for the specified ID

        Raises:
            KeyError: If ID is not found
        """
        if sample_id not in self.metadata:
            raise KeyError(f"ID '{sample_id}' not found in metadata")
        return self.metadata[sample_id]

    def filter_by_criteria(self, criteria: Dict[str, Any]) -> 'ID_dataset':
        """
        Create a new dataset filtered by metadata criteria.

        Args:
            criteria: Dictionary of field-value pairs to filter by

        Returns:
            New ID_dataset instance with filtered metadata
        """
        filtered_metadata = {}
        for sample_id, meta in self.metadata.items():
            if all(meta.get(key) == value for key, value in criteria.items()):
                filtered_metadata[sample_id] = meta

        return ID_dataset(filtered_metadata, self.args_data, self.args_task, self.mode)


class set_dataset(ID_dataset):
    """
    Alias class used by data_factory for dynamic ID datasets.

    This class maintains backward compatibility with the existing factory pattern
    while providing the enhanced functionality of the refactored ID_dataset.

    Note: The 'data' parameter is accepted for compatibility but not used,
    as this class focuses on ID and metadata management only.
    """

    def __init__(
        self,
        data: Optional[Dict[str, Any]],
        metadata: Dict[str, Dict[str, Any]],
        args_data: Any,
        args_task: Any,
        mode: str = "train"
    ):
        """
        Initialize set_dataset with backward compatibility.

        Args:
            data: Data dictionary (accepted for compatibility, not used)
            metadata: Metadata dictionary mapping ID to metadata
            args_data: Data processing parameters
            args_task: Task parameters
            mode: Dataset mode
        """

        super().__init__(metadata, args_data, args_task, mode)


# TODO: Implement balanced ID sampling functionality
def balance_ids_by_label(
    metadata: Dict[str, Dict[str, Any]],
    label_field: str = 'Label'
) -> Dict[str, List[str]]:
    """
    Group IDs by label for balanced sampling.

    Args:
        metadata: Metadata dictionary
        label_field: Field name containing the label information

    Returns:
        Dictionary mapping label to list of IDs with that label
    """
    label_to_ids = {}
    for sample_id, meta in metadata.items():
        label = meta.get(label_field)
        if label is not None:
            if label not in label_to_ids:
                label_to_ids[label] = []
            label_to_ids[label].append(sample_id)
    return label_to_ids


if __name__ == '__main__':
    """Test module functionality."""
    # Create sample metadata for testing
    test_metadata = {
        'sample_001': {'Label': 0, 'Domain_id': 1, 'Sample_rate': 1000},
        'sample_002': {'Label': 1, 'Domain_id': 1, 'Sample_rate': 1000},
        'sample_003': {'Label': 0, 'Domain_id': 2, 'Sample_rate': 2000},
    }

    # Mock args objects
    class MockArgs:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    args_data = MockArgs(batch_size=32, window_size=1024)
    args_task = MockArgs(task_type='classification')

    # Test ID_dataset
    print("Testing ID_dataset...")
    dataset = ID_dataset(test_metadata, args_data, args_task, mode='train')
    print(f"Dataset length: {len(dataset)}")
    print(f"Sample 0: {dataset[0]}")
    print(f"IDs: {dataset.get_ids()}")

    # Test filtering
    filtered = dataset.filter_by_criteria({'Domain_id': 1})
    print(f"Filtered dataset length: {len(filtered)}")

    # Test balanced ID grouping
    balanced = balance_ids_by_label(test_metadata)
    print(f"Balanced groups: {balanced}")

    print("All tests passed!")

