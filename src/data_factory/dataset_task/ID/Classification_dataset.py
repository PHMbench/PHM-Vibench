"""Dataset implementation for ID classification tasks."""

from ..Default_dataset import Default_dataset


class set_dataset(Default_dataset):
    """Dataset class for ID tasks."""

    def __init__(self, data, metadata, args_data, args_task, mode="train"):
        """Initialize ID dataset using the default dataset implementation."""
        super().__init__(data, metadata, args_data, args_task, mode)
