from __future__ import annotations

from typing import Any

from .data_factory import data_factory, register_data_factory
from .dataset_task.ID_dataset import set_dataset
from .dataset_task.Dataset_cluster import IdIncludedDataset


@register_data_factory("id")
class IDDataFactory(data_factory):
    """Data factory using :class:`ID_dataset` for on-demand processing."""

    def _init_dataset(self):
        train_dataset = {}
        val_dataset = {}
        test_dataset = {}
        train_val_ids, test_ids = self.search_id()
        for fid in train_val_ids:
            train_dataset[fid] = set_dataset({fid: self.data[fid]}, self.target_metadata, self.args_data, self.args_task, "train")
            val_dataset[fid] = set_dataset({fid: self.data[fid]}, self.target_metadata, self.args_data, self.args_task, "val")
        for fid in test_ids:
            test_dataset[fid] = set_dataset({fid: self.data[fid]}, self.target_metadata, self.args_data, self.args_task, "test")
        train_dataset = IdIncludedDataset(train_dataset, self.target_metadata)
        val_dataset = IdIncludedDataset(val_dataset, self.target_metadata)
        test_dataset = IdIncludedDataset(test_dataset, self.target_metadata)
        return train_dataset, val_dataset, test_dataset
