"""Target-only metadata/reader path owned by the existing ExplicitDataFactory."""
from __future__ import annotations

from torch.utils.data import DataLoader

from .data_utils import MetadataAccessor
from .dataset_task.Default_dataset import Default_dataset
from .dataset_task.Dataset_cluster import IdIncludedDataset
from .ID.Id_searcher import search_target_dataset_metadata
from .samplers.Sampler import Same_system_Sampler


def initialize_target_stream(factory, args_data, args_task) -> None:
    """Select declared target domains before reading raw signals or creating caches.

    Domain order is explicit replay order, not inferred acquisition chronology.
    Only selected target files enter the normal reader/cache/windowing machinery.
    The source checkpoint's class vocabulary is supplied to Model Factory separately.
    """
    if args_data.num_workers != 0:
        raise ValueError("The maintained target stream requires data.num_workers=0")
    if getattr(args_data, "split", None) is not None:
        raise ValueError("TTA stream selection uses target_domain_id/target_asset_ids, not data.split")
    order = list(args_task.target_domain_id)
    if not order or len(set(order)) != len(order):
        raise ValueError("task.target_domain_id must list each target domain once in replay order")
    factory.args_data, factory.args_task = args_data, args_task
    selected = search_target_dataset_metadata(factory._init_metadata(args_data), args_task)
    frame = selected.df[selected.df["Domain_id"].isin(order)].copy().reset_index(drop=True)
    assets = getattr(args_task, "target_asset_ids", None)
    if assets is not None:
        frame = frame[frame["asset_id"].isin(assets)].copy()
    if set(frame["Domain_id"]) != set(order):
        raise ValueError("Every requested target domain must have selected records")
    if frame["Dataset_id"].nunique() != 1:
        raise ValueError("This target stream requires one declared class namespace")
    rank = {domain: i for i, domain in enumerate(order)}
    frame["_order"] = frame["Domain_id"].map(rank)
    frame = frame.sort_values(["_order", "Id"], kind="stable").drop(columns="_order")
    factory.metadata = MetadataAccessor(frame.reset_index(drop=True), key_column="Id")
    # _init_data uses the existing strict reader, HDF5 cache and system selector.
    # All other records have already been excluded; no source signal is accessed.
    factory.data = factory._init_data(args_data)
    metadata = factory.get_metadata()
    per_file = {
        file_id: Default_dataset({file_id: factory.data[file_id]}, metadata,
                                 args_data, args_task, "test")
        for file_id in metadata.keys()
    }
    factory.test_dataset = IdIncludedDataset(per_file, metadata)
    sampler = Same_system_Sampler(factory.test_dataset, batch_size=args_data.batch_size,
                                  shuffle=False, drop_last=False,
                                  system_metadata_key="Domain_id")
    factory.test_loader = DataLoader(factory.test_dataset, batch_sampler=sampler,
                                    num_workers=0)
