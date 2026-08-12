"""
数据读取模块
负责读取和处理元数据及原始数据文件
"""
import os
import hashlib
import importlib
import glob
import json
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
from .H5DataDict import H5DataDict
from .dataset_task.Dataset_cluster import IdIncludedDataset # ,Balanced_DataLoader_Dict_Iterator # TODO del balanced_data_loader
from torch.utils.data import DataLoader
import copy
import concurrent.futures
from tqdm import tqdm  # 用于显示进度条
from torch.utils.data import Dataset
from .samplers.Sampler import GroupedIdBatchSampler, BalancedIdSampler
from .data_utils import smart_read_csv, MetadataAccessor, download_data
from .samplers.Get_sampler import Get_sampler
from .ID.Id_searcher import search_ids_for_task, search_target_dataset_metadata
from .splitting import SplitResult, resolve_data_splits
from .grouped_split import build_grouped_split, write_frozen_json
from ..utils.registry import Registry

DATA_FACTORY_REGISTRY = Registry()


class SplitContractError(RuntimeError):
    """Raised before I/O when a configured split cannot be enforced safely."""


class DatasetResolutionError(RuntimeError):
    """Raised when the configured dataset implementation cannot be resolved."""


def _config_value(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def validate_split_preflight(args_data):
    """Reject declared split contracts that cannot be consumed before data I/O."""

    split_config = _config_value(args_data, "split")
    if split_config is None:
        return
    strategy = _config_value(split_config, "strategy")
    if not isinstance(strategy, str) or not strategy.strip():
        raise SplitContractError("data.split.strategy is required")
    if strategy == "legacy_windows":
        return
    if strategy not in {
        "grouped_metadata",
        "grouped_kfold",
        "preassigned_metadata",
    }:
        raise SplitContractError(
            f"Unsupported data.split.strategy {strategy!r}; refusing to ignore it"
        )
    group_key = _config_value(split_config, "group_key")
    if not isinstance(group_key, str) or not group_key.strip():
        raise SplitContractError(f"data.split.group_key is required for {strategy}")
    manifest_path = _config_value(split_config, "manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path.strip():
        raise SplitContractError(
            f"data.split.manifest_path is required for {strategy}"
        )
    if strategy == "grouped_metadata" and _config_value(
        split_config, "fractions"
    ) is None:
        raise SplitContractError(
            "grouped_metadata without explicit split fractions is not implemented; "
            "refusing to load data with an inert split contract"
        )
    if strategy == "preassigned_metadata":
        split_key = _config_value(split_config, "split_key")
        if not isinstance(split_key, str) or not split_key.strip():
            raise SplitContractError(
                "data.split.split_key is required for preassigned_metadata"
            )


def resolve_dataset_class(task_type, task_name):
    """Resolve exactly one configured dataset module without silent fallback."""

    if not isinstance(task_type, str) or not task_type.isidentifier():
        raise DatasetResolutionError(f"Invalid task type {task_type!r}")
    if not isinstance(task_name, str) or not task_name.isidentifier():
        raise DatasetResolutionError(f"Invalid task name {task_name!r}")
    if task_type == "Default_task":
        module_name = (
            f"src.data_factory.dataset_task.{task_type}.{task_name}_dataset"
        )
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            from .dataset_task.Default_dataset import Default_dataset

            return Default_dataset
        dataset_class = getattr(module, "set_dataset", None)
        if dataset_class is None:
            raise DatasetResolutionError(
                f"Configured dataset module {module_name} has no set_dataset"
            )
        return dataset_class

    task_root = Path(__file__).resolve().parent / "dataset_task"
    task_dirs = sorted(
        path
        for path in task_root.iterdir()
        if path.is_dir() and path.name.casefold() == task_type.casefold()
    )
    if len(task_dirs) != 1:
        raise DatasetResolutionError(
            f"Expected exactly one dataset task directory for {task_type!r}, "
            f"found {[path.name for path in task_dirs]}"
        )
    expected_stem = f"{task_name}_dataset".casefold()
    module_files = sorted(
        path
        for path in task_dirs[0].glob("*.py")
        if path.stem.casefold() == expected_stem
    )
    if len(module_files) != 1:
        raise DatasetResolutionError(
            f"Expected exactly one dataset module for {task_type}.{task_name}, "
            f"found {[path.name for path in module_files]}"
        )
    module_name = (
        f"{__package__}.dataset_task.{task_dirs[0].name}.{module_files[0].stem}"
    )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise DatasetResolutionError(
            f"Failed to import configured dataset module {module_name}"
        ) from exc
    dataset_class = getattr(module, "set_dataset", None)
    if dataset_class is None:
        raise DatasetResolutionError(
            f"Configured dataset module {module_name} has no set_dataset"
        )
    return dataset_class


def _cache_directory(args_data):
    """Return the writable cache root without changing raw-data lookup paths.

    ``data_dir`` remains the source of metadata and raw samples.  Evidence
    datasets may additionally set ``cache_dir`` so generated HDF5 files never
    mutate an immutable, hash-ledgered data snapshot.  Existing configurations
    retain their historical behavior when ``cache_dir`` is absent or empty.
    """

    configured = getattr(args_data, "cache_dir", None)
    return os.fspath(configured) if configured else os.fspath(args_data.data_dir)

def register_data_factory(name: str):
    """Decorator to register a data factory implementation."""
    return DATA_FACTORY_REGISTRY.register(name)




class data_factory:
    """数据集工厂类，负责读取和处理数据集
    原始数据 -> 根据task构建数据 -> 为trainer 提供迭代器
    data -> dataset -> dataloader -> balanced dataloader
    """
    def __init__(self, args_data,args_task):
        """初始化数据集工厂
        
        Args:
            args_data: 包含data_dir和metadata_file的字典或命名空间
        """
        # A declared split must be enforceable before any metadata download,
        # cache creation, or raw-data access can occur.
        validate_split_preflight(args_data)

        # parameters    
        self.args_data = args_data
        self.args_task = args_task
        # metadata and data cache
        self.metadata = self._init_metadata(args_data)
        self.data = self._init_data(args_data)
        self._data_fingerprint_records = {}
        # dataset and dataloader
        self.train_dataset, self.val_dataset,self.test_dataset = self._init_dataset()
        self.train_loader, self.val_loader, self.test_loader = self._init_dataloader()

    def _init_metadata(self, args_data):
        """
        初始化元数据
        
        Args:
            args_data: 包含data_dir和metadata_file的字典或命名空间
            
        Returns:
            MetadataAccessor: 元数据访问器对象
        """
        # 1. 检查并自动下载元数据文件（如果不存在）
        try:
             download_data(data_file=args_data.metadata_file,
                                           save_path=args_data.data_dir,
                                             source='auto')
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            raise
        
        # 2. 读取元数据
        try:
            metadata_path = os.path.join(args_data.data_dir, args_data.metadata_file)
            meta_df = smart_read_csv(metadata_path, auto_detect=True)
            metadata = MetadataAccessor(meta_df, key_column='Id')
            print(f"[SUCCESS] 成功加载元数据，共 {len(metadata)} 条记录")
            return metadata
        except Exception as e:
            print(f"[ERROR] 读取元数据文件失败: {e}")
            raise
    
    def _read_single_data(self, id_key, meta, args_data):
        """Read one raw file and return an array.

        Parameters
        ----------
        id_key : str
            Example ``"1001"``.
        meta : dict
            Metadata row with at least ``{"Name": str, "File": str}``.
        args_data : Namespace
            Should provide ``data_dir`` and ``metadata_file``.

        Returns
        -------
        Tuple[str, np.ndarray | None, str | None]
            On success ``(id_key, array, None)`` otherwise
            ``(id_key, None, error_message)``.
        """
        try:
            name = meta['Name']
            file_name = meta['File']
            download_data(data_file=args_data.metadata_file, save_path=args_data.data_dir, source='auto')
            mod = importlib.import_module(f"src.data_factory.reader.{name}")
            file_path = os.path.join(args_data.data_dir, f"raw/{name}/{file_name}")
            if not os.path.exists(file_path):
                # Smoke/demo robustness: allow synthetic readers (e.g. Dummy_Data) to generate data.
                if name != "Dummy_Data":
                    return id_key, None, f"原始数据文件未找到: {file_path}"
            data = mod.read(file_path, args_data)
            if data.ndim == 2:
                data = np.expand_dims(data, axis=-1)
            return id_key, data, None
        except Exception as e:
            return id_key, None, str(e)

    def _determine_missing_ids(self, task_meta, args_data, use_cache):
        """Determine which IDs are absent from cache.

        Parameters
        ----------
        task_meta : MetadataAccessor
            Mapping of IDs needed for the current task.
        args_data : Namespace
            Contains ``data_dir``.
        use_cache : bool
            Whether to reuse existing ``Name.h5`` files.

        Returns
        -------
        Dict[str, List[str]]
            Keys are dataset names. Each value is a list of ID keys
            to be fetched from raw files.
        """
        ids_to_fetch = {}
        cache_directory = _cache_directory(args_data)
        os.makedirs(cache_directory, exist_ok=True)
        for id_key in tqdm(task_meta.keys(), desc="检查 Name.h5 缓存", disable=not list(task_meta.keys())):
            try:
                meta = self.metadata[id_key]
            except KeyError:
                continue
            name = meta.get('Name')
            if not name:
                continue
            name_cache_file = os.path.join(cache_directory, f"{name}.h5")
            h5_key = str(id_key)
            need = False
            if not use_cache or not os.path.exists(name_cache_file):
                need = True
            else:
                with h5py.File(name_cache_file, 'r') as h5f:
                    if h5_key not in h5f:
                        need = True
            if need:
                ids_to_fetch.setdefault(name, []).append(id_key)
        return ids_to_fetch

    def _update_name_cache(self, name, ids, args_data, max_workers):
        """Read raw files for one dataset name and update its cache.

        Parameters
        ----------
        name : str
            Dataset name such as ``"CWRU"``.
        ids : List[str]
            ID keys that belong to this dataset.
        args_data : Namespace
            Supplies ``data_dir`` and ``metadata_file``.
        max_workers : int
            Thread pool size for reading files.
        """
        if not ids:
            return
        name_cache_file = os.path.join(
            _cache_directory(args_data), f"{name}.h5"
        )
        id_meta_pairs = []
        for id_k in ids:
            meta = self.metadata[id_k]
            if 'File' not in meta:
                continue
            id_meta_pairs.append((id_k, meta))
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._read_single_data, id_k, meta, args_data) for id_k, meta in id_meta_pairs]
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"并行读取 {name}"):
                results.append(fut.result())
        os.makedirs(os.path.dirname(name_cache_file), exist_ok=True)
        with h5py.File(name_cache_file, 'a') as h5f:
            for id_res, data_res, _ in results:
                if data_res is None:
                    continue
                key = str(id_res)
                if key in h5f:
                    del h5f[key]
                h5f.create_dataset(key, data=data_res)

    def _build_final_cache(self, task_meta, args_data, use_cache):
        """Combine all ``Name.h5`` files into ``cache.h5``.

        Parameters
        ----------
        task_meta : MetadataAccessor
            Metadata for IDs used in this run.
        args_data : Namespace
            Provides ``data_dir`` where caches reside.
        use_cache : bool
            If ``False`` rebuild all entries regardless of existing cache.

        Returns
        -------
        str
            Path to the consolidated ``cache.h5`` file.
        """
        cache_directory = _cache_directory(args_data)
        final_cache_path = os.path.join(cache_directory, "cache.h5")
        os.makedirs(os.path.dirname(final_cache_path), exist_ok=True)
        missing_keys = []
        if use_cache and os.path.exists(final_cache_path):
            with h5py.File(final_cache_path, 'r') as h5f:
                for id_key in task_meta.keys():
                    if str(id_key) not in h5f:
                        missing_keys.append(id_key)
        else:
            missing_keys = list(task_meta.keys())
        if missing_keys:
            with h5py.File(final_cache_path, 'a') as h5f_consolidated:
                for id_key in tqdm(missing_keys, desc="整合 cache.h5"):
                    meta = self.metadata[id_key]
                    name = meta['Name']
                    name_cache_file = os.path.join(cache_directory, f"{name}.h5")
                    if not os.path.exists(name_cache_file):
                        continue
                    with h5py.File(name_cache_file, 'r') as h5f_name:
                        if str(id_key) in h5f_name:
                            data_arr = h5f_name[str(id_key)][()]
                            h5f_consolidated.create_dataset(str(id_key), data=data_arr)
        return final_cache_path

    def _init_data(self, args_data, use_cache=True, max_workers=32):
        """Prepare cache files and return a :class:`H5DataDict`.

        Parameters
        ----------
        args_data : Namespace
            Data configuration with ``data_dir`` and ``metadata_file``.
        use_cache : bool, optional
            If ``False`` force rebuilding all caches.
        max_workers : int, optional
            Number of worker threads for reading raw data.

        Returns
        -------
        H5DataDict
            Dictionary-like access to ``cache.h5``.
        """
        task_meta = self.search_dataset_id()
        if bool(getattr(args_data, "read_only_cache_required", False)):
            cache_path = Path(str(args_data.data_dir)) / "cache.h5"
            if not cache_path.is_file():
                raise FileNotFoundError(
                    f"Read-only evidence cache is missing: {cache_path}"
                )
            with h5py.File(cache_path, "r", libver="latest", swmr=True) as handle:
                available = set(handle.keys())
                missing = sorted(
                    str(identifier)
                    for identifier in task_meta.keys()
                    if str(identifier) not in available
                )
            if missing:
                raise RuntimeError(
                    "Read-only evidence cache is incomplete; refusing mutation: "
                    f"missing_count={len(missing)}, first_missing={missing[:20]}"
                )
            return H5DataDict(str(cache_path))
        ids_to_fetch = self._determine_missing_ids(task_meta, args_data, use_cache)
        for name, ids in ids_to_fetch.items():
            self._update_name_cache(name, ids, args_data, max_workers)
        cache_path = self._build_final_cache(task_meta, args_data, use_cache)
        print(f"数据整合完成。最终缓存文件: {cache_path}")
        return H5DataDict(cache_path)
    
    def get_metadata(self):
        """获取元数据"""
        return self.target_metadata if hasattr(self, 'target_metadata') else self.metadata
    def get_data(self):
        """获取数据"""
        return self.data
    
    def get_data_info(self):
        """获取数据集信息"""

        for id, data in self.data.items():
            print(f"##### ID: {id} #####")
        # TODO

    def _init_dataset(self):
        task_name = self.args_task.name
        task_type = self.args_task.type
        dataset_cls = resolve_dataset_class(task_type, task_name)
        split_cfg = getattr(self.args_data, "split", None)
        if split_cfg is not None and getattr(split_cfg, "strategy", None) in {
            "grouped_metadata",
            "grouped_kfold",
        }:
            return self._init_p01_grouped_dataset(dataset_cls, split_cfg)
        train_dataset = {}
        val_dataset = {}
        test_dataset = {}
        train_val_ids, task_test_ids = self.search_id()
        self.split_result = resolve_data_splits(
            self.target_metadata,
            self.args_data,
            self.args_task,
            train_val_ids,
            task_test_ids,
        )
        print("Initializing training datasets...")
        for id in tqdm(self.split_result.train_ids, desc="Creating train datasets"):
            train_dataset[id] = dataset_cls({id: self.data[id]},
                             self.target_metadata, self.args_data, self.args_task, 'train')
        print("Initializing validation datasets...")
        for id in tqdm(self.split_result.val_ids, desc="Creating val datasets"):
            val_dataset[id] = dataset_cls({id: self.data[id]},
                               self.target_metadata, self.args_data, self.args_task, 'val')

        print("Initializing test datasets...")
        for id in tqdm(self.split_result.test_ids, desc="Creating test datasets"):
            test_dataset[id] = dataset_cls({id: self.data[id]},
                            self.target_metadata, self.args_data, self.args_task, 'test')
        train_dataset = IdIncludedDataset(train_dataset,self.target_metadata)
        val_dataset = IdIncludedDataset(val_dataset,self.target_metadata)
        test_dataset = IdIncludedDataset(test_dataset,self.target_metadata)
        return train_dataset, val_dataset, test_dataset

    def _init_p01_grouped_dataset(self, dataset_cls, split_cfg):
        if getattr(split_cfg, "test_policy", "partition") != "partition":
            raise ValueError(
                "grouped splitting in the in-domain data factory requires "
                "test_policy=partition"
            )
        split = build_grouped_split(self.target_metadata, split_cfg)
        self.split_result = SplitResult(
            train_ids=tuple(split.train_ids),
            val_ids=tuple(split.val_ids),
            test_ids=tuple(split.test_ids),
            strategy=str(getattr(split_cfg, "strategy", "grouped_metadata")),
            manifest_path=str(getattr(split_cfg, "manifest_path", "")) or None,
        )
        expected_sha = getattr(
            split_cfg, "expected_manifest_payload_sha256", None
        )
        if getattr(split_cfg, "strategy", None) == "grouped_kfold" and not expected_sha:
            raise ValueError(
                "grouped_kfold requires data.split.expected_manifest_payload_sha256"
            )
        observed_sha = split.manifest["manifest_payload_sha256"]
        if expected_sha and observed_sha != str(expected_sha):
            raise RuntimeError(
                "Grouped split payload hash does not match the approved protocol: "
                f"observed={observed_sha}, expected={expected_sha}"
            )
        manifest_path = getattr(split_cfg, "manifest_path", None)
        if not manifest_path:
            raise ValueError("grouped splitting requires data.split.manifest_path")

        self.split_manifest = split.manifest
        write_frozen_json(split.manifest, manifest_path)
        self.train_val_ids = list(split.train_ids) + list(split.val_ids)
        self.test_ids = list(split.test_ids)

        def make_partition(identifiers, mode):
            partition = {}
            for identifier in identifiers:
                value = self.data[identifier]
                self._record_data_fingerprint(identifier, value)
                partition[identifier] = dataset_cls(
                    {identifier: value},
                    self.target_metadata,
                    self.args_data,
                    self.args_task,
                    mode,
                )
            return IdIncludedDataset(partition, self.target_metadata)

        train_dataset = make_partition(split.train_ids, "test")
        val_dataset = make_partition(split.val_ids, "test")
        test_dataset = make_partition(split.test_ids, "test")

        pairing_cfg = getattr(self.args_data, "pairing", None)
        pairing_mode = (
            getattr(pairing_cfg, "mode", "paired")
            if pairing_cfg is not None
            else "paired"
        )
        if pairing_mode == "paired":
            return train_dataset, val_dataset, test_dataset
        if pairing_mode != "frozen_within_group_class_derangement":
            raise ValueError(f"Unknown data.pairing.mode: {pairing_mode}")
        if pairing_cfg is None:
            raise ValueError("Frozen P01 pairing requires data.pairing")
        required = ("seed", "splits", "group_key", "manifest_dir", "protocol_id")
        missing = [name for name in required if not hasattr(pairing_cfg, name)]
        if missing:
            raise ValueError(
                "frozen_within_group_class_derangement requires explicit fields: "
                + ", ".join(missing)
            )
        if str(pairing_cfg.group_key) != str(split_cfg.group_key):
            raise ValueError(
                "data.pairing.group_key must equal data.split.group_key"
            )
        if isinstance(pairing_cfg.splits, str):
            raise ValueError("data.pairing.splits must be a non-empty list")
        pairing_splits = list(pairing_cfg.splits)
        if not pairing_splits or len(set(pairing_splits)) != len(pairing_splits):
            raise ValueError(
                "data.pairing.splits must be non-empty and duplicate-free"
            )
        unknown = sorted(set(pairing_splits) - {"train", "val", "test"})
        if unknown:
            raise ValueError(f"Unknown data.pairing.splits entries: {unknown}")

        from .dataset_task.Dataset_cluster import FrozenClassPairDataset

        datasets = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        for split_name in pairing_splits:
            datasets[split_name] = FrozenClassPairDataset(
                datasets[split_name],
                seed=int(pairing_cfg.seed),
                split_name=split_name,
                manifest_dir=str(pairing_cfg.manifest_dir),
                group_key=str(pairing_cfg.group_key),
                protocol_id=str(pairing_cfg.protocol_id),
                split_manifest_sha256=observed_sha,
            )
        write_frozen_json(
            {
                "schema_version": 1,
                "protocol_id": str(pairing_cfg.protocol_id),
                "mode": pairing_mode,
                "seed": int(pairing_cfg.seed),
                "active_splits": pairing_splits,
                "group_key": str(pairing_cfg.group_key),
                "split_manifest_sha256": observed_sha,
            },
            Path(str(pairing_cfg.manifest_dir)) / "index.json",
        )
        return datasets["train"], datasets["val"], datasets["test"]

    def _record_data_fingerprint(self, identifier, value):
        records = getattr(self, "_data_fingerprint_records", None)
        if records is None:
            records = {}
            self._data_fingerprint_records = records
        elif not isinstance(records, dict):
            raise TypeError("data fingerprint record collection must be a dictionary")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError("Evidence data fingerprints reject object-dtype arrays")
        contiguous = np.ascontiguousarray(array)
        key = str(identifier)
        record = {
            "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "nbytes": int(contiguous.nbytes),
        }
        existing = records.get(key)
        if existing is not None and existing != record:
            raise RuntimeError(f"Data content drift detected within run for ID {key}")
        records[key] = record

    def get_data_fingerprint(self):
        expected_ids = {str(identifier) for identifier in self.target_metadata.keys()}
        observed_ids = set(self._data_fingerprint_records)
        if observed_ids != expected_ids:
            raise RuntimeError(
                "Data fingerprint coverage mismatch: "
                f"missing={sorted(expected_ids - observed_ids)}, "
                f"unexpected={sorted(observed_ids - expected_ids)}"
            )
        records = {
            key: self._data_fingerprint_records[key]
            for key in sorted(self._data_fingerprint_records)
        }
        canonical = json.dumps(
            records,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        cache_path = Path(str(self.data.h5_file))
        cache_stat = cache_path.stat()
        return {
            "algorithm": "sha256_over_c_contiguous_array_bytes",
            "eligible_ids": len(records),
            "records": records,
            "data_payload_sha256": hashlib.sha256(canonical).hexdigest(),
            "cache_path": str(cache_path.resolve()),
            "cache_size_bytes": int(cache_stat.st_size),
            "cache_mtime_ns": int(cache_stat.st_mtime_ns),
        }


    def search_dataset_id(self):
        self.target_metadata = search_target_dataset_metadata(self.metadata, self.args_task)
        return self.target_metadata
    
    def search_id(self):
        self.train_val_ids, self.test_ids = search_ids_for_task(self.target_metadata, self.args_task)
        return self.train_val_ids, self.test_ids
        

    def get_sampler(self, mode='train'):
        if mode == 'train':
            dataset = self.train_dataset
        elif mode == 'val':
            dataset = self.val_dataset
        elif mode == 'test':
            dataset = self.test_dataset
        else:
            raise ValueError(f"Unknown mode for get_sampler: {mode}")
        return Get_sampler(self.args_task, self.args_data, dataset, mode)

    def _init_dataloader(self):
        train_sampler = self.get_sampler(mode='train')
        val_sampler = self.get_sampler(mode='val')
        test_sampler = self.get_sampler(mode='test')

        self.train_loader = DataLoader(self.train_dataset,
                                #   batch_size=self.args_data.batch_size,
                                         batch_sampler = train_sampler,
                                        #  shuffle=True,
                                         num_workers=self.args_data.num_workers,)
                                        #  collate_fn=debug_collate_fn)
        self.val_loader = DataLoader(self.val_dataset,
                                #  batch_size=self.args_data.batch_size,
                                        batch_sampler = val_sampler,
                                        # shuffle=False,
                                        num_workers=self.args_data.num_workers,)
        self.test_loader = DataLoader(self.test_dataset,
                                #  batch_size=self.args_data.batch_size,
                                        batch_sampler = test_sampler,
                                        # shuffle=False,
                                        num_workers=self.args_data.num_workers,)



        return self.train_loader, self.val_loader, self.test_loader

    def get_dataset(self, mode = "test"):
        """获取指定ID的数据集
        
        Args:
            id: 数据集ID
        
        Returns:
            数据集
        """
        return self.train_dataset if mode == "train" else self.val_dataset if mode == "val" else self.test_dataset
    def get_dataloader(self, mode = "test"):
        """获取指定ID的数据加载器
        
        Args:
            id: 数据集ID
            batch_size: 批大小
        
        Returns:
            数据加载器
        """
        return self.train_loader if mode == "train" else self.val_loader if mode == "val" else self.test_loader

    def __len__(self):
        """返回数据集数量"""
        return len(self.data)
    



class department_data_factory(data_factory):
    """
    TODO : 处理子集的情况
    """
    def __init__(self, args_data, args_task):
        super().__init__(args_data, args_task)


# Register default factories
register_data_factory("default")(data_factory)
register_data_factory("department")(department_data_factory)
