"""
数据读取模块
负责读取和处理元数据及原始数据文件
"""
import os
import importlib
import glob
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import h5py
from .H5DataDict import H5DataDict
from .dataset_task.Dataset_cluster import IdIncludedDataset # ,Balanced_DataLoader_Dict_Iterator # TODO del balanced_data_loader
from .dataset_task.adapters import resolve_dataset_adapter
from torch.utils.data import DataLoader
import copy
import concurrent.futures
from tqdm import tqdm  # 用于显示进度条
from torch.utils.data import Dataset
from .samplers.Sampler import GroupedIdBatchSampler, BalancedIdSampler
from .data_utils import smart_read_csv, MetadataAccessor, download_data
from .samplers.Get_sampler import Get_sampler
from .ID.Id_searcher import search_ids_for_task, search_target_dataset_metadata
from .splitting import resolve_data_splits
from ..utils.registry import Registry

DATA_FACTORY_REGISTRY = Registry()


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
        # parameters    
        self.args_data = args_data
        self.args_task = args_task
        # metadata and data cache
        self.metadata = self._init_metadata(args_data)
        self.data = self._init_data(args_data)
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
        """Read all requested IDs and atomically update one dataset cache."""
        if not ids:
            return

        id_meta_pairs = []
        missing_metadata = []
        for file_id in ids:
            meta = self.metadata[file_id]
            if not meta.get("File"):
                missing_metadata.append(str(file_id))
                continue
            id_meta_pairs.append((file_id, meta))

        if missing_metadata:
            raise RuntimeError(
                f"Cannot build cache for dataset {name!r}: metadata is missing "
                f"File for ID(s) {', '.join(missing_metadata)}. Fix the metadata "
                "before rerunning."
            )

        results = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = [
                executor.submit(
                    self._read_single_data,
                    file_id,
                    meta,
                    args_data,
                )
                for file_id, meta in id_meta_pairs
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"并行读取 {name}",
            ):
                results.append(future.result())

        failures = [
            (str(file_id), error or "reader returned no data")
            for file_id, data, error in results
            if data is None
        ]
        if failures:
            details = "; ".join(
                f"ID {file_id}: {reason}" for file_id, reason in failures
            )
            raise RuntimeError(
                f"Cannot publish cache for dataset {name!r}; raw-data reading "
                f"failed. {details}"
            )

        cache_path = Path(_cache_directory(args_data)) / f"{name}.h5"
        temp_path = cache_path.with_name(f".{cache_path.name}.tmp")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.unlink(missing_ok=True)

        try:
            if cache_path.is_file():
                shutil.copy2(cache_path, temp_path)
            with h5py.File(temp_path, "a") as h5_file:
                for file_id, data, _ in results:
                    key = str(file_id)
                    if key in h5_file:
                        del h5_file[key]
                    h5_file.create_dataset(key, data=data)

                missing_ids = [
                    str(file_id)
                    for file_id in ids
                    if str(file_id) not in h5_file
                ]
                if missing_ids:
                    raise RuntimeError(
                        f"Temporary cache {temp_path} is missing ID(s) "
                        f"{', '.join(missing_ids)}."
                    )

            os.replace(temp_path, cache_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def _build_final_cache(self, task_meta, args_data, use_cache):
        """Reuse a complete task cache or rebuild it before atomic publication."""
        expected_ids = list(task_meta.keys())
        if not expected_ids:
            raise ValueError(
                "The selected task contains no data IDs. Check task.target_system_id, "
                "domain selection, labels, and metadata."
            )

        expected_keys = {str(file_id) for file_id in expected_ids}
        cache_directory = Path(_cache_directory(args_data))
        cache_path = cache_directory / "cache.h5"
        temp_path = cache_directory / ".cache.h5.tmp"
        cache_directory.mkdir(parents=True, exist_ok=True)

        if use_cache and cache_path.is_file():
            try:
                with h5py.File(cache_path, "r") as published_cache:
                    if expected_keys.issubset(published_cache.keys()):
                        return str(cache_path)
            except OSError as exc:
                raise RuntimeError(
                    f"Existing cache cannot be opened: {cache_path}. Delete this "
                    "cache and rerun so PHMFactory can rebuild it."
                ) from exc

        temp_path.unlink(missing_ok=True)
        missing = []
        try:
            with h5py.File(temp_path, "w") as output_cache:
                for file_id in tqdm(expected_ids, desc="整合 cache.h5"):
                    meta = self.metadata[file_id]
                    dataset_name = meta.get("Name")
                    if not dataset_name:
                        missing.append(
                            (str(file_id), "metadata field Name is missing")
                        )
                        continue

                    source_path = cache_directory / f"{dataset_name}.h5"
                    if not source_path.is_file():
                        missing.append(
                            (str(file_id), f"dataset cache not found: {source_path}")
                        )
                        continue

                    key = str(file_id)
                    with h5py.File(source_path, "r") as source_cache:
                        if key not in source_cache:
                            missing.append(
                                (
                                    key,
                                    f"ID is absent from dataset cache {source_path}",
                                )
                            )
                            continue
                        source_cache.copy(key, output_cache, name=key)

            if missing:
                details = "; ".join(
                    f"ID {file_id}: {reason}" for file_id, reason in missing
                )
                raise RuntimeError(
                    "Cannot publish cache.h5 because the selected data is "
                    f"incomplete. {details}"
                )

            os.replace(temp_path, cache_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

        return str(cache_path)

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
        dataset_cls = resolve_dataset_adapter(task_type, task_name)
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
