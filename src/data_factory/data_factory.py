"""
数据读取模块
负责读取和处理元数据及原始数据文件
"""
import os
import importlib
import glob
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
from ..utils.registry import Registry

DATA_FACTORY_REGISTRY = Registry()

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
        os.makedirs(args_data.data_dir, exist_ok=True)
        for id_key in tqdm(task_meta.keys(), desc="检查 Name.h5 缓存", disable=not list(task_meta.keys())):
            try:
                meta = self.metadata[id_key]
            except KeyError:
                continue
            name = meta.get('Name')
            if not name:
                continue
            name_cache_file = os.path.join(args_data.data_dir, f"{name}.h5")
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
        name_cache_file = os.path.join(args_data.data_dir, f"{name}.h5")
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
        final_cache_path = os.path.join(args_data.data_dir, "cache.h5")
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
                    name_cache_file = os.path.join(args_data.data_dir, f"{name}.h5")
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
        try:
            mod = importlib.import_module(
                f"src.data_factory.dataset_task.{task_type}.{task_name}_dataset"
            )
            dataset_cls = mod.set_dataset
        except ImportError as e:
            print("Using ID_dataset for on-demand processing.")
            from .dataset_task.ID_dataset import set_dataset as dataset_cls
        train_dataset = {}
        val_dataset = {}
        test_dataset = {}
        train_val_ids, test_ids = self.search_id()
        # Initialize datasets with progress bars
        print("Initializing training and validation datasets...")
        for id in tqdm(train_val_ids, desc="Creating train/val datasets"):
            train_dataset[id] = dataset_cls({id: self.data[id]},
                             self.target_metadata, self.args_data, self.args_task, 'train')
            val_dataset[id] = dataset_cls({id: self.data[id]},
                               self.target_metadata, self.args_data, self.args_task, 'val')

        print("Initializing test datasets...")
        for id in tqdm(test_ids, desc="Creating test datasets"):
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

        persistent_workers = self.args_data.num_workers > 0
        self.train_loader = DataLoader(self.train_dataset,
                                #   batch_size=self.args_data.batch_size,
                                         batch_sampler = train_sampler,
                                        #  shuffle=True,
                                         num_workers=self.args_data.num_workers,
                                         pin_memory=True,     
                                         persistent_workers=persistent_workers,)
                                        #  collate_fn=debug_collate_fn)
        self.val_loader = DataLoader(self.val_dataset,
                                #  batch_size=self.args_data.batch_size,
                                        batch_sampler = val_sampler,
                                        # shuffle=False,
                                        num_workers=self.args_data.num_workers,
                                        pin_memory=True,     
                                        persistent_workers=persistent_workers,)
        self.test_loader = DataLoader(self.test_dataset,
                                #  batch_size=self.args_data.batch_size,
                                        batch_sampler = test_sampler,
                                        # shuffle=False,
                                        num_workers=self.args_data.num_workers,
                                        pin_memory=True,     
                                        persistent_workers=persistent_workers,)



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
