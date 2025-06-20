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
    
    def _init_data(self, args_data, use_cache=True, max_workers=32):
        """
        数据读取器 - 并行版本。
        1. 为每个 'Name' (数据集类型) 维护一个独立的、累积的 HDF5 缓存文件 (e.g., NameA.h5, NameB.h5)。
           这些 Name.h5 文件会包含所有处理过的属于该 Name 的数据。
        2. 根据 task_relevant_metadata (通过 self.search_dataset_id() 获取) 确定当前任务需要的数据ID。
        3. 对于任务所需的每个ID，检查其数据是否存在于对应的 Name.h5 缓存中。
           如果数据缺失、Name.h5 文件不存在，或 use_cache=False，则从原始文件读取数据，
           并将其添加/更新到相应的 Name.h5 文件中。
        4. 创建一个名为 "cache.h5" 的新 HDF5 文件 (或覆盖已有的)。此文件将仅包含当前任务
           (由 task_relevant_metadata 定义的ID) 所需的数据，这些数据从各自更新后的 Name.h5 文件中提取。
           "cache.h5" 供后续处理步骤使用。
        
        Args:
            args_data: 包含data_dir和metadata_file的字典或命名空间
            use_cache: 是否使用HDF5缓存，默认为True
            max_workers: 并行工作进程数，默认为32
                
        Returns:
            H5DataDict: 指向最终为当前任务生成的 "cache.h5" 的数据字典对象
        """
        
        # 辅助函数，用于读取单个数据文件
        def read_single_data_op(id_meta_tuple):
            id_key, meta_op = id_meta_tuple 
            try:
                # 确保 'Name' 和 'File' 键存在于元数据中
                if 'Name' not in meta_op or 'File' not in meta_op:
                    return id_key, None, f"元数据中缺少 'Name' 或 'File' 键。"
                
                name_op = meta_op['Name']
                file_op = meta_op['File']

                # 确保数据存在
                download_data(data_file=args_data.metadata_file,
                                            save_path=args_data.data_dir,
                                                source='auto')                
                # 确保导入路径相对于项目结构是正确的
                mod = importlib.import_module(f"src.data_factory.reader.{name_op}")
                
                file_path = os.path.join(args_data.data_dir, f'raw/{name_op}/{file_op}')
                
                if not os.path.exists(file_path):
                    return id_key, None, f"原始数据文件未找到: {file_path}"

                data = mod.read(file_path, args_data)
                if data.ndim == 2:
                    data = np.expand_dims(data, axis=-1)  # B,L,C
                return id_key, data, None
            except ImportError as e_import:
                # 提供更具体的导入错误信息
                name_for_error = meta_op.get('Name', 'UnknownName') if isinstance(meta_op, (dict, pd.Series)) else 'UnknownName'
                return id_key, None, f"导入读取模块 src.data_factory.reader.{name_for_error} 失败: {e_import}"
            except Exception as e:
                return id_key, None, f"处理ID {id_key} (Name: {meta_op.get('Name', 'N/A') if isinstance(meta_op, (dict, pd.Series)) else 'N/A'}) 时发生错误: {e}"

        task_relevant_metadata = self.search_dataset_id() # MetadataAccessor for relevant IDs

        ids_to_fetch_from_raw_map = {}  # {'Name1': [id1_key, id2_key], 'Name2': [id3_key]}

        print("检查并准备各 Name.h5 缓存文件...")
        # 确保 args_data.data_dir 存在
        os.makedirs(args_data.data_dir, exist_ok=True)

        if not task_relevant_metadata.keys():
            print("没有与当前任务相关的目标数据ID。将创建一个空的 cache.h5。")
        
        for id_key in tqdm(task_relevant_metadata.keys(), desc="检查 Name.h5 缓存", disable=not list(task_relevant_metadata.keys())):
            try:
                h5_key = str(int(id_key))
            except ValueError: 
                h5_key = str(id_key)

            try:
                meta = self.metadata[id_key] # 从原始访问器获取完整元数据
                if not isinstance(meta, (pd.Series, dict)): 
                    print(f"警告: ID {id_key} 的元数据类型不正确 ({type(meta)})，跳过缓存检查。")
                    continue
                if 'Name' not in meta:
                    print(f"警告: ID {id_key} 的元数据中缺少 'Name' 键，跳过缓存检查。")
                    continue
            except KeyError:
                print(f"警告: 在 self.metadata (全局元数据) 中找不到ID {id_key} 的元数据，跳过此ID的缓存检查。")
                continue
                
            name = meta['Name']
            name_cache_file = os.path.join(args_data.data_dir, f"{name}.h5")

            needs_processing = False
            if not use_cache:
                needs_processing = True
            else:
                if not os.path.exists(name_cache_file):
                    needs_processing = True
                else:
                    try:
                        with h5py.File(name_cache_file, 'r') as h5f:
                            if h5_key not in h5f:
                                needs_processing = True
                    except Exception as e:
                        print(f"读取缓存 {name_cache_file} 检查ID {h5_key} 时出错，将重新处理: {e}")
                        needs_processing = True # 假定需要重新处理以保证安全

            if needs_processing:
                ids_to_fetch_from_raw_map.setdefault(name, []).append(id_key)
        
        # 处理需要从原始文件读取的数据，并更新对应的 Name.h5 文件
        for name, ids_list_to_read in ids_to_fetch_from_raw_map.items():
            if not ids_list_to_read:
                continue
            
            print(f"为数据集 '{name}' 从原始文件处理 {len(ids_list_to_read)} 个ID...")
            name_cache_file = os.path.join(args_data.data_dir, f"{name}.h5")
            
            id_meta_pairs = []
            for id_k_read in ids_list_to_read:
                try:
                    meta_for_id_read = self.metadata[id_k_read]
                    if not isinstance(meta_for_id_read, (pd.Series, dict)) or \
                       'Name' not in meta_for_id_read or 'File' not in meta_for_id_read:
                        print(f"警告: ID {id_k_read} 的元数据不完整或类型不正确，无法从原始文件读取。")
                        continue
                    id_meta_pairs.append((id_k_read, meta_for_id_read))
                except KeyError:
                    print(f"警告: 在 self.metadata 中找不到ID {id_k_read} 的元数据，跳过此ID的原始数据读取。")
                    continue
            
            if not id_meta_pairs:
                print(f"数据集 '{name}' 没有有效的ID元数据对进行原始数据处理。")
                continue

            results = []
            # 使用ThreadPoolExecutor进行并行读取
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(read_single_data_op, pair) for pair in id_meta_pairs]
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"并行读取 {name} 的原始数据"):
                    results.append(future.result())
            
            # 确保 Name.h5 文件的目录存在
            os.makedirs(os.path.dirname(name_cache_file), exist_ok=True)
            with h5py.File(name_cache_file, 'a') as h5f_name_specific: # 追加/创建模式，用于累积数据
                for id_res_key, data_res, error_res in results:
                    try:
                        h5_key_res = str(int(id_res_key))
                    except ValueError:
                        h5_key_res = str(id_res_key)

                    if data_res is not None:
                        if h5_key_res in h5f_name_specific:
                            del h5f_name_specific[h5_key_res] # 如果存在则先删除以覆盖
                        h5f_name_specific.create_dataset(h5_key_res, data=data_res)
                    else:
                        print(f"为 '{name}' 加载ID {id_res_key} 数据时出错: {error_res}")
                h5f_name_specific.flush()

        # 将当前任务所需的数据整合到 "cache.h5"
        final_cache_path = os.path.join(args_data.data_dir, "cache.h5")
        print(f"正在将数据整合到任务缓存文件: {final_cache_path} ...")
        # 确保最终缓存文件的目录存在
        os.makedirs(os.path.dirname(final_cache_path), exist_ok=True)

        # 先用只读模式收集缺失的 key
        missing_keys = []
        with h5py.File(final_cache_path, 'r') as h5f_consolidated:
            for id_key_final in tqdm(task_relevant_metadata.keys(), desc="检查 cache.h5 是否已存在", disable=not list(task_relevant_metadata.keys())):
                try:
                    h5_key_final = str(int(id_key_final))
                except ValueError:
                    h5_key_final = str(id_key_final)
                if h5_key_final not in h5f_consolidated:
                    missing_keys.append(id_key_final)
        if missing_keys:
            with h5py.File(final_cache_path, 'a') as h5f_consolidated: # 写入模式，为当前任务覆盖/新建
                # if not task_relevant_metadata.keys(): # 再次检查，以防万一
                #     print(f"没有相关数据ID，{final_cache_path} 将为空。")
                
                for id_key_final in tqdm(missing_keys, desc="确认并整合数据到 cache.h5"): # disable=not list(task_relevant_metadata.keys())
                    try:
                        h5_key_final = str(int(id_key_final))
                    except ValueError:
                        h5_key_final = str(id_key_final)
                    
                    try:
                        meta_final = self.metadata[id_key_final]
                        if not isinstance(meta_final, (pd.Series, dict)) or 'Name' not in meta_final:
                            print(f"警告: ID {id_key_final} 的元数据不完整或类型不正确，无法在整合阶段定位其 Name.h5 文件。")
                            continue
                    except KeyError:
                        print(f"警告: 在 self.metadata 中找不到ID {id_key_final} 的元数据，无法在整合阶段定位其 Name.h5 文件。")
                        continue

                    name_final = meta_final['Name']
                    name_specific_cache_file = os.path.join(args_data.data_dir, f"{name_final}.h5")

                    if not os.path.exists(name_specific_cache_file):
                        print(f"警告: 在整合阶段，Name.h5 文件 {name_specific_cache_file} 不存在，无法为ID {id_key_final} 提取数据。这通常不应该发生，因为之前的步骤应该已经创建/更新了它。")
                        continue
                    
                    try:
                        with h5py.File(name_specific_cache_file, 'r') as h5f_name_read:
                            if h5_key_final in h5f_name_read:
                                if h5_key_final in h5f_consolidated:
                                    continue
                                else:
                                    data_to_consolidate = h5f_name_read[h5_key_final][()]
                                    h5f_consolidated.create_dataset(h5_key_final, data=data_to_consolidate)
                            else:
                                print(f"警告: ID {h5_key_final} 在 {name_specific_cache_file} 中未找到（在整合阶段）。这可能表示之前的 Name.h5 更新步骤中存在问题。")
                    except Exception as e_read:
                        print(f"从 {name_specific_cache_file} 读取ID {h5_key_final} 时出错（在整合阶段）: {e_read}")
                h5f_consolidated.flush()
        
        print(f"数据整合完成。最终缓存文件: {final_cache_path}")
        return H5DataDict(final_cache_path)
    
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
            mod = importlib.import_module(f"src.data_factory.dataset_task.{task_type}.{task_name}_dataset")
        except ImportError as e:
            # print(f"Error importing dataset module for task {task_name}: {e}")
            print(f"Using default task.")
            mod = importlib.import_module(f"src.data_factory.dataset_task.DG.Classification_dataset")
        train_dataset = {}
        val_dataset = {}
        test_dataset = {}
        train_val_ids, test_ids = self.search_id()
        # Initialize datasets with progress bars
        print("Initializing training and validation datasets...")
        for id in tqdm(train_val_ids, desc="Creating train/val datasets"):
            train_dataset[id] = mod.set_dataset({id: self.data[id]},
                             self.target_metadata, self.args_data, self.args_task, 'train')
            val_dataset[id] = mod.set_dataset({id: self.data[id]},
                               self.target_metadata, self.args_data, self.args_task, 'val')

        print("Initializing test datasets...")
        for id in tqdm(test_ids, desc="Creating test datasets"):
            test_dataset[id] = mod.set_dataset({id: self.data[id]},
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
