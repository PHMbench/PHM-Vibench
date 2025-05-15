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
from .balanced_data_loader import IdIncludedDataset,Balanced_DataLoader_Dict_Iterator # TODO del balanced_data_loader
from torch.utils.data import DataLoader
import copy
import concurrent.futures
from tqdm import tqdm  # 用于显示进度条
from torch.utils.data import Dataset
from .sampler import GroupedIdBatchSampler


def smart_read_csv(file_path, auto_detect=True):
    """智能读取CSV文件，自动尝试不同的分隔符和编码"""
    if auto_detect:
        # 先尝试检测文件前几行来判断可能的分隔符
        try:
            with open(file_path, 'rb') as f:
                sample = f.read(4096)  # 读取前4KB判断格式
                
            # 尝试用UTF-8解码
            sample_text = sample.decode('utf-8', errors='ignore')
            
            # 根据文件内容推测分隔符
            comma_count = sample_text.count(',')
            tab_count = sample_text.count('\t')
            
            # 根据分隔符频率选择解析策略
            if comma_count > tab_count:
                # 更可能是逗号分隔的文件
                try:
                    return pd.read_csv(file_path)
                except UnicodeDecodeError:
                    return pd.read_csv(file_path, encoding='gbk')
            else:
                # 更可能是制表符分隔的文件
                try:
                    return pd.read_csv(file_path, sep='\t')
                except UnicodeDecodeError:
                    return pd.read_csv(file_path, sep='\t', encoding='gbk')
                
        except Exception as e:
            print(f"自动检测格式失败: {e}，尝试默认方法")
        
        # 如果检测失败，按照优先级尝试不同组合
        encodings = ['utf-8', 'gbk', 'latin1']
        separators = [',', '\t']
        
        for encoding in encodings:
            for sep in separators:
                try:
                    return pd.read_csv(file_path, encoding=encoding, sep=sep)
                except Exception as e:
                    continue
                    
        # 最后的后备方案，使用最宽松的参数
        try:
            return pd.read_csv(file_path, encoding='latin1', sep=None, engine='python')
        except Exception as e:
            raise Exception(f"无法读取文件 {file_path}，尝试了所有常见格式: {e}")


class MetadataAccessor:
    """提供类似字典的接口访问DataFrame数据的类"""
    
    def __init__(self, dataframe, key_column='Id'):
        """初始化元数据访问器
        
        Args:
            dataframe: pandas DataFrame包含元数据
            key_column: 用作键的列名，默认为'Id'
        """
        self.df = dataframe
        self.key_column = key_column
        # 为了加速查询，将索引设置为key_column
        self.df.set_index(key_column, inplace=True, drop=False)
    
    def __getitem__(self, key):
        """通过键获取元数据行，返回一个字典
        
        Args:
            key: 要查找的键值
            
        Returns:
            dict: 包含该行所有数据的字典
        """
        try:
            return self.df.loc[key].to_dict()
        except KeyError:
            raise KeyError(f"找不到ID为{key}的记录")
    
    def __contains__(self, key):
        """检查键是否存在
        
        Args:
            key: 要检查的键值
            
        Returns:
            bool: 键是否存在
        """
        return key in self.df.index
    
    def keys(self):
        """获取所有键的列表
        
        Returns:
            列表: 所有键的列表
        """
        return list(self.df[self.key_column])
    
    def items(self):
        """获取(key, value)对的列表，类似字典的items方法
        
        Returns:
            列表: (key, value)元组的列表
        """
        for key in self.keys():
            yield (key, self[key])
    
    def values(self):
        """获取所有值的列表
        
        Returns:
            列表: 所有行数据字典的列表
        """
        return [row.to_dict() for _, row in self.df.iterrows()]
    
    def get(self, key, default=None):
        """获取键对应的值，如果不存在返回默认值
        
        Args:
            key: 要查找的键
            default: 键不存在时返回的默认值
            
        Returns:
            字典或默认值
        """
        try:
            return self[key]
        except KeyError:
            return default
    
    def __len__(self):
        """返回元数据条目数量"""
        return len(self.df)
    
    def query(self, query_str):
        """使用pandas的query功能直接查询数据
        
        Args:
            query_str: pandas query语法的查询字符串
            
        Returns:
            查询结果DataFrame
        """
        return self.df.query(query_str)




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
            dict: {id: metadata_dict} 格式的元数据字典
        """
        # 1. 读取元数据
        metadata_path = os.path.join(args_data.data_dir, args_data.metadata_file)
        meta_df = smart_read_csv(metadata_path, auto_detect=True)
        metadata =  MetadataAccessor(meta_df, key_column='Id')
        
        return metadata
    
    def _init_data(self, args_data, use_cache=True, max_workers=4):
        """
        极简数据读取器 - 并行版本
        
        Args:
            args_data: 包含data_dir和metadata_file的字典或命名空间
            use_cache: 是否使用HDF5缓存，默认为True
            max_workers: 并行工作进程数，默认为4
                
        Returns:
            H5DataDict: 数据字典对象
        """
        # 构建缓存文件路径
        cache_file = os.path.join(args_data.data_dir, f"{os.path.splitext(args_data.metadata_file)[0]}.h5")
        
        dataset_id = self.search_dataset_id()
        
        # 检查缓存
        missing_ids = []
        if use_cache and os.path.exists(cache_file):
            try:
                with h5py.File(cache_file, 'r') as h5f:
                    missing_ids = [id for id in dataset_id.keys() if str(id) not in h5f]
                    # h5f.flush()
                
                if not missing_ids:
                    print(f"所有目标数据都在缓存中，直接使用缓存文件: {cache_file}")
                    return H5DataDict(cache_file)
                else:
                    print(f"缓存中缺少 {len(missing_ids)} 个ID，将更新缓存")
            except Exception as e:
                print(f"读取缓存出错: {e}")
        
        # 定义单个ID的数据读取函数
        def read_single_data(id_meta_tuple):
            id, meta = id_meta_tuple
            try:
                name = meta['Name']
                file = meta['File']
                mod = importlib.import_module(f"src.data_factory.reader.{name}")
                file_path = os.path.join(args_data.data_dir, f'raw/{name}/{file}')
                data = mod.read(file_path, args_data)
                return id, data, None  # 返回ID、数据和错误信息(None表示没有错误)
            except Exception as e:
                return id, None, str(e)  # 返回ID、None和错误信息
        
        # 确定需要处理的ID
        ids_to_process = missing_ids if use_cache and missing_ids else list(dataset_id.keys())
        id_meta_pairs = [(id, self.metadata[id]) for id in ids_to_process if id in self.metadata]
        
        print(f"开始并行处理 {len(id_meta_pairs)} 个数据文件...")
        
        # 使用进程池并行读取数据
        results = []
        # 使用ThreadPoolExecutor代替ProcessPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(read_single_data, id_meta) for id_meta in id_meta_pairs]
            
            # 使用tqdm显示进度
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="读取数据"):
                results.append(future.result())
        
        # 写入HDF5文件
        mode = 'a' if use_cache and os.path.exists(cache_file) else 'w'
        with h5py.File(cache_file, mode) as h5f:
            for id, data, error in results:
                if data is not None:
                    if str(id) in h5f:
                        continue # del h5f[str(id)]  # 如果已存在，先删除
                    h5f.create_dataset(str(id), data=data)
                else:
                    print(f"Error loading data for ID {id}: {error}")
            h5f.flush()
        
        return H5DataDict(cache_file)
    
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
        task = self.args_task.name
        task_type = self.args_task.type
        mod = importlib.import_module(f"src.data_factory.dataset_task.{task_type}.{task}_dataset")
        train_dataset = {}
        val_dataset = {}
        test_dataset = {}
        train_val_ids, test_ids = self.search_id()
        # Initialize datasets with progress bars
        print("Initializing training and validation datasets...")
        for id in tqdm(train_val_ids, desc="Creating train/val datasets"):
            train_dataset[id] = mod.set_dataset({id: self.data[id]},
                             self.metadata, self.args_data, self.args_task, 'train')
            val_dataset[id] = mod.set_dataset({id: self.data[id]},
                               self.metadata, self.args_data, self.args_task, 'val')
        
        print("Initializing test datasets...")
        for id in tqdm(test_ids, desc="Creating test datasets"):
            test_dataset[id] = mod.set_dataset({id: self.data[id]},
                            self.metadata, self.args_data, self.args_task, 'test')
        train_dataset = IdIncludedDataset(train_dataset)
        val_dataset = IdIncludedDataset(val_dataset)
        test_dataset = IdIncludedDataset(test_dataset)
        return train_dataset, val_dataset, test_dataset
       
    def search_id(self):
        """
        should be implemented in the child class
        """
        if self.args_task.target_dataset_id is not None:
        
            if self.args_task.type == 'DG':

                # 找出Domain_id为0的行作为训练/验证集
                train_df = self.metadata.df[
                    (self.metadata.df['Domain_id'].isin(self.args_task.source_domain_id)) & 
                    (self.metadata.df['Dataset_id'].isin(self.args_task.target_dataset_id))]
                test_df = self.metadata.df[
                    (self.metadata.df['Domain_id'].isin(self.args_task.target_domain_id)) &
                    (self.metadata.df['Dataset_id'].isin(self.args_task.target_dataset_id))]
                
                self.train_val_ids = list(train_df['Id'])  # 或者 list(domain_0_df['Id'])
                self.test_ids = list(test_df['Id'])

        else:
            self.train_val_ids, self.test_ids = self.metadata.keys(), self.metadata.keys()
        return self.train_val_ids, self.test_ids    
    
    
    def search_dataset_id(self):
        
        
        if self.args_task.target_dataset_id is None:
            print("未指定目标数据集ID，返回全部元数据")
            return self.metadata
        
        # 筛选符合条件的数据
        filtered_df = self.metadata.df[
            self.metadata.df['Dataset_id'].isin(self.args_task.target_dataset_id)].copy()
        
        # 记录筛选结果
        print(f"筛选前元数据行数: {len(self.metadata.df)}")
        print(f"筛选后元数据行数: {len(filtered_df)}")
        
        if len(filtered_df) == 0:
            print(f"警告: 目标数据集ID {self.args_task.target_dataset_id} 没有匹配的记录")
        
        # 重置索引以确保索引连续
        filtered_df.reset_index(drop=True, inplace=True)
        # 创建新的MetadataAccessor对象
        self.target_metadata = MetadataAccessor(filtered_df, key_column=self.metadata.key_column)
        return self.target_metadata
        

        


    def _init_dataloader(self):
        train_batch_sampler = GroupedIdBatchSampler(
            data_source=self.train_dataset,
            batch_size=self.args_data.batch_size,
            shuffle=True,
            drop_last=True # 或 True，取决于您的需求
        )
        val_batch_sampler = GroupedIdBatchSampler(
            data_source=self.val_dataset,
            batch_size=self.args_data.batch_size,
            shuffle=False,
            drop_last=False # 或 True，取决于您的需求
        )
        test_batch_sampler = GroupedIdBatchSampler(
            data_source=self.test_dataset,
            batch_size=self.args_data.batch_size,
            shuffle=False,
            drop_last=False # 或 True，取决于您的需求
        )
        self.train_loader = DataLoader(self.train_dataset,
                                #   batch_size=self.args_data.batch_size,
                                         batch_sampler = train_batch_sampler,
                                        #  shuffle=True,
                                         num_workers=self.args_data.num_workers,
                                         pin_memory=True,     
                                         persistent_workers=True)
        self.val_loader = DataLoader(self.val_dataset,
                                #  batch_size=self.args_data.batch_size,
                                        batch_sampler = val_batch_sampler,
                                        # shuffle=False,
                                        num_workers=self.args_data.num_workers,
                                        pin_memory=True,     
                                        persistent_workers=True)
        self.test_loader = DataLoader(self.test_dataset,
                                #  batch_size=self.args_data.batch_size,
                                        batch_sampler = test_batch_sampler,
                                        # shuffle=False,
                                        num_workers=self.args_data.num_workers,
                                        pin_memory=True,     
                                        persistent_workers=True)
#################################################################### DEL ##################################################################
        # train_dataloader = {}
        # val_dataloader = {}
        # test_dataloader = {}
        
        # for id in self.train_val_ids:

        #     train_dataloader[id] = DataLoader(self.train_dataset[id],
        #                                        batch_size=self.args_data.batch_size,
        #                                               shuffle=True,
        #                                                 num_workers=self.args_data.num_workers,
        #                                                 pin_memory=False,     
        #                                                 persistent_workers=False) 
        #     val_dataloader[id] = DataLoader(self.val_dataset[id],
        #                                      batch_size=self.args_data.batch_size,
        #                                             shuffle=False,
        #                                               num_workers=self.args_data.num_workers,
        #                                                 pin_memory=False,     
        #                                                 persistent_workers=False)
        # for id in self.test_ids:
        #     test_dataloader[id] = DataLoader(self.test_dataset[id],
        #                                       batch_size=self.args_data.batch_size,
        #                                              shuffle=False,
        #                                                num_workers=self.args_data.num_workers,
        #                                                 pin_memory=False,     
        #                                                 persistent_workers=False)

        # train_loader = Balanced_DataLoader_Dict_Iterator(train_dataloader,'train',)
        # val_loader = Balanced_DataLoader_Dict_Iterator(val_dataloader,'val',)
        # test_loader = Balanced_DataLoader_Dict_Iterator(test_dataloader,'test')


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
