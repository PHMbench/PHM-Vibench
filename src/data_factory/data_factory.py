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
from .balanced_data_loader import Balanced_DataLoader_Dict_Iterator
from torch.utils.data import DataLoader

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
        meta_df = pd.read_csv(metadata_path)
        metadata = {row.Id: row.to_dict() for _, row in meta_df.iterrows()}
        
        return metadata
    
    def _init_data(self,args_data, use_cache=True):
        """
        极简数据读取器
        
        Args:
            args_data: 包含data_dir和metadata_file的字典或命名空间
            use_cache: 是否使用HDF5缓存，默认为True
            
        Returns:
            tuple: (metadata, {id: data_array})
        """
        missing_ids = []
        # 构建缓存文件路径
        cache_file = os.path.join(args_data.data_dir, f"{os.path.splitext(args_data.metadata_file)[0]}.h5")
        
        # 2. 如果存在缓存且启用了缓存，直接打开并返回
        if use_cache and os.path.exists(cache_file):
            try:
                # 检查缓存文件是否包含所有ID
                with h5py.File(cache_file, 'r') as h5f:
                    missing_ids = [id for id in self.metadata.keys() if str(id) not in h5f]
                
                if not missing_ids:
                    print(f"所有数据都在缓存中，直接使用缓存文件: {cache_file}")
                    # 直接返回 H5DataDict 对象
                    return H5DataDict(cache_file)
                else:
                    print(f"缓存中缺少ID: {missing_ids}，将更新缓存")
            except Exception as e:
                print(f"读取缓存出错: {e}")
        

        # 3. 如果没有缓存或有缺失数据，读取原始数据
        with h5py.File(cache_file, 'w') as h5f:
# 确定需要处理的 ID 列表
            ids_to_process = missing_ids if use_cache and missing_ids else list(self.metadata.keys())

            for id, meta in self.metadata.items():
                if id in ids_to_process:
                    print(f"ID {id} 缺失，重新加载数据")
                    try:
                        name = meta['Name']
                        file = meta['File']
                        mod = importlib.import_module(f"src.data_factory.reader.{name}")
                        file_path = os.path.join(args_data.data_dir, f'raw/{name}/{file}')
                        data = mod.read(args_data, file_path)
                        h5f.create_dataset(str(id), data=data)
                    except Exception as e:
                        print(f"Error loading data for ID {id}: {e}")
            h5f.flush()

        return H5DataDict(cache_file)
    
    def get_metadata(self):
        """获取元数据"""
        return self.metadata
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
        for id in train_val_ids:

            train_dataset[id] = mod.set_dataset({id: self.data[id]},
                                                 self.metadata, self.args_data, self.args_task, 'train')
            val_dataset[id] = mod.set_dataset({id: self.data[id]},
                                               self.metadata, self.args_data, self.args_task, 'val')
        for id in test_ids:
            test_dataset[id] = mod.set_dataset({id: self.data[id]},
                                                self.metadata, self.args_data, self.args_task, 'test')
        return train_dataset, val_dataset, test_dataset
       
    def search_id(self):
        """
        should be implemented in the child class
        """
        train_val_ids, test_ids = self.metadata.keys(), self.metadata.keys()
        return train_val_ids, test_ids

    def _init_dataloader(self):

        train_dataloader = {}
        val_dataloader = {}
        test_dataloader = {}
        for id, meta in self.metadata.items():
            train_dataloader[id] = DataLoader(self.train_dataset[id],
                                               batch_size=self.args_data.batch_size,
                                                      shuffle=True,
                                                        num_workers=self.args_data.num_workers,
                                                        pin_memory=True,     
                                                        persistent_workers=True) 
            val_dataloader[id] = DataLoader(self.val_dataset[id],
                                             batch_size=self.args_data.batch_size,
                                                    shuffle=False,
                                                      num_workers=self.args_data.num_workers,
                                                        pin_memory=True,     
                                                        persistent_workers=True)
            test_dataloader[id] = DataLoader(self.test_dataset[id],
                                              batch_size=self.args_data.batch_size,
                                                     shuffle=False,
                                                       num_workers=self.args_data.num_workers,
                                                        pin_memory=True,     
                                                        persistent_workers=True)

        train_loader = Balanced_DataLoader_Dict_Iterator(train_dataloader,'train',)
        val_loader = Balanced_DataLoader_Dict_Iterator(val_dataloader,'val',)
        test_loader = Balanced_DataLoader_Dict_Iterator(test_dataloader,'test')


        return train_loader, val_loader, test_loader

    def get_dataset(self, mode = "train"):
        """获取指定ID的数据集
        
        Args:
            id: 数据集ID
        
        Returns:
            数据集
        """
        return self.train_dataset if mode == "train" else self.val_dataset if mode == "val" else self.test_dataset
    def get_dataloader(self, mode = "train"):
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
