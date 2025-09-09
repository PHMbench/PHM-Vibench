import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
import random
from ..dataset_task.Dataset_cluster import IdIncludedDataset
from .FS_sampler import HierarchicalFewShotSampler



class Same_system_Sampler(Sampler):

    def __init__(self, dataset: IdIncludedDataset,
                  batch_size: int,
                    shuffle: bool = True,
                      drop_last: bool = False,
                        system_metadata_key: str = 'Dataset_id'):
        """
        Batch sampler，确保每个批次中的所有样本都来自同一个 system_id。
        system_id 是从 dataset.metadata[file_id][system_metadata_key] 获取的。

        Args:
            dataset (IdIncludedDataset): 要从中采样的 IdIncludedDataset 实例。
            batch_size (int): 每个批次的大小。
            shuffle (bool): 如果为 True，则会打乱系统ID的处理顺序，
                             打乱每个系统内的样本顺序，
                             并且（可选地）打乱生成的批次本身的顺序。
            drop_last (bool): 如果为 True，则对于每个系统，如果其最后一个批次小于 batch_size，则丢弃它。
            system_metadata_key (str): 用于从 dataset.metadata 中查找系统ID的键。
        """
        # if not isinstance(dataset, IdIncludedDataset):
        #     raise ValueError("dataset 必须是 IdIncludedDataset 的实例。")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size 必须是正整数。")
        if not hasattr(dataset, 'metadata') or dataset.metadata is None:
            raise ValueError("dataset 必须具有 'metadata' 属性。")


        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.system_metadata_key = system_metadata_key

        # 1. 按 system_id 对全局索引进行分组
        self.indices_per_system = {}
        for global_idx, sample_info in enumerate(self.dataset.file_windows_list):
            file_id = sample_info['file_id']
            if file_id not in self.dataset.metadata:
                # print(f"警告: file_id '{file_id}' 在元数据中未找到，跳过样本 {global_idx}")
                continue
            
            meta_entry = self.dataset.metadata[file_id]
            if self.system_metadata_key not in meta_entry:
                # print(f"警告: 系统元数据键 '{self.system_metadata_key}' 在 file_id '{file_id}' 的元数据中未找到，跳过样本 {global_idx}")
                continue

            system_id = meta_entry[self.system_metadata_key]

            if system_id not in self.indices_per_system:
                self.indices_per_system[system_id] = []
            self.indices_per_system[system_id].append(global_idx)
        
        self.system_id_list = list(self.indices_per_system.keys())
        
        # 2. 预计算此 sampler 在一个 epoch 中将生成的总批次数
        self._num_batches_epoch = 0
        if self.system_id_list:
            for system_id in self.system_id_list:
                num_samples_for_system = len(self.indices_per_system[system_id])
                if num_samples_for_system == 0: continue

                if self.drop_last:
                    self._num_batches_epoch += num_samples_for_system // self.batch_size
                else:
                    self._num_batches_epoch += (num_samples_for_system + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        all_batches_for_epoch = []
        
        systems_to_process = list(self.system_id_list)
        if self.shuffle:
            random.shuffle(systems_to_process) # 打乱系统ID的处理顺序

        for system_id in systems_to_process:
            system_specific_global_indices = list(self.indices_per_system[system_id])
            if not system_specific_global_indices:
                continue

            if self.shuffle:
                random.shuffle(system_specific_global_indices) # 打乱当前系统内的样本顺序
            
            for i in range(0, len(system_specific_global_indices), self.batch_size):
                batch_indices = system_specific_global_indices[i : i + self.batch_size]
                
                if len(batch_indices) < self.batch_size and self.drop_last:
                    continue
                
                all_batches_for_epoch.append(batch_indices)
        
        # (可选) 如果需要，可以对所有生成的批次本身再进行一次全局打乱。
        # 这会使得来自不同系统ID的批次序列完全随机，而不仅仅是系统ID处理顺序随机。
        if self.shuffle and len(all_batches_for_epoch) > 1 :
            random.shuffle(all_batches_for_epoch)
            
        return iter(all_batches_for_epoch)

    def __len__(self):
        """返回一个 epoch 中的总批次数。"""
        return self._num_batches_epoch

class BalancedIdSampler(Sampler):
    def __init__(self, data_source: IdIncludedDataset, batch_size=32, common_samples_per_id=None, shuffle_within_id=True, shuffle_all=True):
        """
        Sampler 实现对不同原始数据集(ID)的平衡加载。

        Args:
            data_source (IdIncludedDataset): 被包装的数据集，必须是 IdIncludedDataset 类型。
            common_samples_per_id (int, optional): 
                每个ID在一个epoch中被采样的目标次数。
                如果为 None, 则默认为样本量最大的ID的样本数 (即对小ID进行过采样)。
            shuffle_within_id (bool): 是否在为每个ID选择样本时进行随机选择。
            shuffle_all (bool): 是否在最后将所有选出的索引进行全局随机打乱。
        """
        super().__init__(data_source,batch_size = batch_size)
        self.data_source = data_source
        self.shuffle_within_id = shuffle_within_id
        self.shuffle_all = shuffle_all

        # 1. 按ID对全局索引进行分组
        self.indices_per_id = {}
        if not hasattr(self.data_source, 'flat_sample_map'):
            raise ValueError("data_source 必须是 IdIncludedDataset 的实例，或具有 'flat_sample_map' 属性。")

        for global_idx, sample_info in enumerate(self.data_source.flat_sample_map):
            original_id = sample_info['id']
            if original_id not in self.indices_per_id:
                self.indices_per_id[original_id] = []
            self.indices_per_id[original_id].append(global_idx)

        self.id_list = list(self.indices_per_id.keys())
        if not self.id_list: # 如果没有任何有效的ID
            print("没有有效的ID，Sampler将不会工作。")
            self._num_samples_epoch = 0
            self.target_samples_per_id = 0
            return

        # 2. 确定每个ID的目标采样数
        num_actual_samples_per_id = {id_str: len(indices) for id_str, indices in self.indices_per_id.items()}
        
        if common_samples_per_id is None:
            # 默认目标是最大ID的样本量
            if not num_actual_samples_per_id:
                 self.target_samples_per_id = 0 # 防止空字典
            else:
                 self.target_samples_per_id = max(num_actual_samples_per_id.values()) if num_actual_samples_per_id else 0
        else:
            self.target_samples_per_id = common_samples_per_id
        
        # 3. 计算一个epoch的总样本数
        self._num_samples_epoch = self.target_samples_per_id * len(self.id_list)

    def __iter__(self):
        if not self.id_list or self.target_samples_per_id == 0:
            return iter([])

        all_epoch_indices = []
        for id_str in self.id_list:
            id_specific_global_indices = self.indices_per_id[id_str]
            num_actual_id_samples = len(id_specific_global_indices)

            if num_actual_id_samples == 0:
                continue # 这个ID没有样本

            if num_actual_id_samples < self.target_samples_per_id:
                # 过采样: 需要重复采样
                # random.choices 实现带放回采样
                chosen_for_id = random.choices(id_specific_global_indices, k=self.target_samples_per_id)
            elif num_actual_id_samples > self.target_samples_per_id:
                # 欠采样或精确采样: 不带放回采样
                if self.shuffle_within_id:
                    chosen_for_id = random.sample(id_specific_global_indices, k=self.target_samples_per_id)
                else: # 按顺序取前k个
                    chosen_for_id = id_specific_global_indices[:self.target_samples_per_id]
            else: # num_actual_id_samples == self.target_samples_per_id
                # 样本数正好等于目标数
                if self.shuffle_within_id:
                    chosen_for_id = random.sample(id_specific_global_indices, k=num_actual_id_samples)
                else:
                    chosen_for_id = list(id_specific_global_indices) # 复制列表
            
            all_epoch_indices.extend(chosen_for_id)

        if self.shuffle_all:
            random.shuffle(all_epoch_indices) # 打乱所有ID组合后的总索引列表
        
        return iter(all_epoch_indices)

    def __len__(self):
        return self._num_samples_epoch
    
    
    
class GroupedIdBatchSampler(Sampler):
    def __init__(self, data_source: IdIncludedDataset, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        """
        Batch sampler，确保每个批次中的所有样本都来自同一个 original_id。

        Args:
            data_source (IdIncludedDataset): 要从中采样的 IdIncludedDataset 实例。
            batch_size (int): 每个批次的大小。
            shuffle (bool): 如果为 True，则会打乱ID的处理顺序，并打乱每个ID内的样本顺序，
                             最终还会打乱生成的批次本身的顺序。
            drop_last (bool): 如果为 True，则对于每个ID，如果其最后一个批次小于 batch_size，则丢弃它。
        """
        if not isinstance(data_source, IdIncludedDataset):
            raise ValueError("data_source 必须是 IdIncludedDataset 的实例。")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size 必须是正整数。")

        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # 1. 按 original_id 对全局索引进行分组
        self.indices_per_id = {}
        for global_idx, sample_info in enumerate(self.data_source.file_windows_list):
            file_id = sample_info['file_id']

            if file_id not in self.indices_per_id:
                self.indices_per_id[file_id] = []
            self.indices_per_id[file_id].append(global_idx) # 每个dataset的id
        
        self.file_id_list = list(self.indices_per_id.keys())

        # 2. 预计算此 sampler 在一个 epoch 中将生成的总批次数
        self._num_batches_epoch = 0
        if self.file_id_list: # 仅当有ID时才计算
            for file_id in self.file_id_list:
                num_samples_for_id = len(self.indices_per_id[file_id])
                if num_samples_for_id == 0: continue

                if self.drop_last:
                    self._num_batches_epoch += num_samples_for_id // self.batch_size
                else:
                    self._num_batches_epoch += (num_samples_for_id + self.batch_size - 1) // self.batch_size # 向上取整

    def __iter__(self):
        all_batches_for_epoch = [] # 存储这个 epoch 要生成的所有批次 (每个批次是索引列表)
        
        ids_to_process = list(self.id_list) # 复制一份ID列表以进行可能的打乱
        if self.shuffle:
            random.shuffle(ids_to_process) # 打乱ID的处理顺序

        for id_str in ids_to_process:
            id_specific_global_indices = list(self.indices_per_id[id_str]) # 复制以进行可能的打乱
            if not id_specific_global_indices: # 如果某个ID没有样本 (理论上不应发生，因为IdIncludedDataset会跳过空dataset)
                continue

            if self.shuffle:
                random.shuffle(id_specific_global_indices) # 打乱当前ID内的样本顺序
            
            # 为当前ID创建批次
            for i in range(0, len(id_specific_global_indices), self.batch_size):
                batch_indices = id_specific_global_indices[i : i + self.batch_size]
                
                if len(batch_indices) < self.batch_size and self.drop_last:
                    # 如果是最后一个批次且不完整，并且设置了drop_last，则丢弃
                    continue
                
                all_batches_for_epoch.append(batch_indices)
        
        # # (可选) 对所有生成的批次本身再进行一次全局打乱
        # # 这会使得来自不同ID的批次序列完全随机，而不仅仅是ID处理顺序随机
        # if self.shuffle and len(all_batches_for_epoch) > 1 :
        #     random.shuffle(all_batches_for_epoch)
            
        return iter(all_batches_for_epoch) # [[1,2,3],[4,5,6],...,[7,8,9]] 这样的列表

    def __len__(self):
        """返回一个 epoch 中的总批次数。"""
        return self._num_batches_epoch