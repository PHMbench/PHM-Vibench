import torch
from torch.utils.data import Dataset
# Reference:UniTS


class IdIncludedDataset(Dataset):
    def __init__(self, dataset_dict, metadata=None):
        """
        包装一个 PyTorch Dataset 字典，使得每个样本都包含其原始ID。

        Args:
            dataset_dict (dict): 一个字典，键是字符串ID，值是 PyTorch Dataset 对象。
                                 例如：{'id1': train_dataset_for_id1, 'id2': train_dataset_for_id2}
                                 其中 train_dataset_for_id1 等实例的 __getitem__ 返回 (x, y)。
        """
        self.dataset_dict = dataset_dict # 保存对原始数据集字典的引用
        self.file_windows_list = [] # 用于全局索引到 (id, 原始数据集中的索引) 的映射
        self.metadata = metadata # 保存元数据，可能包含数据集的其他信息
        for file_id, original_dataset in self.dataset_dict.items():
            if original_dataset is None:
                print(f"警告: ID '{file_id}' 对应的 dataset 为 None，已跳过。")
                continue
            if len(original_dataset) == 0:
                print(f"警告: ID '{file_id}' 对应的 dataset 为空，已跳过。")
                continue
            # if not isinstance(file_id,str):
            #     print(f"警告: ID '{file_id}' 不是字符串，已跳过。")
            #     continue
            
            for window_id in range(len(original_dataset)): # 数据集id ，样本id； 样本id 当前数据集的id
                self.file_windows_list.append({'File_id': file_id, 'Window_id': window_id}) # 1,2,3 | 1,2,3,4 ~ 1,2,3,4,5,6,7
        
        self._total_samples = len(self.file_windows_list) # 计算所有原始数据集的样本总数

    def __len__(self):
        """
        返回所有原始数据集中样本的总数。
        """
        return self._total_samples
    def get_file_windows_list(self):
        """
        获取文件窗口列表。

        Returns:
            list: 包含所有样本的文件窗口列表，每个元素是一个字典，包含 'File_id' 和 'Window_id'。
        """
        return self.file_windows_list
    def get_File_id(self, global_idx):
        """
        根据全局索引获取文件ID。

        Args:
            global_idx (int): 全局样本索引。

        Returns:
            str: 文件ID。
        """
        return self.file_windows_list[global_idx]['File_id']

    def __getitem__(self, global_idx):
        """
        根据全局索引获取样本，并返回 (id, (x, y))。

        Args:
            global_idx (int): 全局样本索引。

        Returns:
            tuple: (str, tuple), 即 (id, (x, y))
                   其中 x 是特征数据, y 是标签。
        """
        if global_idx < 0 or global_idx >= self._total_samples:
            raise IndexError(f"全局索引 {global_idx} 超出范围 (总样本数: {self._total_samples})")

        sample_info = self.file_windows_list[global_idx]

        File_id = sample_info['File_id']
        # dataset_id = self.metadata[data_id]['Dataset_id'] # 获取数据集的ID
        window_id_in_original_dataset = sample_info['Window_id']

        # 从原始数据集中获取 (x, y)
        original_dataset_instance = self.dataset_dict[File_id]
        out = original_dataset_instance[window_id_in_original_dataset] # may be (x, y) or (x, y, z)

        out.update({"Id": File_id}) # 添加 id 信息
        return  out