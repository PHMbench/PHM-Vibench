import torch
# Reference:UniTS
class BalancedDataLoaderIterator:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders

        self.num_dataloaders = len(dataloaders)

        max_length = max(len(dataloader) for dataloader in dataloaders)

        length_list = [len(dataloader) for dataloader in dataloaders]
        print("data loader length:", length_list)
        print("max dataloader length:", max_length,
              "epoch iteration:", max_length * self.num_dataloaders)
        self.total_length = max_length * self.num_dataloaders
        self.current_iteration = 0
        self.probabilities = torch.ones(
            self.num_dataloaders, dtype=torch.float) / self.num_dataloaders

    def __iter__(self):
        self.iterators = [iter(dataloader) for dataloader in self.dataloaders]
        self.current_iteration = 0
        return self

    def __next__(self):
        if self.current_iteration >= self.total_length:
            raise StopIteration

        chosen_index = torch.multinomial(self.probabilities, 1).item()
        try:
            sample = next(self.iterators[chosen_index])
        except StopIteration:
            self.iterators[chosen_index] = iter(self.dataloaders[chosen_index])
            sample = next(self.iterators[chosen_index])

        self.current_iteration += 1
        return sample, chosen_index

    def __len__(self):
        return self.total_length

    def generate_fake_samples_for_batch(self, dataloader_id, batch_size):  # Try
        if dataloader_id >= len(self.dataloaders) or dataloader_id < 0:
            raise ValueError("Invalid dataloader ID")

        dataloader = self.dataloaders[dataloader_id]
        iterator = iter(dataloader)

        try:
            sample_batch = next(iterator)
            fake_samples = []

            for sample in sample_batch:
                if isinstance(sample, torch.Tensor):
                    fake_sample = torch.zeros(
                        [batch_size] + list(sample.shape)[1:])
                    fake_samples.append(fake_sample)
                else:
                    pass

            return fake_samples, dataloader_id
        except StopIteration:
            return None


class Balanced_DataLoader_Dict_Iterator:
    def __init__(self, dataloaders_dict, mode='test'):
        """
        初始化平衡数据加载器迭代器
        
        Args:
            dataloaders_dict: 数据加载器字典 {数据集名称: DataLoader}
            mode: 运行模式，可选值为 'train', 'val', 'test'
                 'train' - 随机采样各数据集
                 'val'/'test' - 顺序遍历所有数据集
        """
        self.dataloaders_dict = dataloaders_dict
        self.data_names = list(dataloaders_dict.keys())
        self.num_dataloaders = len(dataloaders_dict)
        self.mode = mode
        
        # 创建各个模式的索引和迭代器
        if mode in ['val', 'test']:
            # 验证/测试模式：准备顺序索引
            self.current_dataloader_idx = 0
            self.total_samples = sum(len(dataloader.dataset) for dataloader in dataloaders_dict.values())
            self.samples_seen = 0
            
            # 记录每个数据集的样本数量，用于确定何时切换到下一个数据集
            self.dataset_sizes = {name: len(loader.dataset) for name, loader in dataloaders_dict.items()}
            self.current_dataset_samples_seen = 0
        else:
            # 训练模式：使用随机抽样
            self.max_length = max(len(dataloader) for dataloader in dataloaders_dict.values())
            length_list = [len(dataloader) for dataloader in dataloaders_dict.values()]
            print("data loader length:", length_list)
            print("max dataloader length:", self.max_length,
                  "epoch iteration:", self.max_length * self.num_dataloaders)
            self.total_length = self.max_length * self.num_dataloaders
            self.probabilities = torch.ones(
                self.num_dataloaders, dtype=torch.float) / self.num_dataloaders
        
        self.current_iteration = 0
        
        # 初始化迭代器
        self.iterators = {data_name: iter(dataloader) 
                         for data_name, dataloader in self.dataloaders_dict.items()}

    def __iter__(self):
        # 重置迭代器和计数器
        self.iterators = {data_name: iter(dataloader) 
                         for data_name, dataloader in self.dataloaders_dict.items()}
        self.current_iteration = 0
        
        if self.mode in ['val', 'test']:
            self.current_dataloader_idx = 0
            self.current_dataset_samples_seen = 0
            self.samples_seen = 0
            
        return self

    def __next__(self):
        if self.mode in ['val', 'test']:
            # 验证/测试模式：顺序迭代
            if self.samples_seen >= self.total_samples:
                raise StopIteration
            
            # 获取当前数据集名称
            chosen_data_name = self.data_names[self.current_dataloader_idx]
            
            try:
                sample = next(self.iterators[chosen_data_name])
                batch_size = sample[0].shape[0] if isinstance(sample[0], torch.Tensor) else 1
                self.samples_seen += batch_size
                self.current_dataset_samples_seen += batch_size
                
                # 检查是否需要切换到下一个数据集
                if self.current_dataset_samples_seen >= self.dataset_sizes[chosen_data_name]:
                    self.current_dataloader_idx = (self.current_dataloader_idx + 1) % self.num_dataloaders
                    self.current_dataset_samples_seen = 0
                
                return sample, chosen_data_name
            except StopIteration:
                # 如果当前数据集迭代完成，切换到下一个
                self.current_dataloader_idx = (self.current_dataloader_idx + 1) % self.num_dataloaders
                self.current_dataset_samples_seen = 0
                
                if self.current_dataloader_idx == 0:  # 所有数据集都迭代完了一遍
                    raise StopIteration
                
                # 重置当前数据集的迭代器
                chosen_data_name = self.data_names[self.current_dataloader_idx]
                self.iterators[chosen_data_name] = iter(self.dataloaders_dict[chosen_data_name])
                return self.__next__()  # 递归调用，获取下一个数据集的样本
        else:
            # 训练模式：随机抽样
            if self.current_iteration >= self.total_length:
                raise StopIteration

            # 随机选择一个数据名称
            chosen_index = torch.multinomial(self.probabilities, 1).item()
            chosen_data_name = self.data_names[chosen_index]
            
            try:
                sample = next(self.iterators[chosen_data_name])
            except StopIteration:
                # 重新初始化迭代器
                self.iterators[chosen_data_name] = iter(self.dataloaders_dict[chosen_data_name])
                sample = next(self.iterators[chosen_data_name])

            self.current_iteration += 1
            return sample, chosen_data_name

    def __len__(self):
        if self.mode in ['val', 'test']:
            return sum(len(dataloader) for dataloader in self.dataloaders_dict.values())
        else:
            return self.total_length

    def generate_fake_samples_for_batch(self, data_name, batch_size):
        if data_name not in self.dataloaders_dict:
            raise ValueError(f"Invalid data name: {data_name}")

        dataloader = self.dataloaders_dict[data_name]
        iterator = iter(dataloader)

        try:
            sample_batch = next(iterator)
            fake_samples = []

            for sample in sample_batch:
                if isinstance(sample, torch.Tensor):
                    fake_sample = torch.zeros(
                        [batch_size] + list(sample.shape)[1:])
                    fake_samples.append(fake_sample)
                else:
                    pass

            return fake_samples, data_name
        except StopIteration:
            return None
        
#%% 
if __name__ == "__main__":
    # 测试基本初始化和迭代
    import torch
    from torch.utils.data import TensorDataset, DataLoader        
    # 创建测试数据
    def create_test_dataloaders():
        # 创建三个不同的数据集
        data1 = torch.randn(100, 10)  # 100个样本，每个10维特征
        labels1 = torch.randint(0, 2, (100,))  # 二分类标签
        dataset1 = TensorDataset(data1, labels1)
        
        data2 = torch.randn(80, 10)  # 80个样本
        labels2 = torch.randint(0, 3, (80,))  # 三分类标签
        dataset2 = TensorDataset(data2, labels2)
        
        data3 = torch.randn(120, 10)  # 120个样本
        labels3 = torch.randint(0, 4, (120,))  # 四分类标签
        dataset3 = TensorDataset(data3, labels3)
        
        # 创建数据加载器字典
        dataloaders_dict = {
            "dataset_a": DataLoader(dataset1, batch_size=16, shuffle=True),
            "dataset_b": DataLoader(dataset2, batch_size=16, shuffle=True),
            "dataset_c": DataLoader(dataset3, batch_size=16, shuffle=True)
        }
        
        return dataloaders_dict

    # 测试基本初始化和迭代
    def test_basic_iteration():
        dataloaders_dict = create_test_dataloaders()
        
        # 初始化 Balanced_DataLoader_Dict_Iterator
        iterator = Balanced_DataLoader_Dict_Iterator(dataloaders_dict)
        iterator = iter(iterator)
        
        print(f"Number of dataloaders: {iterator.num_dataloaders}")
        print(f"Total length: {iterator.total_length}")
        
        # 迭代几次并打印结果
        samples_per_dataset = {"dataset_a": 0, "dataset_b": 0, "dataset_c": 0}
        
        for i in range(30):  # 迭代30次
            sample, data_name = next(iterator)
            samples_per_dataset[data_name] += 1
            print(f"Iteration {i}, Dataset: {data_name}, Sample shape: {sample[0].shape}")
        
        print("Samples per dataset:", samples_per_dataset)

    # 运行测试


    def test_fake_samples_generation():
        dataloaders_dict = create_test_dataloaders()
        
        # 初始化 Balanced_DataLoader_Dict_Iterator
        iterator = Balanced_DataLoader_Dict_Iterator(dataloaders_dict)
        
        # 为每个数据集生成假样本
        batch_size = 8
        for data_name in dataloaders_dict.keys():
            result = iterator.generate_fake_samples_for_batch(data_name, batch_size)
            if result:
                fake_samples, returned_name = result
                print(f"Dataset: {data_name}")
                print(f"  Returned name: {returned_name}")
                print(f"  Number of fake samples: {len(fake_samples)}")
                for i, sample in enumerate(fake_samples):
                    print(f"  Sample {i} shape: {sample.shape}")
                print()

    # 运行测试


    def test_different_dataset_sizes():
        # 创建非常不平衡的数据集
        data1 = torch.randn(200, 10)  # 200个样本
        labels1 = torch.randint(0, 2, (200,))
        dataset1 = TensorDataset(data1, labels1)
        
        data2 = torch.randn(50, 10)  # 50个样本
        labels2 = torch.randint(0, 3, (50,))
        dataset2 = TensorDataset(data2, labels2)
        
        data3 = torch.randn(10, 10)  # 10个样本
        labels3 = torch.randint(0, 4, (10,))
        dataset3 = TensorDataset(data3, labels3)
        
        # 创建数据加载器字典，使用不同的批量大小
        dataloaders_dict = {
            "large_dataset": DataLoader(dataset1, batch_size=20, shuffle=True),
            "medium_dataset": DataLoader(dataset2, batch_size=10, shuffle=True),
            "small_dataset": DataLoader(dataset3, batch_size=2, shuffle=True)
        }
        
        # 初始化 Balanced_DataLoader_Dict_Iterator
        iterator = Balanced_DataLoader_Dict_Iterator(dataloaders_dict)
        
        print(f"Maximum length: {iterator.max_length}")
        
        # 迭代并计数
        dataset_counts = {"large_dataset": 0, "medium_dataset": 0, "small_dataset": 0}
        
        # 迭代一个完整的周期
        for i in range(iterator.total_length):
            _, data_name = next(iterator)
            dataset_counts[data_name] += 1
        
        print("Dataset counts after one full cycle:")
        for name, count in dataset_counts.items():
            print(f"  {name}: {count}")

    # 运行测试


    def test_error_handling():
        dataloaders_dict = create_test_dataloaders()
        
        # 初始化 Balanced_DataLoader_Dict_Iterator
        iterator = Balanced_DataLoader_Dict_Iterator(dataloaders_dict)
        
        # 测试无效数据名称
        try:
            iterator.generate_fake_samples_for_batch("non_existent_dataset", 8)
        except ValueError as e:
            print(f"Expected error caught: {e}")
        
        # 测试迭代结束
        try:
            # 先耗尽迭代器
            for _ in range(iterator.total_length):
                next(iterator)
            
            # 应该抛出 StopIteration
            next(iterator)
        except StopIteration:
            print("Correctly raised StopIteration at the end")

    # 运行测试
    try:
        # test_basic_iteration()
        test_different_dataset_sizes()
        test_fake_samples_generation()
        test_error_handling()
    except Exception as e:
        print(e)
