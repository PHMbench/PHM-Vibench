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
from .balanced_data_loader import IdIncludedDataset # ,Balanced_DataLoader_Dict_Iterator # TODO del balanced_data_loader
from torch.utils.data import DataLoader
import copy
import concurrent.futures
from tqdm import tqdm  # 用于显示进度条
from torch.utils.data import Dataset
from .samplers.sampler import GroupedIdBatchSampler, BalancedIdSampler


def smart_read_csv(file_path, auto_detect=True):
    """智能读取CSV/Excel文件，自动尝试不同的分隔符和编码"""
    # 检查文件扩展名
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # 如果是Excel文件，直接使用pandas读取
    if file_ext in ['.xlsx', '.xls']:
        try:
            return pd.read_excel(file_path)
        except Exception as e:
            print(f"读取Excel文件失败: {e}")
            raise Exception(f"无法读取Excel文件 {file_path}: {e}")
    
    # CSV读取逻辑
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
                    return pd.read_csv(file_path, sep='\t', encoding='gbk', low_memory=False)
                
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
            return self.df.loc[key]#.to_dict()
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

        with h5py.File(final_cache_path, 'a') as h5f_consolidated: # 写入模式，为当前任务覆盖/新建
            if not task_relevant_metadata.keys(): # 再次检查，以防万一
                print(f"没有相关数据ID，{final_cache_path} 将为空。")
            
            for id_key_final in tqdm(task_relevant_metadata.keys(), desc="确认并整合数据到 cache.h5", disable=not list(task_relevant_metadata.keys())):
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
        train_dataset = IdIncludedDataset(train_dataset,self.metadata)
        val_dataset = IdIncludedDataset(val_dataset,self.metadata)
        test_dataset = IdIncludedDataset(test_dataset,self.metadata)
        return train_dataset, val_dataset, test_dataset
       
    def search_id(self):
        """
        should be implemented in the child class
        """

        def remove_invalid_labels(df, label_column='Label'):
            """
            从DataFrame中删除指定列值为-1的行
            
            Args:
                df (pd.DataFrame): 输入的DataFrame
                label_column (str): 标签列名，默认为'Label'
            
            Returns:
                pd.DataFrame: 删除指定条件行后的新DataFrame
            """
            # 检查列是否存在
            if label_column not in df.columns:
                raise ValueError(f"列 '{label_column}' 不存在于DataFrame中")
            
            # 删除Label为-1的行
            filtered_df = df[df[label_column] != -1].copy()
            
            # 重置索引（可选）
            filtered_df.reset_index(drop=True, inplace=True)
            
            return filtered_df
            
            
        if self.args_task.target_dataset_id is not None:
        
            if self.args_task.type == 'DG':

                # 找出Domain_id为0的行作为训练/验证集
                train_df = self.metadata.df[
                    (self.metadata.df['Domain_id'].isin(self.args_task.source_domain_id)) & 
                    (self.metadata.df['Dataset_id'].isin(self.args_task.target_dataset_id))]
                test_df = self.metadata.df[
                    (self.metadata.df['Domain_id'].isin(self.args_task.target_domain_id)) &
                    (self.metadata.df['Dataset_id'].isin(self.args_task.target_dataset_id))]
                
                train_df = remove_invalid_labels(train_df)
                test_df = remove_invalid_labels(test_df)
                
                self.train_val_ids = list(train_df['Id'])  # 或者 list(domain_0_df['Id'])
                self.test_ids = list(test_df['Id'])
            elif self.args_task.type == 'CDDG':
                # 筛选出目标数据集
                filtered_df = self.metadata.df[self.metadata.df['Dataset_id'].isin(self.args_task.target_dataset_id)]

                filtered_df = remove_invalid_labels(filtered_df)
                
                # 找出每个数据集中的所有唯一domain_id
                dataset_domains = {}
                for dataset_id in self.args_task.target_dataset_id:
                    dataset_df = filtered_df[filtered_df['Dataset_id'] == dataset_id]
                    domains = sorted(dataset_df['Domain_id'].unique())
                    # Filter out NaN values from domains
                    domains = [d for d in domains if not pd.isna(d)]
                    dataset_domains[dataset_id] = domains
                
                # 为每个数据集选择训练和测试domain
                train_domains = {}
                test_domains = {}
                for dataset_id, domains in dataset_domains.items():
                    test_count = min(self.args_task.target_domain_num, len(domains))
                    train_domains[dataset_id] = domains[:-test_count] if test_count > 0 else domains
                    test_domains[dataset_id] = domains[-test_count:] if test_count > 0 else []
                
                # 构建训练和测试集
                train_rows = []
                test_rows = []
                for dataset_id in self.args_task.target_dataset_id:
                    # 训练集rows
                    for domain_id in train_domains[dataset_id]:
                        train_rows.extend(
                            filtered_df[(filtered_df['Dataset_id'] == dataset_id) & 
                                    (filtered_df['Domain_id'] == domain_id)]['Id'].tolist()
                        )
                    # 测试集rows
                    for domain_id in test_domains[dataset_id]:
                        test_rows.extend(
                            filtered_df[(filtered_df['Dataset_id'] == dataset_id) & 
                                    (filtered_df['Domain_id'] == domain_id)]['Id'].tolist()
                        )
                
                self.train_val_ids = train_rows
                self.test_ids = test_rows
                
                # 记录分割信息
                print(f"CDGD划分 - 选择每个数据集的最后{self.args_task.target_domain_num}个domain作为测试集")
                for dataset_id in self.args_task.target_dataset_id:
                    print(f"数据集 {dataset_id}:")
                    print(f"  - 训练域: {train_domains[dataset_id]}")
                    print(f"  - 测试域: {test_domains[dataset_id]}")
                print(f"训练/验证样本数: {len(self.train_val_ids)}")
                print(f"测试样本数: {len(self.test_ids)}")
                
        
        else:
            self.train_val_ids, self.test_ids = self.metadata.keys(), self.metadata.keys()
        return self.train_val_ids, self.test_ids    
    
    
    def search_dataset_id(self):
        
        
        if not self.args_task.target_dataset_id:
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
        # few-shot mode uses special episode dataset
        if self.args_task.type == 'FS':
            from .dataset_task.FS.episode_dataset import FewShotEpisodeDataset
            n_way = getattr(self.args_task, 'n_way', 5)
            k_shot = getattr(self.args_task, 'k_shot', 1)
            q_query = getattr(self.args_task, 'q_query', 1)
            episodes = getattr(self.args_task, 'episodes_per_epoch', 100)
            self.train_loader = DataLoader(
                FewShotEpisodeDataset(self.train_dataset, n_way, k_shot, q_query, episodes),
                batch_size=1,
                num_workers=self.args_data.num_workers,
                pin_memory=True,
            )
            self.val_loader = DataLoader(
                FewShotEpisodeDataset(self.val_dataset, n_way, k_shot, q_query, episodes),
                batch_size=1,
                num_workers=self.args_data.num_workers,
                pin_memory=True,
            )
            self.test_loader = DataLoader(
                FewShotEpisodeDataset(self.test_dataset, n_way, k_shot, q_query, episodes),
                batch_size=1,
                num_workers=self.args_data.num_workers,
                pin_memory=True,
            )
            return self.train_loader, self.val_loader, self.test_loader

        train_batch_sampler = GroupedIdBatchSampler(
            data_source=self.train_dataset,
            batch_size=self.args_data.batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_batch_sampler = GroupedIdBatchSampler(
            data_source=self.val_dataset,
            batch_size=self.args_data.batch_size,
            shuffle=False,
            drop_last=True # 或 True，取决于您的需求
        )
        test_batch_sampler = GroupedIdBatchSampler(
            data_source=self.test_dataset,
            batch_size=self.args_data.batch_size,
            shuffle=False,
            drop_last=True # 或 True，取决于您的需求
        )
        
        # def debug_collate_fn(batch):
        #     """
        #     自定义collate函数，修复字节序问题并提供调试信息
        #     """
        #     import torch
        #     import numpy as np
        #     from torch.utils.data._utils.collate import default_collate

        #     def fix_byte_order(item):
        #         """修复NumPy数组的字节序问题"""
        #         if isinstance(item, np.ndarray) and item.dtype.byteorder not in ('=', '|'):
        #             print(f"修复字节序: {item.dtype} -> {item.dtype.newbyteorder('=')}")
        #             return item.astype(item.dtype.newbyteorder('='), copy=False)
        #         return item

        #     def process_item(item):
        #         """递归处理复杂数据结构"""
        #         if isinstance(item, dict):
        #             return {k: process_item(v) for k, v in item.items()}
        #         elif isinstance(item, (list, tuple)):
        #             return type(item)(process_item(i) for i in item)
        #         else:
        #             return fix_byte_order(item)

        #     # 打印批次结构，帮助调试
        #     print(f"\n>>> 批次类型: {type(batch)}, 长度: {len(batch)}")
        #     if batch and isinstance(batch[0], tuple):
        #         print(f">>> 第一个样本类型: {type(batch[0])}, 长度: {len(batch[0])}")
        #         for i, part in enumerate(batch[0]):
        #             print(f">>> 样本部分[{i}]类型: {type(part)}")
        #             if isinstance(part, dict):
        #                 print(f">>> 字典键: {list(part.keys())}")

        #     # 处理所有样本
        #     processed_batch = [process_item(item) for item in batch]
            
        #     try:
        #         # 尝试使用默认collate函数
        #         return default_collate(processed_batch)
        #     except Exception as e:
        #         print(f">>> 默认collate失败: {e}")
        #         # 如果失败，返回处理后但未合并的批次
        #         return processed_batch
        persistent_workers = self.args_data.num_workers > 0
        self.train_loader = DataLoader(self.train_dataset,
                                #   batch_size=self.args_data.batch_size,
                                         batch_sampler = train_batch_sampler,
                                        #  shuffle=True,
                                         num_workers=self.args_data.num_workers,
                                         pin_memory=True,     
                                         persistent_workers=persistent_workers,)
                                        #  collate_fn=debug_collate_fn)
        self.val_loader = DataLoader(self.val_dataset,
                                #  batch_size=self.args_data.batch_size,
                                        batch_sampler = val_batch_sampler,
                                        # shuffle=False,
                                        num_workers=self.args_data.num_workers,
                                        pin_memory=True,     
                                        persistent_workers=persistent_workers,)
                                                                                #  collate_fn=debug_collate_fn)
        self.test_loader = DataLoader(self.test_dataset,
                                #  batch_size=self.args_data.batch_size,
                                        batch_sampler = test_batch_sampler,
                                        # shuffle=False,
                                        num_workers=self.args_data.num_workers,
                                        pin_memory=True,     
                                        persistent_workers=persistent_workers,)
                                                                                #  collate_fn=debug_collate_fn)
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
