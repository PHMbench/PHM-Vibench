# PHMBench/Vbench 数据贡献指南 （AI生成）

本文档详细规定了数据集的标准处理流程，确保所有贡献者遵循统一规范。
内测阶段只需完成[3. 数据标准化](#3-数据标准化)部分，后续由@liq22负责数据集注册和数据加载器实现。

## 1. 任务分配与沟通

- **任务分发**：所有数据集任务通过群组聊天进行分配和协调，参考在线共享文档。
- **进度报告**：在群内定期提供进度更新。
- **问题解决**：遇到技术问题请立即在群内提出，确保及时解决。
- **决策过程**：重大决策需在群内讨论并达成共识后执行。

## 2. 数据下载

- **目录结构**：将原始数据集组织到以下目录：
    ```
    ./PHMBench/Vbench/data/raw/[dataset_name]/
    ```
- **文件组织**：保持原始数据集的文件结构不变
- **来源记录**：记录数据集来源、版本及下载日期，便于追溯

## 3. 数据标准化

### 3.1 格式提取

- 从原始数据提取为标准长度(L)和通道数(C)格式的numpy数组
- 确保数据类型一致性，推荐使用float32类型

### 3.2 命名规范

- **必须严格遵循**以下命名约定：
    ```
    dataset_modality_condition1-condition2-condition3-..._other.npy
    ```
- **条件顺序**规定：
    1. 速度(s)：如 s1970rpm
    2. 负载(l)：如 l1kn
    3. 故障程度：如 3mm
- **文件名示例**：`CWRU_vib_s1970rpm_l1kn_3mm.npy`
- 可根据需要创建分级目录，但最终文件命名必须符合上述规范

### 3.3 元数据生成@liq22 给出模板

创建`metadata.csv`文件，包含以下列：

| 字段名 | 说明 | 示例值 |
|--------|------|------|
| original_shape_L | 原始数据长度 | 4096 |
| original_shape_C | 原始数据通道数 | 1 |
| processed_indices | 后处理数据标号 | 0,1,2,3 |
| is_labeled | 是否有标签 | True/False |
| file_name | 对应的npy文件名 | 'CWRU_vib_s1970rpm_l1kn_3mm.npy' |
| is_normal | 是否为正常样本 | True |
| is_valid | 数据是否有效 | True |
| has_metadata | 是否包含额外元数据 | True |
| source_file | 原始源文件名 | 'original_file.mat' |
| dataset_name | 数据集名称 | 'CWRU', 'THU' |
| class_id | 故障类别ID | 1,2,3 |

- 每行对应一个标准化的npy文件
- CSV文件应与npy文件保存在同一目录下
- 所有标准化的npy文件及metadata.csv将上传至Hugging Face

## 4. 数据结构

创建数据读取类，该类应：

- 加载并解析标准化的`.npy`文件和`metadata.csv`
- 组织成以下字典结构：
    ```python
    {
            'data': numpy_array,  # 形状为(L, C)的数组
            'condition_speed': speed_value,  # 从文件名解析
            'condition_load': load_value,    # 从文件名解析
            'fault_size': fault_size,        # 从文件名解析
            'dataset_name': dataset_name,    # 从metadata获取
            'class_id': class_id,            # 从metadata获取
            # 其他相关元数据
    }
    ```
- 根据pipeline需求，将处理好的数据保存为`pipeline_name_cache.hdf5`

## 5. 预处理

实现预处理函数，这些函数应：

- 对numpy数组应用归一化（Min-Max或Z-score标准化）
- 处理缺失值和异常值
- 应用领域特定信号处理变换（FFT、小波变换等）
- 记录所有预处理步骤，确保可重现性
- 建立缓存机制：`cache_path = f"cache/{dataset_name}_{pipeline_name}.json"`

## 6. 任务类实现

通过继承PyTorch的Dataset创建数据集类：

```python
class YourDataset(torch.utils.data.Dataset):
        def __init__(self, standardized_data, **config):
                # 使用标准化数据初始化
                # 根据参数配置数据分割方式
                # 将数据转换为B, L, C格式用于分类任务
                
        def transform_to_BLC(self):
                # 实现L,C到B,L,C的转换
                
        def __getitem__(self, index):
                # 返回单个数据样本及其标签
                
        def __len__(self):
                # 返回数据集大小
```

## 7. 数据集注册

在DataFactory中注册您的数据集：

```python
# 在datafactory.py中
from .your_dataset import YourDataset

def register_datasets():
        DATA_REGISTRY["your_dataset_name"] = YourDataset
```

## 8. DataLoader封装

为您的数据集创建DataLoader实例：

```python
dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
)
```

## 9. 多DataLoader实现

如有需要，为模型训练实现MultipleDataLoader包装器：

```python
class MultipleDataLoader:
        def __init__(self, dataloaders):
                self.dataloaders = dataloaders
                # 额外配置
                
        def __iter__(self):
                # 自定义迭代逻辑，如轮流从不同数据加载器取批次
```

## 质量检查清单

- [ ] 数据文件命名符合规范
- [ ] metadata.csv包含所有必需字段且格式正确
- [ ] 预处理步骤有详细文档记录
- [ ] 数据集类正确实现所有必需方法
- [ ] 数据集在DataFactory中正确注册
- [ ] 测试确认数据能够正确加载和预处理

# updatecode

## 007
```python
import pandas as pd
df0 = pd.read_csv(r'/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/data/metadata_25_5_25.csv')

# 根据列筛选行: 'Name'
df = df0[df0['Name'].str.contains("007", regex=False, na=False, case=False)]

# Add Label column with specified values
df['Label'] = [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3]
# Add Domain_id column with specified values
df['Domain_id'] = [0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7, 9, 9, 9]
# Add Sample_rate column with specified values
df['Sample_rate'] = [97656, 97656, 97656, 97656, 97656, 97656, 48828, 48828, 48828, 48828, 48828, 48828, 48828, 48828, 48828, 48828, 48828, 48828, 48828, 48828, 48828, 48828, 48828]

# Fill missing values for specific columns
df['Fault_Diagnosis'] = df['Fault_Diagnosis'].fillna(1)
df['Anomaly_Detection'] = df['Anomaly_Detection'].fillna(1)
df['Remaining_Life'] = df['Remaining_Life'].fillna(0)
df['Digital_Twin_Prediction'] = df['Digital_Twin_Prediction'].fillna(1)

df0.loc[df.index, df.columns] = df

df = df0

# 根据列筛选行: 'Name'

```

## 008
```python
import pandas as pd
import numpy as np
import re # For regular expressions

# --- 前置条件：假设 df 已经是一个从CSV加载的Pandas DataFrame ---
# 例如，您应该有类似这样的代码来加载它：
# csv_file_path = 'YOUR_ACTUAL_CSV_FILE.csv' # <--- 请务必替换为您的CSV文件路径
# try:
#     df = pd.read_csv(csv_file_path)
# except FileNotFoundError:
#     print(f"错误: 文件 '{csv_file_path}' 未找到。")
#     exit()
# except Exception as e:
#     print(f"读取CSV文件时出错: {e}")
#     exit()
# print("CSV文件加载成功，原始DataFrame 'df' 的形状:", df.shape)
# print(df.head()) # 打印头部以确认加载正确
# --- 前置条件结束 ---


# --- 开始直接修改 'df' ---

# 筛选 Name 为 ‘RM_008_UNSW’ 的条目的 *索引*
# 我们将使用这些索引来定位 df 中的行进行修改
rm008_indices = df[df['Name'] == 'RM_008_UNSW'].index

if rm008_indices.empty:
    print("未找到 Name 为 'RM_008_UNSW' 的条目。将不进行任何修改。")
else:
    print(f"找到 {len(rm008_indices)} 条 Name 为 'RM_008_UNSW' 的记录进行处理。")

    # 定义辅助函数 (可以放在脚本的更前面或此处)
    def extract_test_label(description):
        if pd.isna(description):
            return pd.NA # 使用Pandas的NA以保持Int64类型
        match = re.search(r"Test\s*(\d+)", str(description), re.IGNORECASE)
        if match:
            try:
                return int(match.group(1)) -1
            except ValueError:
                return pd.NA
        return pd.NA

    def extract_domain_id(description, speed_map):
        if pd.isna(description):
            return pd.NA
        match = re.search(r"_(\d{2})\.mat$", str(description))
        if match:
            speed_str = match.group(1)
            return speed_map.get(speed_str, pd.NA)
        return pd.NA

    def extract_running_cycle(description):
        if pd.isna(description):
            return -1 # 用一个特殊值处理NaN，确保可排序且在数值比较时有意义
        match = re.search(r"vib_(\d+)_", str(description))
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return -1 # 如果数字转换失败
        return -1 # 如果没有匹配

    def get_rul_group_key(description):
        if pd.isna(description):
            return "unknown_group" # 未知组
        parts = str(description).split('/')
        if len(parts) > 1: # 例如: 'Test 1/Multiple speeds/vib_...'
            return "/".join(parts[:-1]) # 取最后一个'/'之前的部分作为组名
        return str(description) # 如果没有'/'，则整个Description作为组名 (可能需要调整)

    # --- 步骤 3: Label ---
    if 'Label' not in df.columns:
        df['Label'] = pd.Series(dtype='Int64') # 使用可空的整数类型
    df.loc[rm008_indices, 'Label'] = df.loc[rm008_indices, 'File'].apply(extract_test_label)
    print("步骤 3: 'Label' 列处理完毕。")

    # --- 步骤 4: Domain_id ---
    speed_to_domain_map = {'20': 3, '15': 2, '12': 1, '06': 0}
    if 'Domain_id' not in df.columns:
        df['Domain_id'] = pd.Series(dtype='Int64')
    df.loc[rm008_indices, 'Domain_id'] = df.loc[rm008_indices, 'File'].apply(
        lambda desc: extract_domain_id(desc, speed_to_domain_map)
    )
    print("步骤 4: 'Domain_id' 列处理完毕。")

    # --- 步骤 5: Sample_rate ---
    if 'Sample_rate' not in df.columns:
        df['Sample_rate'] = pd.Series(dtype='Int64')
    df.loc[rm008_indices, 'Sample_rate'] = 51200
    print("步骤 5: 'Sample_rate' 列处理完毕。")

    # --- 步骤 6: Sample_lenth ---
    # (如果实际列名是 Sample_length，请在下面和列定义中一并修改)
    if 'Sample_lenth' not in df.columns:
        df['Sample_lenth'] = pd.Series(dtype='Int64')
    df.loc[rm008_indices, 'Sample_lenth'] = 614400
    print("步骤 6: 'Sample_lenth' 列处理完毕。")

    # --- 步骤 7: Channel ---
    if 'Channel' not in df.columns:
        df['Channel'] = pd.Series(dtype='Int64')
    df.loc[rm008_indices, 'Channel'] = 6
    print("步骤 7: 'Channel' 列处理完毕。")

    # --- 步骤 8: 任务相关列 ---
    task_columns = ['Fault_Diagnosis', 'Anomaly_Detection', 'Remaining_Life', 'Digital_Twin_Prediction']
    # 根据您的规则，Remaining_Life 的默认值也是1
    default_task_values = {'Fault_Diagnosis': 1, 'Anomaly_Detection': 1, 'Remaining_Life': 1, 'Digital_Twin_Prediction': 1}

    for col, default_val in default_task_values.items():
        if col not in df.columns:
            df[col] = pd.Series(dtype='Int64') # 假设这些也是整数标志
        df.loc[rm008_indices, col] = default_val
    print("步骤 8: 任务相关列处理完毕。")

    # --- 步骤 9: RUL_label ---
    if 'RUL_label' not in df.columns:
        df['RUL_label'] = pd.Series(dtype='float64') # RUL通常是浮点数

    # 为了进行分组和排序，我们从筛选出的行（rm008_indices）创建一个临时视图或副本
    # 并添加临时列。这些临时列不会直接加到原始df上。
    temp_df_for_rul_calc = df.loc[rm008_indices].copy() # 使用.copy()确保是副本
    temp_df_for_rul_calc['Running_Cycle_Sort_Key'] = temp_df_for_rul_calc['File'].apply(extract_running_cycle)
    temp_df_for_rul_calc['RUL_Group_Key'] = temp_df_for_rul_calc['File'].apply(get_rul_group_key)

    # 按 RUL_Group_Key 分组，并在每个组内排序和计算RUL
    print("开始计算 RUL_label...")
    for group_name, group_data_from_temp in temp_df_for_rul_calc.groupby('RUL_Group_Key'):
        # group_data_from_temp 是 temp_df_for_rul_calc 中的一个子DataFrame
        # 其索引是原始 df 中的索引

        # 对当前分组内的数据按照运行周期 (Running_Cycle_Sort_Key) 从小到大排序
        sorted_group_data = group_data_from_temp.sort_values(by='Running_Cycle_Sort_Key', ascending=True)
        
        num_files_in_group = len(sorted_group_data)
        
        if num_files_in_group > 0:
            # 生成从1到0线性递减的RUL标签
            rul_values = np.linspace(1, 0, num_files_in_group) if num_files_in_group > 1 else [1.0]
            
            # 使用 sorted_group_data 中的原始索引 (它们来自 df.loc[rm008_indices])
            # 将计算出的 RUL 值赋回原始 DataFrame 'df' 的 'RUL_label' 列
            df.loc[sorted_group_data.index, 'RUL_label'] = rul_values
            # print(f"  为组 '{group_name}' ({num_files_in_group}个文件) 分配了RUL标签。") # 调试信息
        else:
            print(f"警告: 分组 '{group_name}' 中没有文件，无法计算RUL标签。")
    print("步骤 9: 'RUL_label' 列处理完毕。")

print("\n--- DataFrame 'df' 修改完成 ---")

# --- 可选：验证和打印修改后的结果 ---
# print("\n--- 修改后 'RM_008_UNSW' 条目的部分列进行验证 ---")
# df_check_rm008 = df[df['Name'] == 'RM_008_UNSW'].copy()
# if not df_check_rm008.empty:
#     # 为了验证排序，我们可以添加临时的group和cycle key进行显示
#     df_check_rm008['temp_group_key_check'] = df_check_rm008['Description'].apply(get_rul_group_key)
#     df_check_rm008['temp_cycle_key_check'] = df_check_rm008['Description'].apply(extract_running_cycle)
#     df_check_rm008_sorted = df_check_rm008.sort_values(by=['temp_group_key_check', 'temp_cycle_key_check'])
    
#     columns_to_display = [
#         'Name', 'Description', 'Label', 'Domain_id',
#         'Sample_rate', 'Sample_lenth', 'Channel',
#         'Fault_Diagnosis', 'Anomaly_Detection', 'Remaining_Life', 
#         'Digital_Twin_Prediction', 'RUL_label',
#         # 'temp_group_key_check', 'temp_cycle_key_check' # 可选，用于验证分组和排序键
#     ]
#     # 确保所有要显示的列都存在
#     columns_to_display = [col for col in columns_to_display if col in df_check_rm008_sorted.columns]

#     print(df_check_rm008_sorted[columns_to_display].to_string())
# else:
#     print("在最终的df中未找到 'RM_008_UNSW' 数据进行验证。")

# --- 可选：保存修改后的 DataFrame ---
# output_csv_path = 'processed_phm_data.csv'
# try:
#     df.to_csv(output_csv_path, index=False)
#     print(f"\n已将修改后的 DataFrame 保存到: {output_csv_path}")
# except Exception as e:
#     print(f"\n保存 DataFrame 到CSV时出错: {e}")
```