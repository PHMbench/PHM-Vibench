import pandas as pd
import numpy as np
import io
import re # For natural sorting
import os

# 定义各轴承的阶段划分阈值 (normal_end_percentage, degradation_end_percentage)
# 这些百分比是基于文件序号的累积百分比
BEARING_STAGE_THRESHOLDS = {
    'Bearing1_1': [0.60, 0.80],
    'Bearing1_2': [0.35, 0.80],
    'Bearing1_3': [0.35, 0.94],
    'Bearing1_4': [0.96, 0.98], # 注意这个衰退期很短
    'Bearing1_5': [0.60, 0.90],
    'Bearing2_1': [0.90, 0.95],
    'Bearing2_2': [0.50, 0.90],
    'Bearing2_3': [0.65, 0.88],
    'Bearing2_4': [0.72, 0.80],
    'Bearing2_5': [0.50, 0.60],
    'Bearing3_1': [0.90, 0.96],
    'Bearing3_2': [0.90, 0.96],
    'Bearing3_3': [0.90, 0.93],
    'Bearing3_4': [0.95, 0.96], # 注意这个衰退期也很短
    'Bearing3_5': [0.10, 0.60],
}


def update_xjtu_sy_metadata_v3_staging(metadata_csv_path: str, output_csv_path: str) -> pd.DataFrame:
    """
    更新XJTU-SY数据集的元数据CSV文件。
    V3_staging: 根据文件在时间序列中的百分比位置，为每个文件打上正常(0)、衰退(-1)或早期故障(bearing_id)的标签。
                同时更新任务相关性列。

    Args:
        metadata_csv_path (str): 原始元数据CSV文件的路径。
        output_csv_path (str): 更新后的元数据CSV文件的保存路径。

    Returns:
        pd.DataFrame: 更新后的元数据DataFrame。
    """
    try:
        metadata_df = pd.read_csv(metadata_csv_path)
        print(f"成功加载元数据，共 {len(metadata_df)} 条记录。")
    except Exception as e:
        print(f"读取元数据文件 '{metadata_csv_path}' 时出错: {e}")
        return pd.DataFrame()

    xjtu_indices = metadata_df[metadata_df['Name'] == 'RM_002_XJTU'].index
    if xjtu_indices.empty:
        print("在元数据中未找到 RM_002_XJTU 数据集的记录。")
        return metadata_df
    
    print(f"找到 {len(xjtu_indices)} 条 RM_002_XJTU 数据集的记录进行更新。")

    # --- 固定值 ---
    sample_rate = 25600
    sample_length = 32768 # XJTU 每个文件固定32768点
    channels = 2          # XJTU 水平和垂直两个通道

    # --- 文件信息提取函数 ---
    def extract_file_info(filepath_str):
        if pd.isna(filepath_str):
            return None, None, None, None
        parts = filepath_str.split('/')
        if len(parts) == 3: 
            condition_str = parts[0]
            bearing_id_str = parts[1]
            filename_str = parts[2]
            try:
                file_num_int = int(filename_str.split('.')[0])
                return condition_str, bearing_id_str, file_num_int, f"{condition_str}/{bearing_id_str}"
            except ValueError:
                return condition_str, bearing_id_str, None, f"{condition_str}/{bearing_id_str}"
        return None, None, None, None

    temp_df = metadata_df.loc[xjtu_indices, 'File'].apply(lambda x: pd.Series(extract_file_info(x), index=['condition', 'bearing_id_parsed', 'file_num', 'bearing_path']))
    
    for col in temp_df.columns:
        metadata_df.loc[xjtu_indices, col] = temp_df[col]

    # --- 创建 Bearing ID 到唯一整数ID的映射 (用于早期故障阶段的标签) ---
    valid_bearing_ids_xjtu = metadata_df.loc[xjtu_indices, 'bearing_id_parsed'].dropna().unique()

    def natural_sort_key_bearing_id(bearing_id_str):
        match = re.match(r'Bearing(\d+)_(\d+)', bearing_id_str)
        if match:
            return int(match.group(1)), int(match.group(2))
        return (0, 0) 

    sorted_unique_bearing_ids = sorted(list(valid_bearing_ids_xjtu), key=natural_sort_key_bearing_id)
    
    # 这是为每个轴承实验单元分配的唯一ID，从0开始 (Bearing1_1 -> 0, Bearing1_2 -> 1, ..., Bearing3_5 -> 14)
    bearing_to_unique_id_map = {name: i for i, name in enumerate(sorted_unique_bearing_ids)}
    print("\nBearing ID 到其唯一整数ID的映射 (用于早期故障标签):")
    for bearing_name, assigned_id in bearing_to_unique_id_map.items():
        print(f"- {bearing_name}: {assigned_id}")

    # --- 计算每个轴承的总文件数 (N_total for RUL 和 阶段划分) ---
    valid_rul_data = metadata_df.loc[xjtu_indices].dropna(subset=['bearing_path', 'file_num'])
    # rul_info现在存储的是每个bearing_path的最大文件号 (即 N_total)
    n_total_files_per_bearing = valid_rul_data.groupby('bearing_path')['file_num'].max().to_dict()

    # --- 更新元数据列 ---
    for index in xjtu_indices:
        metadata_df.loc[index, 'Sample_rate'] = sample_rate
        metadata_df.loc[index, 'Sample_lenth'] = sample_length # 对于XJTU是固定的
        metadata_df.loc[index, 'Channel'] = channels           # 对于XJTU是固定的
        # metadata_df.loc[index, 'Fault_level'] = np.nan # XJTU不直接提供此信息

        condition = metadata_df.loc[index, 'condition']
        bearing_id = metadata_df.loc[index, 'bearing_id_parsed']
        file_num = pd.to_numeric(metadata_df.loc[index, 'file_num'], errors='coerce')
        bearing_path = metadata_df.loc[index, 'bearing_path']

        # --- Domain_id ---
        if pd.notna(condition):
            if '35Hz12kN' in condition:
                metadata_df.loc[index, 'Domain_id'] = 0
                metadata_df.loc[index, 'Domain_description'] = '35Hz12kN'
            elif '37.5Hz11kN' in condition:
                metadata_df.loc[index, 'Domain_id'] = 1
                metadata_df.loc[index, 'Domain_description'] = '37.5Hz11kN'
            elif '40Hz10kN' in condition:
                metadata_df.loc[index, 'Domain_id'] = 1 
                metadata_df.loc[index, 'Domain_description'] = '40Hz10kN'
        
        # --- Label, RUL_label, 和任务相关性 ---
        current_label = np.nan
        rul_label_val = np.nan
        fault_diag_task = False
        anomaly_det_task = False
        remaining_life_task = False
        digital_twin_task = False

        if pd.notna(bearing_id) and pd.notna(bearing_path) and pd.notna(file_num) and \
           bearing_path in n_total_files_per_bearing and bearing_id in BEARING_STAGE_THRESHOLDS:
            
            n_total = n_total_files_per_bearing[bearing_path]
            thresholds = BEARING_STAGE_THRESHOLDS[bearing_id]
            normal_end_perc = thresholds[0]
            degradation_end_perc = thresholds[1]

            # 计算当前文件序号在总文件数中的百分比位置 (从0到1)
            # file_num 是从1开始的
            current_file_percentage = (file_num -1) / n_total if n_total > 0 else 0
            
            # RUL_label 计算 (与阶段划分无关，但可以一起算)
            if n_total > 1:
                rul_label_val = 1 - (file_num - 1) / (n_total - 1)
            elif n_total == 1:
                rul_label_val = 0.0
            
            # 阶段判断和标签分配
            if current_file_percentage <= normal_end_perc: # 正常阶段
                current_label = 0
                # metadata_df.loc[index, 'Label_Description'] = f"{bearing_id} - Normal Stage"
                fault_diag_task = True
                anomaly_det_task = True
                remaining_life_task = True # 整个过程都可用于RUL
                digital_twin_task = True
            elif current_file_percentage <= degradation_end_perc: # 衰退阶段
                current_label = -1
                # metadata_df.loc[index, 'Label_Description'] = f"{bearing_id} - Degradation Stage"
                fault_diag_task = False # 不可执行故障诊断
                anomaly_det_task = True
                remaining_life_task = True
                digital_twin_task = True
            else: # 早期故障阶段
                current_label = bearing_to_unique_id_map.get(bearing_id, np.nan) + 1 # 使用轴承的唯一整数ID
                # metadata_df.loc[index, 'Label_Description'] = f"{bearing_id} - Early Fault Stage (ID: {current_label})"
                fault_diag_task = True
                anomaly_det_task = True
                remaining_life_task = True
                digital_twin_task = True
        else:
            if pd.notna(bearing_id) and bearing_id not in BEARING_STAGE_THRESHOLDS:
                print(f"警告: 轴承ID '{bearing_id}' 的阶段阈值未在 BEARING_STAGE_THRESHOLDS 中定义。文件 '{metadata_df.loc[index, 'File']}' 的标签将为NaN。")
            # metadata_df.loc[index, 'Label_Description'] = "Unknown or unmappable stage"


        metadata_df.loc[index, 'Label'] = current_label
        metadata_df.loc[index, 'RUL_label'] = rul_label_val
        # metadata_df.loc[index, 'RUL_label_description'] = np.nan # Per previous request to keep empty

        metadata_df.loc[index, 'Fault_Diagnosis'] = fault_diag_task
        metadata_df.loc[index, 'Anomaly_Detection'] = anomaly_det_task
        metadata_df.loc[index, 'Remaining_Life'] = remaining_life_task
        metadata_df.loc[index, 'Digital_Twin_Prediction'] = digital_twin_task
            
    # 移除辅助列
    cols_to_drop = ['condition', 'bearing_id_parsed', 'file_num', 'bearing_path']
    metadata_df.drop(columns=[col for col in cols_to_drop if col in metadata_df.columns], inplace=True, errors='ignore')

    # 保存更新后的元数据
    try:
        metadata_df.to_csv(output_csv_path, index=False)
        print(f"\n更新后的元数据已保存到: {output_csv_path}")
    except Exception as e:
        print(f"保存更新后的元数据时出错: {e}")
        
    return metadata_df

if __name__ == '__main__':
    # 确保 Root_path 指向包含 'metadata_19_5_4.csv' 的目录
    Root_path = '/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/data/' # 请根据你的实际路径修改
    example_metadata_file_path = os.path.join(Root_path, 'metadata_19_5_4.csv')
    output_file_path = os.path.join(Root_path, "RM_002_XJTU_staged_labels.csv")
    
    if not os.path.exists(example_metadata_file_path):
        print(f"错误: 元数据文件未找到于 '{example_metadata_file_path}'")
        print("请确保路径正确，并且文件存在。")
    else:
        print(f"开始更新XJTU-SY元数据 (V3_staging - 带阶段划分的标签)...")
        updated_df = update_xjtu_sy_metadata_v3_staging(example_metadata_file_path, output_file_path)

        if not updated_df.empty:
            print("\n更新后的XJTU-SY数据子集示例:")
            xjtu_subset_for_display = updated_df[updated_df['Name'] == 'RM_002_XJTU']
            
            if not xjtu_subset_for_display.empty:
                # 为了更清晰地展示，我们选择每个bearing_id的几个代表性文件（初期、中期、末期）
                # 这里简化，只显示每个bearing_id的前几个和后几个文件
                
                # 需要临时恢复bearing_id用于分组显示
                def get_bearing_id_from_file_for_display(filepath_str_series, condition_series):
                    results = []
                    for filepath_str, cond_str in zip(filepath_str_series, condition_series):
                        if pd.isna(filepath_str):
                            results.append(None)
                            continue
                        parts = filepath_str.split('/')
                        if len(parts) == 3:
                            results.append(parts[1]) # bearing_id
                        else:
                            results.append(None)
                    return results

                display_df = xjtu_subset_for_display.copy()
                # File列现在是原始的，所以可以直接用
                display_df['temp_bearing_id_for_display'] = display_df.apply(
                    lambda row: row['File'].split('/')[1] if pd.notna(row['File']) and len(row['File'].split('/')) > 1 else None, axis=1
                )

                print("\n--- 示例：Bearing3_5 的标签变化 ---")
                b3_5_sample = display_df[display_df['temp_bearing_id_for_display'] == 'Bearing3_5']
                if not b3_5_sample.empty:
                    # 为了看到阶段变化，我们需要排序并查看不同百分位的文件
                    # 获取文件序号用于排序和百分比计算
                    b3_5_sample['file_num_temp'] = b3_5_sample['File'].apply(lambda x: int(x.split('/')[-1].split('.')[0]) if pd.notna(x) else 0)
                    b3_5_sample = b3_5_sample.sort_values(by='file_num_temp')
                    n_total_b3_5 = len(b3_5_sample)
                    
                    # 选取文件来展示阶段
                    indices_to_show = []
                    if n_total_b3_5 > 0:
                        indices_to_show.append(b3_5_sample.index[0]) # 第一个
                        # 正常结束点附近 (10%)
                        idx_norm_end = int(n_total_b3_5 * BEARING_STAGE_THRESHOLDS['Bearing3_5'][0])
                        if idx_norm_end < n_total_b3_5 : indices_to_show.append(b3_5_sample.index[idx_norm_end])
                        if idx_norm_end + 1 < n_total_b3_5 : indices_to_show.append(b3_5_sample.index[idx_norm_end+1])
                        # 衰退结束点附近 (60%)
                        idx_degr_end = int(n_total_b3_5 * BEARING_STAGE_THRESHOLDS['Bearing3_5'][1])
                        if idx_degr_end < n_total_b3_5 : indices_to_show.append(b3_5_sample.index[idx_degr_end])
                        if idx_degr_end + 1 < n_total_b3_5 : indices_to_show.append(b3_5_sample.index[idx_degr_end+1])
                        indices_to_show.append(b3_5_sample.index[-1]) # 最后一个
                    
                    # 去重并保持顺序
                    final_indices_to_show = sorted(list(set(indices_to_show)))
                    print(display_df.loc[final_indices_to_show, ['File', 'Label', 'Label_Description', 'RUL_label', 'Fault_Diagnosis', 'Anomaly_Detection', 'Remaining_Life']])
                    
                del display_df['temp_bearing_id_for_display']
                if 'file_num_temp' in b3_5_sample.columns: del b3_5_sample['file_num_temp']

            else:
                print("未找到RM_002_XJTU的子集数据进行展示。")

            print("\n更新后的XJTU-SY数据子集Label分布:")
            print(xjtu_subset_for_display['Label'].value_counts(dropna=False).sort_index())
        
    print("\n脚本执行完毕。")