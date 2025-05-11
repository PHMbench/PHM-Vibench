import pandas as pd
import numpy as np
import io
import re # For natural sorting

def update_xjtu_sy_metadata_v2(metadata_csv: str, output_csv_path: str) -> pd.DataFrame:
    """
    更新XJTU-SY数据集的元数据CSV文件。
    V2: Label为每个Bearing ID分配唯一整数标识。

    Args:
        metadata_csv_content (str): 原始元数据CSV文件
        output_csv_path (str): 更新后的元数据CSV文件的保存路径。

    Returns:
        pd.DataFrame: 更新后的元数据DataFrame。
    """
    try:
        metadata_df = pd.read_csv(metadata_csv)
        print(f"成功加载元数据，共 {len(metadata_df)} 条记录。")
    except Exception as e:
        print(f"读取元数据内容时出错: {e}")
        return pd.DataFrame()

    # 筛选 RM_002_XJTU 数据集
    # 使用 .loc 来避免 SettingWithCopyWarning
    xjtu_indices = metadata_df[metadata_df['Name'] == 'RM_002_XJTU'].index
    if xjtu_indices.empty:
        print("在元数据中未找到 RM_002_XJTU 数据集的记录。")
        return metadata_df
    
    print(f"找到 {len(xjtu_indices)} 条 RM_002_XJTU 数据集的记录进行更新。")

    # --- 根据PDF及用户要求定义固定值 ---
    sample_rate = 25600
    sample_length = 32768
    channels = 2

    # --- 提取文件信息 ---
    def extract_file_info(filepath_str):
        if pd.isna(filepath_str):
            return None, None, None, None
        parts = filepath_str.split('/')
        if len(parts) == 3: # e.g., "35Hz12kN/Bearing1_1/1.csv"
            condition_str = parts[0]
            bearing_id_str = parts[1]
            filename_str = parts[2]
            try:
                file_num_int = int(filename_str.split('.')[0])
                return condition_str, bearing_id_str, file_num_int, f"{condition_str}/{bearing_id_str}"
            except ValueError:
                return condition_str, bearing_id_str, None, f"{condition_str}/{bearing_id_str}"
        return None, None, None, None

    # 创建临时列用于处理，直接在原始 DataFrame 的相关子集上操作
    temp_df = metadata_df.loc[xjtu_indices, 'File'].apply(lambda x: pd.Series(extract_file_info(x), index=['condition', 'bearing_id_parsed', 'file_num', 'bearing_path']))
    
    for col in temp_df.columns:
        metadata_df.loc[xjtu_indices, col] = temp_df[col]


    # --- 创建新的 Bearing ID 到 Label 的映射 ---
    # 筛选出 RM_002_XJTU 的有效 bearing_id_parsed
    valid_bearing_ids_xjtu = metadata_df.loc[xjtu_indices, 'bearing_id_parsed'].dropna().unique()

    # 自然排序函数，确保 Bearing1_10 在 Bearing1_2 之后
    def natural_sort_key_bearing_id(bearing_id_str):
        # Extracts numbers from "BearingX_Y" -> (X, Y)
        match = re.match(r'Bearing(\d+)_(\d+)', bearing_id_str)
        if match:
            return int(match.group(1)), int(match.group(2))
        return (0, 0) # Fallback for unexpected formats

    sorted_unique_bearing_ids = sorted(list(valid_bearing_ids_xjtu), key=natural_sort_key_bearing_id)
    
    bearing_to_label_map = {name: i for i, name in enumerate(sorted_unique_bearing_ids)}
    print("\nBearing ID 到 Label 的映射:")
    for bearing_name, assigned_label in bearing_to_label_map.items():
        print(f"- {bearing_name}: {assigned_label}")


    # 计算每个轴承的总文件数 (N_total for RUL)
    # 使用在metadata_df上创建的临时列进行groupby
    valid_rul_data = metadata_df.loc[xjtu_indices].dropna(subset=['bearing_path', 'file_num'])
    rul_info = valid_rul_data.groupby('bearing_path')['file_num'].max().to_dict()


    # --- 更新元数据列 ---
    for index in xjtu_indices:
        metadata_df.loc[index, 'Sample_rate'] = sample_rate
        metadata_df.loc[index, 'Sample_lenth'] = sample_length
        metadata_df.loc[index, 'Channel'] = channels
        metadata_df.loc[index, 'Fault_level'] = np.nan # Per user request

        condition = metadata_df.loc[index, 'condition']
        bearing_id = metadata_df.loc[index, 'bearing_id_parsed']
        file_num = metadata_df.loc[index, 'file_num']
        bearing_path = metadata_df.loc[index, 'bearing_path']

        # 更新 Domain_id
        if pd.notna(condition):
            if '35Hz12kN' in condition:
                metadata_df.loc[index, 'Domain_id'] = 0
                metadata_df.loc[index, 'Domain_description'] = '35Hz12kN'
            elif '37.5Hz11kN' in condition:
                metadata_df.loc[index, 'Domain_id'] = 1
                metadata_df.loc[index, 'Domain_description'] = '37.5Hz11kN'
            elif '40Hz10kN' in condition:
                metadata_df.loc[index, 'Domain_id'] = 1 # As per user request
                metadata_df.loc[index, 'Domain_description'] = '40Hz10kN'
        
        # 更新 Label (新的逻辑)
        if pd.notna(bearing_id) and bearing_id in bearing_to_label_map:
            label_val = bearing_to_label_map[bearing_id]
            metadata_df.loc[index, 'Label'] = label_val
            # metadata_df.loc[index, 'Label_Description'] = f"Unique Bearing ID: {bearing_id} (Experiment Unit)"
            metadata_df.loc[index, 'Fault_Diagnosis'] = True # Still a classification target
        else:
            metadata_df.loc[index, 'Label'] = np.nan
            # metadata_df.loc[index, 'Label_Description'] = "Unknown or Unmapped Bearing ID"
            metadata_df.loc[index, 'Fault_Diagnosis'] = False


        # 更新 RUL_label
        if pd.notna(bearing_path) and bearing_path in rul_info and pd.notna(file_num):
            n_total = rul_info[bearing_path]
            if n_total > 1: # Avoid division by zero if n_total is 1
                # Ensure file_num and n_total are numeric for calculation
                current_f_num = pd.to_numeric(file_num, errors='coerce')
                total_f_num = pd.to_numeric(n_total, errors='coerce')
                if pd.notna(current_f_num) and pd.notna(total_f_num):
                    rul = 1 - (current_f_num - 1) / (total_f_num - 1)
                else:
                    rul = np.nan # if conversion failed
            elif n_total == 1: # Single file case
                rul = 0.0 
            else: # Should not happen if rul_info is correctly populated from max file_num
                rul = np.nan

            metadata_df.loc[index, 'RUL_label'] = rul
            # metadata_df.loc[index, 'RUL_label_description'] = 'Normalized RUL (1=start_of_life, 0=end_of_life)'
            metadata_df.loc[index, 'Remaining_Life'] = pd.notna(rul)
        else:
            metadata_df.loc[index, 'RUL_label'] = np.nan
            metadata_df.loc[index, 'Remaining_Life'] = False
            
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
    Root_path = '/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/data/'
    # 使用你提供的 metadata_19_5_4.csv 的实际内容
    example_metadata = Root_path + 'metadata_19_5_4.csv'
    output_file = Root_path + "RM_002_XJTU.csv"
    
    print("开始更新XJTU-SY元数据 (V2 - Unique Label per Bearing ID)...")
    updated_df = update_xjtu_sy_metadata_v2(example_metadata, output_file)

    if not updated_df.empty:
        print("\n更新后的XJTU-SY数据子集示例 (查看Label和Label_Description):")
        xjtu_subset_for_display = updated_df[updated_df['Name'] == 'RM_002_XJTU']
        
        # 为了更清晰地展示，我们选择每个bearing_id的第一条记录
        if not xjtu_subset_for_display.empty:
            # 需要先恢复bearing_id_parsed来进行展示，或者从File列提取
            def get_bearing_id_from_file(file_path_str):
                 if pd.isna(file_path_str): return None
                 return file_path_str.split('/')[1] if len(file_path_str.split('/')) > 1 else None
            
            display_df = xjtu_subset_for_display.copy() # 避免SettingWithCopyWarning
            display_df['temp_bearing_id'] = display_df['File'].apply(get_bearing_id_from_file)
            
            print(display_df.drop_duplicates(subset=['temp_bearing_id'])[['File', 'Label', 'Label_Description', 'RUL_label']].head(20)) # 显示前20个唯一轴承的信息
            del display_df['temp_bearing_id'] # 清理临时列
        else:
            print("未找到RM_002_XJTU的子集数据进行展示。")


        print("\n更新后的XJTU-SY数据子集Label分布:")
        print(updated_df[updated_df['Name'] == 'RM_002_XJTU']['Label'].value_counts(dropna=False).sort_index())
        
    print("\n脚本执行完毕。")