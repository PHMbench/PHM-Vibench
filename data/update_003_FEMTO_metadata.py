import pandas as pd
import numpy as np
import os
import re # For natural sorting

# --- 全局常量和配置 ---
BEARING_STAGE_THRESHOLDS_FEMTO = {
    'Bearing1_1': [0, 0.50, 0.78], 'Bearing1_2': [0, 0.66, 0.94],
    'Bearing1_3': [0, 0.67, 0.90], 'Bearing1_4': [0, 0.72, 0.80],
    'Bearing1_5': [0, 0.27, 0.80], 'Bearing1_6': [0, 0.27, 0.64],
    'Bearing1_7': [0, 0.42, 0.90], 'Bearing2_1': [0, 0.15, 0.60],
    'Bearing2_2': [0, 0.23, 0.90], 'Bearing2_3': [0, 0.60, 0.98],
    'Bearing2_4': [0, 0.40, 0.96], 'Bearing2_5': [0, 0.90, 0.96],
    'Bearing2_6': [0, 0.25, 0.80], 'Bearing2_7': [0, 0.60, 0.90],
    'Bearing3_1': [0, 0.16, 0.80], 'Bearing3_2': [0, 0.71, 0.80],
    'Bearing3_3': [0, 0.62, 0.96],
}
SAMPLE_RATE_FEMTO = 25600
FIXED_SAMPLE_LENTH_FEMTO = 2560
FIXED_CHANNELS_FEMTO = 2

# --- 辅助函数 ---
def extract_femto_file_info(filepath_str: str):
    if pd.isna(filepath_str):
        return None, None, None, None, None
    try:
        parts = filepath_str.split('/')
        if len(parts) == 3:
            set_type = parts[0]
            bearing_id_parsed = parts[1]
            filename_str = parts[2] # e.g., acc_00001.csv or temp_00001.csv
            match_num = re.search(r'(\d+)\.csv$', filename_str)
            if match_num:
                file_num_int = int(match_num.group(1))
            else:
                file_num_int = None
            bearing_run_id = f"{set_type}/{bearing_id_parsed}"
            return set_type, bearing_id_parsed, file_num_int, bearing_run_id, filename_str
    except Exception:
        return None, None, None, None, None
    return None, None, None, None, None

def natural_sort_key_bearing_id(bearing_id_str: str):
    if pd.isna(bearing_id_str): return (0, 0)
    match = re.match(r'Bearing(\d+)_(\d+)', bearing_id_str)
    if match: return int(match.group(1)), int(match.group(2))
    return (0, 0)

# --- 主更新函数 ---
def update_femto_pronostia_metadata(
    metadata_csv_path: str,
    data_files_root_path: str,
    output_csv_path: str
) -> pd.DataFrame:
    try:
        metadata_df = pd.read_csv(metadata_csv_path)
        print(f"成功加载元数据，共 {len(metadata_df)} 条记录。")
    except Exception as e:
        print(f"读取元数据文件 '{metadata_csv_path}' 时出错: {e}")
        return pd.DataFrame()

    femto_indices = metadata_df[metadata_df['Name'] == 'RM_003_FEMTO'].index
    if femto_indices.empty:
        print("在元数据中未找到 RM_003_FEMTO 数据集的记录。")
        return metadata_df
    print(f"找到 {len(femto_indices)} 条 RM_003_FEMTO 数据集的记录进行更新。")

    # 提取信息，包括 filename
    temp_info_cols = ['set_type', 'bearing_id_parsed', 'file_num', 'bearing_run_id', 'filename']
    temp_df_femto = metadata_df.loc[femto_indices, 'File'].apply(
        lambda x: pd.Series(extract_femto_file_info(x), index=temp_info_cols)
    )
    for col in temp_info_cols:
        metadata_df.loc[femto_indices, col] = temp_df_femto[col]

    valid_bearing_ids_femto = metadata_df.loc[femto_indices, 'bearing_id_parsed'].dropna().unique()
    sorted_unique_bearing_ids_femto = sorted(list(valid_bearing_ids_femto), key=natural_sort_key_bearing_id)
    bearing_to_unique_id_map_femto = {name: i for i, name in enumerate(sorted_unique_bearing_ids_femto)}
    print("\nFEMTO Bearing ID 到其唯一整数ID的映射 (用于故障标签):")
    for bn, aid in bearing_to_unique_id_map_femto.items(): print(f"- {bn}: {aid}")

    metadata_df['file_num'] = pd.to_numeric(metadata_df['file_num'], errors='coerce')
    valid_rul_data_femto = metadata_df.loc[femto_indices].dropna(subset=['bearing_run_id', 'file_num'])
    max_file_num_per_bearing_run = valid_rul_data_femto.groupby('bearing_run_id')['file_num'].max().to_dict()

    for index in femto_indices:
        bearing_id = metadata_df.loc[index, 'bearing_id_parsed']
        file_num = metadata_df.loc[index, 'file_num']
        bearing_run_id = metadata_df.loc[index, 'bearing_run_id']
        current_filename = metadata_df.loc[index, 'filename']

        # 填充通用元数据 (对所有 FEMTO 文件)
        metadata_df.loc[index, 'Sample_rate'] = SAMPLE_RATE_FEMTO
        metadata_df.loc[index, 'Sample_lenth'] = FIXED_SAMPLE_LENTH_FEMTO
        metadata_df.loc[index, 'Channel'] = FIXED_CHANNELS_FEMTO
        metadata_df.loc[index, 'Fault_level'] = np.nan

        domain_id_val, domain_desc_val = np.nan, np.nan
        if pd.notna(bearing_id):
            if bearing_id.startswith('Bearing1_'): domain_id_val, domain_desc_val = 0, '1800rpm_4000N'
            elif bearing_id.startswith('Bearing2_'): domain_id_val, domain_desc_val = 1, '1650rpm_4200N'
            elif bearing_id.startswith('Bearing3_'): domain_id_val, domain_desc_val = 2, '1500rpm_5000N'
        metadata_df.loc[index, 'Domain_id'] = domain_id_val
        metadata_df.loc[index, 'Domain_description'] = domain_desc_val

        # 初始化特定于信号的标签和任务 (默认为NaN/False)
        current_label_val, rul_label_val = np.nan, np.nan
        fault_diag_task, anomaly_det_task, remaining_life_task, digital_twin_task = False, False, False, False

        # 仅为加速度计数据 ("acc_") 计算标签和特定任务
        if pd.notna(current_filename) and current_filename.startswith('acc_'):
            anomaly_det_task = True # 加速度信号可用于异常检测
            remaining_life_task = True # 加速度信号可用于RUL预测
            digital_twin_task = True # 加速度信号可用于数字孪生

            if pd.notna(bearing_id) and pd.notna(file_num) and pd.notna(bearing_run_id) and \
               bearing_run_id in max_file_num_per_bearing_run and \
               bearing_id in BEARING_STAGE_THRESHOLDS_FEMTO:

                max_fn = max_file_num_per_bearing_run[bearing_run_id]
                thresholds = BEARING_STAGE_THRESHOLDS_FEMTO[bearing_id]
                normal_end_perc, degradation_end_perc = thresholds[1], thresholds[2]

                if max_fn > 1:
                    rul_label_val = max(0.0, min(1.0, 1 - (file_num - 1) / (max_fn - 1)))
                elif max_fn == 1: rul_label_val = 0.0
                
                current_file_normalized_index = file_num / max_fn if max_fn > 0 else 0

                if current_file_normalized_index < normal_end_perc: # Normal
                    current_label_val = 0
                    fault_diag_task = True
                elif current_file_normalized_index < degradation_end_perc: # Degradation
                    current_label_val = -1
                    fault_diag_task = False # 按要求
                else: # Fault
                    unique_id = bearing_to_unique_id_map_femto.get(bearing_id, -1000) # Default if somehow not mapped
                    current_label_val = unique_id + 1
                    fault_diag_task = True
        
        metadata_df.loc[index, 'Label'] = current_label_val
        metadata_df.loc[index, 'RUL_label'] = rul_label_val
        metadata_df.loc[index, 'Fault_Diagnosis'] = fault_diag_task
        metadata_df.loc[index, 'Anomaly_Detection'] = anomaly_det_task
        metadata_df.loc[index, 'Remaining_Life'] = remaining_life_task
        metadata_df.loc[index, 'Digital_Twin_Prediction'] = digital_twin_task

        if 'Label_Description' in metadata_df.columns:
            metadata_df.loc[index, 'Label_Description'] = np.nan # 确保移除

    cols_to_drop_femto = ['set_type', 'bearing_id_parsed', 'file_num', 'bearing_run_id', 'filename']
    metadata_df.drop(columns=[col for col in cols_to_drop_femto if col in metadata_df.columns], inplace=True, errors='ignore')
    if 'Label_Description' in metadata_df.columns and metadata_df['Label_Description'].isnull().all():
        metadata_df.drop(columns=['Label_Description'], inplace=True, errors='ignore')

    try:
        metadata_df.to_csv(output_csv_path, index=False)
        print(f"\n更新后的元数据已保存到: {output_csv_path}")
    except Exception as e: print(f"保存更新后的元数据时出错: {e}")
    return metadata_df

# --- 主程序入口 ---
if __name__ == '__main__':
    ALL_DATASETS_ROOT_PATH = "/home/user/data/PHMbenchdata/"
    # 请确保 INPUT_METADATA_CSV_PATH 和 OUTPUT_METADATA_CSV_PATH 指向正确的文件
    INPUT_METADATA_CSV_PATH = "/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/data/metadata_19_5_4.csv" 
    OUTPUT_METADATA_CSV_PATH = "/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/data/RM_003_FEMTO_updated_metadata_v3.csv"

    print(f"所有数据集的根目录: {ALL_DATASETS_ROOT_PATH}")
    print(f"输入元数据文件: {INPUT_METADATA_CSV_PATH}")
    print(f"输出元数据文件: {OUTPUT_METADATA_CSV_PATH}")

    femto_data_path_check = os.path.join(ALL_DATASETS_ROOT_PATH, 'RM_003_FEMTO')
    if not os.path.exists(INPUT_METADATA_CSV_PATH):
        print(f"错误: 元数据文件未找到于 '{INPUT_METADATA_CSV_PATH}'")
    elif not os.path.exists(femto_data_path_check):
        print(f"错误: RM_003_FEMTO 数据集文件夹未找到于 '{femto_data_path_check}'")
    else:
        print(f"\n开始更新 FEMTO PRONOSTIA (RM_003_FEMTO) 元数据 (V3 - 仅处理acc文件)...")
        updated_df_femto = update_femto_pronostia_metadata(
            metadata_csv_path=INPUT_METADATA_CSV_PATH,
            data_files_root_path=ALL_DATASETS_ROOT_PATH,
            output_csv_path=OUTPUT_METADATA_CSV_PATH
        )

        if not updated_df_femto.empty:
            print("\n--- 更新后的 FEMTO (RM_003_FEMTO) 数据子集概览 (V3) ---")
            femto_subset_for_display = updated_df_femto[updated_df_femto['Name'] == 'RM_003_FEMTO'].copy()

            if not femto_subset_for_display.empty:
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', 1000)
                
                # 为了展示，临时再次提取文件名信息
                temp_info_display = femto_subset_for_display['File'].apply(
                     lambda x: pd.Series(extract_femto_file_info(x), index=['set_type', 'temp_bearing_id', 'temp_file_num', 'temp_bearing_run_id', 'temp_filename'])
                )
                femto_subset_for_display['temp_filename'] = temp_info_display['temp_filename']
                femto_subset_for_display['temp_bearing_id'] = temp_info_display['temp_bearing_id'] # for easy filtering
                femto_subset_for_display['temp_file_num'] = pd.to_numeric(temp_info_display['temp_file_num'])


                print("\n示例：Bearing1_3 的加速度文件 (acc_) vs 可能存在的其他文件类型")
                # 选择一个同时包含 acc_ 和其他类型文件（如果元数据中有）的轴承进行比较
                # 这里我们主要关注 acc_ 文件如何被填充，以及非 acc_ 文件如何留空（RUL/Label等）
                b1_3_sample_all_files = femto_subset_for_display[femto_subset_for_display['temp_bearing_id'] == 'Bearing1_3']
                if not b1_3_sample_all_files.empty:
                    b1_3_sample_all_files = b1_3_sample_all_files.sort_values(by='temp_file_num')
                    
                    # 选取一些文件来展示，特别是 acc 和非 acc
                    indices_to_show = []
                    acc_files_indices = b1_3_sample_all_files[b1_3_sample_all_files['temp_filename'].str.startswith('acc_', na=False)].index
                    non_acc_files_indices = b1_3_sample_all_files[~b1_3_sample_all_files['temp_filename'].str.startswith('acc_', na=False)].index

                    if not acc_files_indices.empty:
                        indices_to_show.extend(acc_files_indices[[0, len(acc_files_indices)//2, -1]].unique()) # 首、中、尾 acc 文件
                    if not non_acc_files_indices.empty:
                         indices_to_show.extend(non_acc_files_indices[[0, len(non_acc_files_indices)//2, -1]].unique()) # 首、中、尾 非acc 文件

                    if not indices_to_show: # 如果上面没选到，随便选几个
                         indices_to_show.extend(b1_3_sample_all_files.index[[0, len(b1_3_sample_all_files)//2, -1]].unique())
                    
                    final_indices_to_show = sorted(list(set(indices_to_show)))
                    
                    columns_to_display = [
                        'File', 'Label', 'RUL_label',
                        'Sample_lenth', 'Channel', 'Domain_id',
                        'Fault_Diagnosis', 'Anomaly_Detection', 'Remaining_Life', 'Digital_Twin_Prediction'
                    ]
                    columns_to_display = [col for col in columns_to_display if col in femto_subset_for_display.columns]
                    print(femto_subset_for_display.loc[final_indices_to_show, columns_to_display])
                else:
                    print("未找到 Bearing1_3 的数据用于展示。")

                print("\n更新后的 FEMTO 数据子集 Label 分布 (主要应来自 acc_ 文件):")
                print(femto_subset_for_display['Label'].value_counts(dropna=False).sort_index())
                
                femto_subset_for_display.drop(columns=['temp_filename', 'temp_bearing_id', 'temp_file_num'], inplace=True, errors='ignore')
            else:
                print("未找到 RM_003_FEMTO 的子集数据进行展示。")
    print("\n脚本执行完毕。")