import os
import pandas as pd

def update_metadata(data_map, metadata_path, output_metadata_path=None, dataset_name='RM_001_CWRU'):
    """
    直接从数据映射更新元数据文件。
    
    Args:
        data_map (dict): 映射文件编号到其元数据信息的字典
        metadata_path (str): 原始元数据CSV文件的路径
        output_metadata_path (str, optional): 保存更新后元数据的位置。默认为原始元数据路径
        dataset_name (str): 元数据文件中使用的数据集名称前缀
    
    Returns:
        pd.DataFrame: 包含更新后元数据的DataFrame
    """
    # 如未指定输出路径，则使用输入路径
    if output_metadata_path is None:
        output_metadata_path = metadata_path
    
    # 加载元数据文件
    try:
        metadata_df = pd.read_csv(metadata_path)
        print(f"成功从以下位置加载元数据: {metadata_path}")
        print(f"元数据包含 {len(metadata_df)} 条记录")
    except Exception as e:
        print(f"读取元数据文件时出错: {e}")
        return None
    
    if not data_map:
        print("未提供数据映射，无法更新元数据。")
        return metadata_df
    
    # 查找与数据集名称匹配的元数据行
    dataset_mask = metadata_df['Name'].str.contains(dataset_name, na=False)
    dataset_indices = metadata_df[dataset_mask].index
    
    # 更新匹配的记录
    updated_rows = 0
    not_found = []
    
    for idx in dataset_indices:
        file_name = metadata_df.loc[idx, 'File']
        if isinstance(file_name, str) and "." in file_name:
            file_num = file_name.split('.')[0]
            
            # 检查数据映射中是否有匹配的文件
            if file_num in data_map:
                # 更新元数据
                for key, value in data_map[file_num].items():
                    # 将列名首字母大写
                    column = key[0].upper() + key[1:] if key and len(key) > 0 else key
                    
                    # 仅当列在元数据中存在时更新
                    if column in metadata_df.columns:
                        metadata_df.loc[idx, column] = value
                
                updated_rows += 1
            else:
                not_found.append(file_name)
    
    # 保存更新后的元数据
    metadata_df.to_csv(output_metadata_path, index=False)
    
    print(f"更新了 {updated_rows} 条元数据记录")
    
    if not_found:
        print(f"数据映射中未找到 {len(not_found)} 个文件:")
        for f in not_found[:10]:  # 只显示前10个
            print(f"  - {f}")
        if len(not_found) > 10:
            print(f"  ... 以及其他 {len(not_found) - 10} 个")
    
    return metadata_df

# 示例用法:
if __name__ == "__main__":
    # 示例数据映射
    bearing_data_map = {
        "97": {"label": 0, "fault_level": 0, "sample_rate": 48000, "load_hp": 0, "dataset": "Normal"}
        # 根据需要添加更多条目
    }
    
    metadata_path = '/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/data/metadata_19_5_4.csv'
    output_metadata = '/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/data/metadata_updated.csv'
    
    # 更新元数据
    updated_metadata = update_metadata(
        data_map=bearing_data_map,
        metadata_path=metadata_path,
        output_metadata_path=output_metadata,
        dataset_name='RM_001_CWRU'
    )
    
    # 显示结果预览
    if updated_metadata is not None:
        print("\n更新后的元数据预览:")
        print(updated_metadata.head())