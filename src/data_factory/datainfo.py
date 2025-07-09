import os
import re
import pandas as pd
from pathlib import Path
from collections import defaultdict
# 提取 metadata
def natural_sort_key(s):
    """
    用于自然排序的键函数，确保10排在2后面
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def determine_dataset_name(file_path):
    """
    从文件路径确定数据集名称
    
    Args:
        file_path (str): 文件路径
    
    Returns:
        str: 数据集名称
    """
    # 从路径中提取目录结构
    path_parts = Path(file_path).parts
    
    # 尝试查找符合命名模式的部分 (如RM_001_CWRU)
    for part in path_parts:
        if re.match(r'RM_\d+_\w+', part):
            return part
    
    # 如果没有找到匹配的模式，则使用父目录名作为数据集名称
    parent_dir = os.path.basename(os.path.dirname(file_path))
    if parent_dir and parent_dir != ".":
        return parent_dir
    
    # 最后的备选方案是使用文件名的一部分
    base_name = os.path.basename(file_path)
    name_parts = base_name.split('_')
    if len(name_parts) > 1:
        return name_parts[0]
    
    # 如果都不符合，返回无扩展名的文件名
    return os.path.splitext(base_name)[0]

def explore_datasets(root_dir):
    """
    遍历目录收集数据集信息，并按数据集分组
    
    Args:
        root_dir (str): 数据集的根目录
    
    Returns:
        list: 包含所有数据集信息的列表，已按数据集顺序排序
    """
    # 用于存储各数据集文件的字典
    dataset_files = defaultdict(list)
    
    # 首先收集所有文件并按数据集分组
    print("正在收集文件信息...")
    for root, dirs, files in os.walk(root_dir):
        # 对目录和文件进行自然排序
        dirs.sort(key=natural_sort_key)
        files.sort(key=natural_sort_key)
        
        for file in files:
            # 跳过一些非数据文件
            if file.startswith('.') or file == 'dataset_info.csv' or file == 'datainfo.py':
                continue
            
            # 检查是否在RM_004_IMS目录下，如果是，即使没有后缀也接受该文件
            is_ims_file = 'RM_004_IMS' in root
            has_valid_extension = file.endswith(('.csv', '.mat', '.txt', '.dat', '.npy', '.npz', '.h5', '.hdf5', '.xlsx', '.xls'))
            
            # 如果文件有有效后缀或者在RM_004_IMS目录下，则处理它
            if has_valid_extension or is_ims_file:
                file_path = os.path.join(root, file)
                # 确定此文件属于哪个数据集
                dataset_name = determine_dataset_name(file_path)
                
                # 获取文件扩展名，如果没有则标记为'BIN'(二进制)或'DAT'
                ext = os.path.splitext(file)[1][1:].upper()
                if not ext and is_ims_file:
                    ext = 'BIN'  # 为IMS文件指定一个默认类型
                
                dataset_files[dataset_name].append({
                    'path': file_path,
                    'rel_path': os.path.relpath(file_path, root_dir + '/' + dataset_name),
                    'name': os.path.splitext(file)[0] if os.path.splitext(file)[0] else file,  # 如果没有扩展名，使用完整文件名
                    'file': file,
                    'type': ext
                })
    
    # 对数据集名称进行排序，确保如RM_001在RM_002之前
    datasets = []
    sorted_dataset_names = sorted(dataset_files.keys(), key=natural_sort_key)
    
    # 为每个数据集分配ID并处理其文件
    for dataset_id, dataset_name in enumerate(sorted_dataset_names, 1):
        print(f"处理数据集 {dataset_id}: {dataset_name}")
        file_id_in_dataset = 0
        
        # 对每个数据集中的文件进行处理
        for file_info in dataset_files[dataset_name]:
            file_id_in_dataset += 1
            file_path = file_info['path']
            rel_path = file_info['rel_path']
            file = os.path.basename(file_path)
            file_part = file_info['file']
            
            # 获取文件基本信息
            file_info = {
                'Id': len(datasets) + 1,
                'Dataset_id': dataset_id,
                'Name': dataset_name,
                'Description': "",
                'TYPE': '',
                'File': rel_path,
                'Visiable': 1,
                'Label': '',
                'Label_Description': '',
                'RUL_label': '',
                'RUL_label_description': '',
                'Fault level': '',
                'Domain_id': '',
                'Domain_description': '',
                'Sample_rate': '',
                'Sample_lenth': '',
                'Channel': '',
                'Fault_Diagnosis': '',
                'Anomaly_Detection': '',
                'Remaining_Life': '',
                'Digital_Twin_Prediction': '',

            }
            
            # 从文件名和路径推断信息
            path_lower = rel_path.lower()
            filename_lower = file.lower()
            
            # # 尝试识别数据集类型和领域
            # domain_keywords = {
            #     'bearing': {'id': 1, 'description': '轴承故障诊断'},
            #     'gear': {'id': 2, 'description': '齿轮故障诊断'},
            #     'motor': {'id': 3, 'description': '电机故障诊断'},
            #     'pump': {'id': 4, 'description': '泵故障诊断'},
            #     'fan': {'id': 5, 'description': '风机故障诊断'},
            #     'cwru': {'id': 6, 'description': 'Case Western Reserve University 轴承数据集'},
            #     'nasa': {'id': 7, 'description': 'NASA 进行性退化数据集'},
            #     'phm': {'id': 8, 'description': 'PHM 竞赛数据集'},
            # }
            
            # for keyword, info in domain_keywords.items():
            #     if keyword in filename_lower or keyword in path_lower or keyword in dataset_name.lower():
            #         file_info['Domain_id'] = info['id']
            #         file_info['Domain_description'] = info['description']
            #         break
            
            # # 尝试识别任务类型
            # task_keywords = {
            #     'Fault_Diagnosis': ['fault', 'failure', 'diagnosis', 'diagnostic'],
            #     'Anomaly_Detection': ['anomaly', 'outlier', 'detection'],
            #     'Remaining_Life': ['remaining', 'life', 'rul', 'prognostics', 'prognosis'],
            #     'Digital_Twin_Prediction': ['digital', 'twin', 'simulation']
            # }
            
            # for task, keywords in task_keywords.items():
            #     if any(kw in filename_lower or kw in path_lower or kw in dataset_name.lower() for kw in keywords):
            #         file_info[task] = 'Yes'
                    
            #         # 如果是剩余寿命任务，尝试填充RUL标签信息
            #         if task == 'Remaining_Life' and file_info['Remaining_Life'] == 'Yes':
            #             file_info['RUL_label'] = 'RUL'
            #             file_info['RUL_label_description'] = '设备剩余使用寿命(天/小时/循环)'
            
            datasets.append(file_info)
    
    return datasets

def save_to_csv(datasets, output_file):
    """
    将数据集信息保存到CSV文件
    
    Args:
        datasets (list): 数据集信息列表
        output_file (str): 输出文件路径
    """
    if not datasets:
        print("没有找到数据集")
        return
        
    columns = [
        'Id', 'Dataset_id', 'Name', 'Description', 'TYPE', 'File', 'Visiable', 
        'Label', 'Label_Description',        'RUL_label', 'RUL_label_description', 'Fault level','Domain_id', 'Domain_description', 
        'Sample_rate', 'Sample_lenth', 'Channel', 'Fault_Diagnosis', 
        'Anomaly_Detection', 'Remaining_Life', 'Digital_Twin_Prediction',

    ]
    
    # # 处理可能包含非ASCII字符的文件路径
    # for item in datasets:
    #     if 'File' in item and item['File']:
    #         try:
    #             # 确保文件路径是有效的 UTF-8 字符串
    #             item['File'] = item['File'].encode('utf-8', errors='ignore').decode('utf-8')
    #         except Exception as e:
    #             print(f"处理文件路径时出错: {e}, 路径: {item['File']}")
    #             item['File'] = repr(item['File'])[1:-1]  # 使用字符串表示作为后备
    
    df = pd.DataFrame(datasets, columns=columns)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"找到 {len(datasets)} 个数据集文件")  # , encoding='utf-8-sig'
    print(f"数据已保存到 {output_file}")

def main():
    # 指定根目录和输出文件
    root_dir = '/home/user/data/PHMbenchdata'
    output_file = os.path.join(root_dir, 'dataset_info.csv')
    
    print(f"开始遍历目录: {root_dir}")
    # 遍历目录收集信息
    datasets = explore_datasets(root_dir)
    
    # 保存到CSV
    save_to_csv(datasets, output_file)

if __name__ == "__main__":
    main()
