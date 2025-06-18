
import pandas as pd
from ..data_utils import MetadataAccessor

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

def search_ids_for_task(metadata_accessor, args_task):
    """
    根据任务参数从元数据中搜索训练/验证和测试ID。
    """
    train_val_ids = []
    test_ids = []

    if args_task.target_system_id is not None:
        if args_task.type == 'DG':
            train_df = metadata_accessor.df[
                (metadata_accessor.df['Domain_id'].isin(args_task.source_domain_id)) & 
                (metadata_accessor.df['Dataset_id'].isin(args_task.target_system_id))]
            test_df = metadata_accessor.df[
                (metadata_accessor.df['Domain_id'].isin(args_task.target_domain_id)) &
                (metadata_accessor.df['Dataset_id'].isin(args_task.target_system_id))]
            
            train_df = remove_invalid_labels(train_df)
            test_df = remove_invalid_labels(test_df)
            
            train_val_ids = list(train_df['Id'])
            test_ids = list(test_df['Id'])
        elif args_task.type == 'CDDG':
            filtered_df = metadata_accessor.df[metadata_accessor.df['Dataset_id'].isin(args_task.target_system_id)]
            filtered_df = remove_invalid_labels(filtered_df)
            
            dataset_domains = {}
            for dataset_id in args_task.target_system_id:
                dataset_df = filtered_df[filtered_df['Dataset_id'] == dataset_id]
                domains = sorted(dataset_df['Domain_id'].unique())
                domains = [d for d in domains if not pd.isna(d)]
                dataset_domains[dataset_id] = domains
            
            train_domains = {}
            test_domains = {}
            for dataset_id, domains in dataset_domains.items():
                test_count = min(args_task.target_domain_num, len(domains))
                train_domains[dataset_id] = domains[:-test_count] if test_count > 0 else domains
                test_domains[dataset_id] = domains[-test_count:] if test_count > 0 else []
            
            current_train_ids = []
            current_test_ids = []
            for dataset_id_val in args_task.target_system_id:
                for domain_id in train_domains[dataset_id_val]:
                    current_train_ids.extend(
                        filtered_df[(filtered_df['Dataset_id'] == dataset_id_val) & 
                                (filtered_df['Domain_id'] == domain_id)]['Id'].tolist()
                    )
                for domain_id in test_domains[dataset_id_val]:
                    current_test_ids.extend(
                        filtered_df[(filtered_df['Dataset_id'] == dataset_id_val) & 
                                (filtered_df['Domain_id'] == domain_id)]['Id'].tolist()
                    )
            train_val_ids = current_train_ids
            test_ids = current_test_ids
            
            print(f"CDGD划分 - 选择每个数据集的最后{args_task.target_domain_num}个domain作为测试集")
            for dataset_id_print in args_task.target_system_id:
                print(f"数据集 {dataset_id_print}:")
                print(f"  - 训练域: {train_domains[dataset_id_print]}")
                print(f"  - 测试域: {test_domains[dataset_id_print]}")
            print(f"训练/验证样本数: {len(train_val_ids)}")
            print(f"测试样本数: {len(test_ids)}")
    else:
        train_val_ids, test_ids = list(metadata_accessor.keys()), list(metadata_accessor.keys())
    
    return train_val_ids, test_ids

def search_target_dataset_metadata(metadata_accessor, args_task):
    """
    根据任务参数筛选目标数据集的元数据。
    """
    if not args_task.target_system_id:
        print("未指定目标数据集ID，返回全部元数据")
        return metadata_accessor
    
    filtered_df = metadata_accessor.df[
        metadata_accessor.df['Dataset_id'].isin(args_task.target_system_id)].copy()
    
    print(f"筛选前元数据行数: {len(metadata_accessor.df)}")
    print(f"筛选后元数据行数: {len(filtered_df)}")
    
    if len(filtered_df) == 0:
        print(f"警告: 目标数据集ID {args_task.target_system_id} 没有匹配的记录")
    
    filtered_df.reset_index(drop=True, inplace=True)
    target_metadata = MetadataAccessor(filtered_df, key_column=metadata_accessor.key_column)
    return target_metadata
