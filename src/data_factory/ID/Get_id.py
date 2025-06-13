
import pandas as pd

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

def Get_DG_ids(metadata_accessor, args_task):
    """
    Retrieves training/validation and test IDs for Domain Generalization (DG) tasks.
    """
    train_df = metadata_accessor.df[
        (metadata_accessor.df['Domain_id'].isin(args_task.source_domain_id)) & 
        (metadata_accessor.df['Dataset_id'].isin(args_task.target_system_id))
    ]
    test_df = metadata_accessor.df[
        (metadata_accessor.df['Domain_id'].isin(args_task.target_domain_id)) &
        (metadata_accessor.df['Dataset_id'].isin(args_task.target_system_id))
    ]
    
    train_df = remove_invalid_labels(train_df)
    test_df = remove_invalid_labels(test_df)
    
    train_val_ids = list(train_df['Id'])
    test_ids = list(test_df['Id'])
    
    return train_val_ids, test_ids

# def Get_GFS_ids(metadata_accessor, args_task):
#     。。

# def Get_pretrain_ids(metadata_accessor, args_task):
#     """
#     Retrieves training/validation and test IDs for pretraining tasks.
    
#     output:
#         train_val_ids (list): List of IDs for training/validation set.
#         test_ids (list): List of IDs for test set.
#     """
#     # For pretraining, we typically use all available IDs
#     train_val_ids = list(metadata_accessor.df['Id'])
#     test_ids = list(metadata_accessor.df['Id'])
    
#     print(f"Pretrain划分 - 使用全部数据集ID")
#     print(f"训练/验证样本数: {len(train_val_ids)}")
#     print(f"测试样本数: {len(test_ids)}")
    
#     return train_val_ids, test_ids

def Get_CDDG_ids(metadata_accessor, args_task):
    """
    Retrieves training/validation and test IDs for Cross-Domain Domain Generalization (CDDG) tasks.

    output:
        train_val_ids (list): List of IDs for source domain for different systems.
        test_ids (list): List of IDs for test set.
    """
    filtered_df = metadata_accessor.df[metadata_accessor.df['Dataset_id'].isin(args_task.target_system_id)]
    filtered_df = remove_invalid_labels(filtered_df)
    
    dataset_domains = {}
    for dataset_id in args_task.target_system_id:
        dataset_df = filtered_df[filtered_df['Dataset_id'] == dataset_id]
        domains = sorted(dataset_df['Domain_id'].unique())
        # Filter out NaN values from domains
        domains = [d for d in domains if not pd.isna(d)]
        dataset_domains[dataset_id] = domains
    
    train_domains = {}
    test_domains = {}
    for dataset_id, domains in dataset_domains.items():
        test_count = min(args_task.target_domain_num, len(domains))
        train_domains[dataset_id] = domains[:-test_count] if test_count > 0 else domains
        test_domains[dataset_id] = domains[-test_count:] if test_count > 0 else []
    
    train_val_ids = []
    test_ids = []
    for dataset_id_val in args_task.target_system_id:
        # Training set rows
        for domain_id in train_domains[dataset_id_val]:
            train_val_ids.extend(
                filtered_df[(filtered_df['Dataset_id'] == dataset_id_val) & 
                            (filtered_df['Domain_id'] == domain_id)]['Id'].tolist()
            )
        # Test set rows
        for domain_id in test_domains[dataset_id_val]:
            test_ids.extend(
                filtered_df[(filtered_df['Dataset_id'] == dataset_id_val) & 
                            (filtered_df['Domain_id'] == domain_id)]['Id'].tolist()
            )
    
    print(f"CDGD划分 - 选择每个数据集的最后{args_task.target_domain_num}个domain作为测试集")
    for dataset_id_print in args_task.target_system_id:
        print(f"数据集 {dataset_id_print}:")
        print(f"  - 训练域: {train_domains[dataset_id_print]}")
        print(f"  - 测试域: {test_domains[dataset_id_print]}")
    print(f"训练/验证样本数: {len(train_val_ids)}")
    print(f"测试样本数: {len(test_ids)}")
    
    return train_val_ids, test_ids
