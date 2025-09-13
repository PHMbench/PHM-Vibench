import pandas as pd
from ..data_utils import MetadataAccessor
from .Get_id import Get_DG_ids, Get_CDDG_ids # ,Get_GFS_ids


def search_ids_for_task(metadata_accessor, args_task):
    """
    根据任务参数从元数据中搜索训练/验证和测试ID。
    """
    train_val_ids = []
    test_ids = []

    if args_task.target_system_id is not None:
        if args_task.type == 'DG':
            train_val_ids, test_ids = Get_DG_ids(metadata_accessor, args_task)
        elif args_task.type == 'CDDG':
            train_val_ids, test_ids = Get_CDDG_ids(metadata_accessor, args_task)
        elif args_task.type == 'GFS':
            train_val_ids, test_ids = list(metadata_accessor.keys()), list(metadata_accessor.keys())
        elif args_task.type == 'Pretrain':
            # For pretraining, we typically use all available IDs
            train_val_ids, test_ids = list(metadata_accessor.keys()), list(metadata_accessor.keys())
        # Add other task types here if needed
        # elif args_task.type == 'SOME_OTHER_TYPE':
        #     train_val_ids, test_ids = get_some_other_type_ids(metadata_accessor, args_task)
        else:
            # Default or error handling if task type is unknown or not specified for ID searching
            print(f"Warning: Task type {args_task.type} not specifically handled for ID searching. Defaulting to all keys.")
            train_val_ids, test_ids = list(metadata_accessor.keys()), list(metadata_accessor.keys())

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
    

    # 检查Label是否有空值 
    filtered_df = filtered_df[filtered_df['Label'].notna()]

    # 检查Label是否有-1
    filtered_df = filtered_df[filtered_df['Label'] != -1]

    # # 检查是否visiable？
    # filtered_df = filtered_df[filtered_df['Visible'] == True]

    print(f"筛选前元数据行数: {len(metadata_accessor.df)}")
    print(f"筛选后元数据行数: {len(filtered_df)}")
    
    if len(filtered_df) == 0:
        print(f"警告: 目标数据集ID {args_task.target_system_id} 没有匹配的记录")
    
    filtered_df.reset_index(drop=True, inplace=True)
    target_metadata = MetadataAccessor(filtered_df, key_column=metadata_accessor.key_column)
    return target_metadata
