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

def data_reader(args_data, use_cache=True):
    """
    极简数据读取器
    
    Args:
        args_data: 包含data_dir和metadata_file的字典或命名空间
        use_cache: 是否使用HDF5缓存，默认为True
        
    Returns:
        tuple: (metadata, {id: data_array})
    """
    # 1. 读取元数据
    metadata_path = os.path.join(args_data.data_dir, args_data.metadata_file)
    meta_df = pd.read_csv(metadata_path)
    metadata = {row.Id: row.to_dict() for _, row in meta_df.iterrows()}

    # 构建缓存文件路径
    cache_file = os.path.join(args_data.data_dir, f"{os.path.splitext(args_data.metadata_file)[0]}.h5")
    
    # 2. 如果存在缓存且启用了缓存，直接读取缓存
    if use_cache and os.path.exists(cache_file):
        try:
            data_dict = {}
            with h5py.File(cache_file, 'r') as h5f:
                for id in metadata.keys():
                    if str(id) in h5f:
                        data_dict[id] = h5f[str(id)][:]
                    else:
                        print(f"Warning: ID {id} not found in cache, will read from raw data")
                        
            # 检查是否所有ID都已加载
            missing_ids = set(metadata.keys()) - set(data_dict.keys())
            print(f"Missing IDs in cache: {missing_ids}")
            if not missing_ids:
                return metadata, data_dict
        except Exception as e:
            print(f"Error reading cache: {e}")
    
    # 3. 如果没有缓存或有缺失数据，读取原始数据
    data_dict = {} if not 'data_dict' in locals() else data_dict
    for id, meta in metadata.items():
        if id in data_dict:
            continue
        
        try:
            name = meta['Name']
            file = meta['File']
            mod = importlib.import_module(f"src.data_factory.{name}")
            file_path = os.path.join(args_data.data_dir, f'raw/{name}/{file}')
            data_dict[id] = mod.read(args_data, file_path)
        except Exception as e:
            print(f"Error loading data for ID {id}: {e}")
    
    # 4. 如果启用了缓存，保存数据到HDF5文件
    if use_cache:
        try:
            with h5py.File(cache_file, 'w') as h5f:
                for id, data in data_dict.items():
                    h5f.create_dataset(str(id), data=data)
        except Exception as e:
            print(f"Error writing cache: {e}")

    return metadata, data_dict