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

def read_data(args_data):
    """
    极简数据读取器
    输入: args_data (dict或命名空间，需有data_dir和metadata_file)
    输出: (metadata, {id: data_array})
    """
    # 1. 读取元数据
    metadata_path = os.path.join(args_data.data_dir, args_data.metadata_file)
    meta_df = pd.read_csv(metadata_path)
    metadata = {row.Id: row.to_dict() for _, row in meta_df.iterrows()}

    # # 动态加载当前目录下所有py模块
    # current_dir = os.path.dirname(__file__)
    # module_map = {}
    # for file in glob.glob(os.path.join(current_dir, "*.py")):
    #     if not os.path.basename(file).startswith("_") and not file.endswith("__init__.py"):
    #         module_name = os.path.basename(file)[:-3]
    #         mod = importlib.import_module(f"src.data_factory.{module_name}")
    #         module_map[module_name.lower()] = mod


    
    # 2. 遍历元数据，读取数据
    data_dict = {}
    for id, meta in metadata.items():
        name = meta['Name']
        file = meta['File']

        module_name = os.path.basename(file)[:-3]
        mod = importlib.import_module(f"src.data_factory.{module_name}")
        file_path = os.path.join(args_data.data_dir, file)
        data_dict[id] = mod.read(args_data,file_path)


    return metadata, data_dict