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


class H5DataDict:
    """HDF5数据字典类，模拟字典接口但实际按需从HDF5读取数据"""
    
    def __init__(self, h5file):
        """初始化HDF5数据字典
        
        Args:
            h5file: 打开的h5py.File对象

        Note:
        直接返回 h5f 的问题
        数据访问不完整：h5f[key] 返回的是 h5py.Dataset 对象，不是实际数据。要获取实际数据，需要使用 h5f[key][:]，这对用户不直观。

        类型转换：HDF5 文件中的键必须是字符串，而你的 metadata 字典中的键可能是整数。H5DataDict 类自动处理了这种转换，但直接使用 h5f 需要手动转换：

        文件管理：没有明确的文件关闭机制。如果你的程序运行时间长，可能会导致文件句柄泄露。

        接口一致性：如果其他代码假定 data_dict[id] 直接返回 NumPy 数组，使用原始 h5f 会导致接口不一致。
                
        """
        self.h5file = h5file
        self._keys = set(self.h5file.keys())
    
    def __getitem__(self, key):
        """获取指定ID的数据，惰性加载"""
        if str(key) not in self.h5file:
            raise KeyError(f"ID {key} not found in HDF5 file")
        # 调用时才实际加载数据到内存
        return self.h5file[str(key)][:]
    
    def __contains__(self, key):
        """检查是否包含指定ID"""
        return str(key) in self.h5file
    
    def keys(self):
        """返回所有可用的ID"""
        return self._keys
    
    def items(self):
        """返回ID和数据的迭代器（惰性加载）"""
        for k in self._keys:
            yield int(k), self.h5file[k][:]
    
    def __len__(self):
        """返回数据集数量"""
        return len(self._keys)


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
    
    # 2. 如果存在缓存且启用了缓存，直接打开并返回
    if use_cache and os.path.exists(cache_file):
        try:
            # 检查缓存文件是否包含所有ID
            with h5py.File(cache_file, 'r') as h5f:
                missing_ids = [id for id in metadata.keys() if str(id) not in h5f]
            
            if not missing_ids:
                print(f"所有数据都在缓存中，直接使用缓存文件: {cache_file}")
                # 用'r+'模式打开，这样即使文件已存在也可以写入
                h5f = h5py.File(cache_file, 'r')
                return metadata, H5DataDict(h5f)
            else:
                print(f"缓存中缺少ID: {missing_ids}，将更新缓存")
        except Exception as e:
            print(f"读取缓存出错: {e}")
    

    # 3. 如果没有缓存或有缺失数据，读取原始数据
    h5f = h5py.File(cache_file, 'w')

    for id, meta in metadata.items():
        try:
            name = meta['Name']
            file = meta['File']
            mod = importlib.import_module(f"src.data_factory.{name}")
            file_path = os.path.join(args_data.data_dir, f'raw/{name}/{file}')
            data = mod.read(args_data, file_path)
            h5f.create_dataset(str(id), data=data)
        except Exception as e:
            print(f"Error loading data for ID {id}: {e}")
    h5f.flush()

    return metadata, H5DataDict(h5f)