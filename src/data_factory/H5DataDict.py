import h5py

class H5DataDict:
    """HDF5数据字典类，模拟字典接口但实际按需从HDF5读取数据"""
    
    def __init__(self, h5file, mode='r'):
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
        # if isinstance(h5file, str):
        #     self.h5file = h5py.File(h5file, mode)
        #     self.should_close = True
        # else:
        #     self.h5file = h5file
        self.should_close = True
        self.h5_file = h5file
        self.h5f = None

        
        
    def _open_if_needed(self):
        if self.h5f is None or not hasattr(self.h5f, 'id') or not self.h5f.id.valid:
            # 先关闭旧的文件句柄，防止泄漏
            if self.h5f is not None:
                try:
                    self.h5f.close()
                except:
                    pass  # 忽略关闭时的异常
            self.h5f = h5py.File(self.h5_file, 'r', libver='latest', swmr=True)
            self._keys = set(self.h5f.keys())
    
    def __getitem__(self, key):
        """获取指定ID的数据，惰性加载"""
        self._open_if_needed()
        if str(key) not in self.h5f:
            raise KeyError(f"ID {key} not found in HDF5 file")
        # 调用时才实际加载数据到内存
        return self.h5f[str(key)][:]
    
    def __contains__(self, key):
        self._open_if_needed()
        """检查是否包含指定ID"""
        return str(key) in self.h5f
    
    def keys(self):
        """返回所有可用的ID"""
        self._open_if_needed()
        return self._keys
    
    def items(self):
        """返回ID和数据的迭代器（惰性加载）"""
        self._open_if_needed()
        for k in self._keys:
            yield int(k), self.h5f[k][:]
    
    def __len__(self):
        """返回数据集数量"""
        self._open_if_needed()
        return len(self._keys)
    def close(self):
        """显式关闭HDF5文件"""
        if self.should_close and hasattr(self, 'h5f') and self.h5f:
            self.h5f.close()
            self.h5f = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __del__(self):
        """析构函数，确保文件被关闭"""
        try:
            self.close()
        except:
            pass  # 忽略析构时的异常