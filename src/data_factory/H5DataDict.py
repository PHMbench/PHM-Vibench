from pathlib import Path

import h5py

class H5DataDict:
    """HDF5数据字典类，模拟字典接口但实际按需从HDF5读取数据"""
    
    def __init__(
        self,
        h5file,
        mode='r',
        *,
        allowed_ids=None,
        manifest_path=None,
        metadata=None,
    ):
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
        self.verified = manifest_path is not None
        self.h5_file = str(Path(h5file).expanduser().resolve(strict=self.verified))
        self.h5f = None
        self._entries = {}
        self._allowed_keys = None
        self._chunk_rows = None
        if self.verified:
            # Keep the manifest module lazy so ``python -m
            # src.data_factory.protocol_cache`` remains a clean CLI entrypoint.
            from .protocol_cache import load_cache_manifest, validate_entry_metadata

            if mode != 'r':
                raise ValueError("verified H5DataDict supports read-only mode 'r' only")
            if allowed_ids is None or metadata is None:
                raise ValueError(
                    "verified H5DataDict requires allowed_ids and active metadata"
                )
            manifest, entries = load_cache_manifest(
                manifest_path,
                expected_cache_path=self.h5_file,
            )
            allowed = []
            for value in allowed_ids:
                try:
                    sample_id = int(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"verified cache Id must be an integer: {value!r}") from exc
                if str(sample_id) != str(value) and not isinstance(value, int):
                    raise ValueError(f"verified cache Id is not canonical: {value!r}")
                if sample_id not in entries:
                    raise KeyError(f"cache manifest is missing active metadata Id {sample_id}")
                validate_entry_metadata(entries[sample_id], metadata[sample_id])
                allowed.append(str(sample_id))
            if len(set(allowed)) != len(allowed):
                raise ValueError("verified cache allowed_ids contain duplicates")
            self._entries = entries
            self._allowed_keys = frozenset(allowed)
            self._chunk_rows = int(manifest["hashing"]["chunk_rows"])

        
        
    def _open_if_needed(self):
        if self.h5f is None or not hasattr(self.h5f, 'id') or not self.h5f.id.valid:
            # 先关闭旧的文件句柄，防止泄漏
            if self.h5f is not None:
                try:
                    self.h5f.close()
                except Exception:
                    pass  # 忽略关闭时的异常
            self.h5f = h5py.File(self.h5_file, 'r', libver='latest', swmr=True)
            if self.verified:
                missing = sorted(key for key in self._allowed_keys if key not in self.h5f)
                if missing:
                    self.h5f.close()
                    self.h5f = None
                    raise KeyError(f"cache is missing active metadata IDs: {missing[:10]}")
                self._keys = set(self._allowed_keys)
            else:
                self._keys = set(self.h5f.keys())
    
    def __getitem__(self, key):
        """获取指定ID的数据，惰性加载"""
        self._open_if_needed()
        if self.verified:
            try:
                sample_id = int(key)
            except (TypeError, ValueError) as exc:
                raise KeyError(f"invalid verified cache Id request: {key!r}") from exc
            cache_key = str(sample_id)
            if cache_key not in self._allowed_keys:
                raise KeyError(f"cache Id {sample_id} is outside the active metadata allowlist")
            dataset = self.h5f[cache_key]
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError(f"cache root key {cache_key!r} is not an HDF5 dataset")
            from .protocol_cache import read_verified_dataset

            return read_verified_dataset(
                dataset,
                entry=self._entries[sample_id],
                chunk_rows=self._chunk_rows,
            )
        if str(key) not in self.h5f:
            raise KeyError(f"ID {key} not found in HDF5 file")
        # 调用时才实际加载数据到内存
        return self.h5f[str(key)][:]
    
    def __contains__(self, key):
        self._open_if_needed()
        """检查是否包含指定ID"""
        if self.verified:
            try:
                return str(int(key)) in self._allowed_keys and str(int(key)) in self.h5f
            except (TypeError, ValueError):
                return False
        return str(key) in self.h5f
    
    def keys(self):
        """返回所有可用的ID"""
        self._open_if_needed()
        return self._keys
    
    def items(self):
        """返回ID和数据的迭代器（惰性加载）"""
        self._open_if_needed()
        for k in sorted(self._keys, key=lambda value: int(value)):
            yield int(k), self[k]
    
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
        except Exception:
            pass  # 忽略析构时的异常
