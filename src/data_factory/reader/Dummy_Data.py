import pandas as pd
import numpy as np
from pathlib import Path

def _make_synthetic(args_data=None, seed: int = 0) -> np.ndarray:
    window_size = int(getattr(args_data, "window_size", 128)) if args_data is not None else 128
    num_window = int(getattr(args_data, "num_window", 8)) if args_data is not None else 8
    stride = int(getattr(args_data, "stride", 16)) if args_data is not None else 16

    min_len = window_size + max(0, num_window - 1) * stride
    length = max(min_len, 2048)

    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10, length, dtype=np.float32)
    s1 = np.sin(2 * np.pi * 5.0 * t) + 0.1 * rng.standard_normal(length, dtype=np.float32)
    s2 = np.sin(2 * np.pi * 13.0 * t + 0.3) + 0.1 * rng.standard_normal(length, dtype=np.float32)
    return np.stack([s1, s2], axis=1)


def read(file_path, *args):
    """
    Reads data from a CSV file specified by file_path.

    Args:
        args_data: Data configuration arguments (currently unused).
        file_path (str): Path to the CSV data file (e.g., Vbench/data/Dummy_Dataset/dummy1.csv).

    Returns:
        numpyarray: dimention as lenth \times channel
    """
    try:
        args_data = args[0] if args else None
        file_path = str(file_path)
        if not Path(file_path).exists():
            # Repo-shipped smoke/demo mode: generate deterministic synthetic data.
            seed = abs(hash(file_path)) % (2**32)
            data = _make_synthetic(args_data=args_data, seed=int(seed))
            print(f"[Dummy_Data] raw file missing; generated synthetic data for: {file_path}")
            return data

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
        # Depending on downstream requirements, you might want to convert
        # the DataFrame to a NumPy array, e.g., return df.values
        # For now, returning the DataFrame is flexible.
        print(f"Successfully read data from: {file_path}")
        # 💯这里加对应数据的各种读取
        # Extract columns 2-5 (indices 1-4 in zero-based indexing)
        df = df.iloc[:, 1:3]
        print(f"Selected columns 2-5 from the dataset.")

        return df.values.astype(np.float32)
    
    
    except FileNotFoundError:
        args_data = args[0] if args else None
        seed = abs(hash(str(file_path))) % (2**32)
        data = _make_synthetic(args_data=args_data, seed=int(seed))
        print(f"[Dummy_Data] file not found; generated synthetic data for: {file_path}")
        return data
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {file_path} is empty.")
        return None
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")
        return None
    
    # 加入更多的异常处理
    # 例如：处理文件格式错误、数据解析错误等

def get_dataset(args_data, file_path):
    """
    Reads data from a CSV file specified by file_path.

    Args:
        args_data: Data configuration arguments (currently unused).
        file_path (str): Path to the CSV data file (e.g., Vbench/data/Dummy_Dataset/dummy1.csv).

    Returns:
        numpyarray: dimention as lenth \times channel
    """
    # 读取数据
    data = read(args_data, file_path)
    
    # 处理数据
    if data is not None:
        # 这里可以添加数据预处理的代码
        print(f"Data shape: {data.shape}")
        return data
    else:
        print("No data to process.")
        return None
