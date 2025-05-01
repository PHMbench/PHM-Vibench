import pandas as pd
import numpy as np

def read(file_path,*args):
    """
    Reads data from a CSV file specified by file_path.

    Args:
        args_data: Data configuration arguments (currently unused).
        file_path (str): Path to the CSV data file (e.g., Vbench/data/Dummy_Dataset/dummy1.csv).

    Returns:
        numpyarray: dimention as lenth \times channel
    """
    try:
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

        return df.values[:truncate_lenth]
    
    
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        return None
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