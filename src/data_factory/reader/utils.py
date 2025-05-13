import pandas as pd
import scipy.io as sio

def load_mat_file(filepath):
    """加载MATLAB .mat文件"""
    try:
        return sio.loadmat(filepath)
    except Exception as e:
        print(f"加载文件 {filepath} 时出错: {e}")
        return None

def load_csv_file(filepath, **kwargs):
    """加载CSV文件"""
    try:
        return pd.read_csv(filepath, **kwargs)
    except Exception as e:
        print(f"加载文件 {filepath} 时出错: {e}")
        return None
    
def load_txt_file(filepath, **kwargs):
    """加载文本文件"""
    try:
        return pd.read_csv(filepath, **kwargs)
    except Exception as e:
        print(f"加载文件 {filepath} 时出错: {e}")
        return None
def load_data(file_path, file_type='mat', **kwargs):
    """
    加载数据文件
    :param file_path: 文件路径
    :param file_type: 文件类型 ('mat', 'csv', 'txt')
    :param kwargs: 其他参数
    :return: 数据
    """
    if file_type == 'mat':
        return load_mat_file(file_path)
    elif file_type == 'csv':
        return load_csv_file(file_path, **kwargs)
    elif file_type == 'txt':
        return load_txt_file(file_path, **kwargs)
    else:
        raise ValueError("Unsupported file type. Supported types are: 'mat', 'csv', 'txt'.")