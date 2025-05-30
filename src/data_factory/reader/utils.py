import pandas as pd
import scipy.io as sio
import numpy as np
import os

def load_mat_file(filepath):
    """加载MATLAB .mat文件"""
    try:
        # 尝试使用scipy.io.loadmat加载
        return sio.loadmat(filepath)
    except UnicodeDecodeError:
        # 如果遇到编码错误，尝试使用不同的参数
        try:
            return sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
        except Exception as e:
            print(f"加载文件 {filepath} 时出错: {e}")
            return None
    except Exception as e:
        print(f"加载文件 {filepath} 时出错: {e}")
        return None

def load_csv_file(filepath, **kwargs):
    """加载CSV文件"""
    try:
        # 默认尝试utf-8编码
        return pd.read_csv(filepath, **kwargs)
    except UnicodeDecodeError:
        # 如果utf-8失败，尝试其他编码
        try:
            return pd.read_csv(filepath, encoding='gb2312', **kwargs)
        except UnicodeDecodeError:
            try:
                return pd.read_csv(filepath, encoding='latin-1', **kwargs)
            except Exception as e:
                print(f"加载文件 {filepath} 时出错: {e}")
                return None
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
    
def fix_byte_order(data):
    """
    修正字节序
    :param data: 原始数据
    :return: 修正后的数据
    """
    if isinstance(data, np.ndarray) and data.dtype.byteorder not in ('=', '|'):
        # 转换为本机字节序
        data = data.astype(data.dtype.newbyteorder('='), copy=False)
    return data

def test_reader(metadata_path = '/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/data/metadata_5_data.csv',
                 data_dir = '/user/data/PHMbenchdata/raw/',name = 'RM_017_Ottawa19',output_dir = './output',read = None):
    """
    测试函数：读取metadata，筛选文件，执行reader并保存结果
    """
    # 1. 读取metadata
    

    
    try:
        # Check file extension
        file_ext = os.path.splitext(metadata_path)[1].lower()
        
        if file_ext == '.csv':
            try:
                df = pd.read_csv(metadata_path, encoding='utf-8', sep='\t')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(metadata_path, encoding='gb2312', sep='\t')
                except UnicodeDecodeError:
                    df = pd.read_csv(metadata_path, encoding='latin-1', sep='\t')
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(metadata_path)
        else:
            # Try CSV first, then Excel if that fails
            try:
                df = pd.read_csv(metadata_path, encoding='utf-8', sep='\t')
            except Exception:
                try:
                    df = pd.read_excel(metadata_path)
                except Exception as e:
                    raise ValueError(f"无法读取文件 {metadata_path}: {str(e)}")
    except Exception as e:
        print(f"读取元数据文件时出错: {str(e)}")
        raise
    
    # 2. 根据当前python文件名筛选数据
    current_name = name
    filtered_df = df[df['Name'] == current_name]
    
    print(f"找到 {len(filtered_df)} 个相关文件")
    
    # 3. 准备结果存储
    results = []
    
    # 4. 遍历文件目录，执行reader
    for index, row in filtered_df.iterrows():
        file_name = row['File']
        file_path = os.path.join(data_dir, current_name, file_name)
        
        try:
            print(f"正在处理文件: {file_name}")
            data = read(file_path)
            
            if data is not None:
                length, channels = data.shape
                results.append({
                    'File': file_name,
                    'Length': length,
                    'Channels': channels,
                    'Description': row['Description']
                })
                print(f"  - Shape: {data.shape}")
            else:
                print(f"  - 读取失败")
                
        except Exception as e:
            print(f"  - 错误: {str(e)}")
    
    # 5. 保存结果
    if results:
        results_df = pd.DataFrame(results)
        output_path = f'{output_dir}/test_results_{current_name}.csv'
        results_df.to_csv(output_path, index=False)
        
        print(f"\n结果汇总:")
        print(results_df[['File', 'Length', 'Channels']].to_string(index=False))
        print(f"\n结果已保存到: {output_path}")
    else:
        print("没有成功读取任何文件")