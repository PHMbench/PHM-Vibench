import os
import numpy as np
import pandas as pd
import signal
from contextlib import contextmanager
from utils import load_data, fix_byte_order

# @contextmanager
# def timeout(duration):
#     """超时上下文管理器"""
#     def timeout_handler(signum, frame):
#         raise TimeoutError(f"操作超时 ({duration}秒)")
    
#     signal.signal(signal.SIGALRM, timeout_handler)
#     signal.alarm(duration)
#     try:
#         yield
#     finally:
#         signal.alarm(0)

def read(file_path, *args):
    """
    读取RM_011_pump数据集，支持.xlsx和.csv格式
    """
    try:
        # 根据文件扩展名选择读取方式
        if file_path.endswith('.xlsx'):
            # 尝试多个引擎和超时机制
            engines = [ 'openpyxl'] #  'xlrd','calamine',
            data = None
            
            for engine in engines:
                try:
                    # print(f"尝试使用 {engine} 引擎读取Excel文件...")
                    # with timeout(60):  # 30秒超时
                        # Get the last sheet
                    excel_file = pd.ExcelFile(file_path, engine=engine)
                    last_sheet = excel_file.sheet_names[-1]
                    data = pd.read_excel(file_path, sheet_name=last_sheet, header=None, engine=engine).values
                    print(f"成功使用 {engine} 引擎读取文件")
                    break
                # except TimeoutError:
                #     print(f"{engine} 引擎超时，尝试下一个引擎...")
                #     continue
                except Exception as e:
                    print(f"{engine} 引擎失败: {e}，尝试下一个引擎...")
                    continue
            
            if data is None:
                # 如果所有引擎都失败，尝试转换为CSV后读取
                print("所有Excel引擎都失败，尝试其他方法...")
                raise Exception("所有Excel读取引擎都失败")
                
        # elif file_path.endswith('.csv'):
        #     data = pd.read_csv(file_path, header=None).values
        # else:
        #     # 尝试作为csv读取
        #     data = pd.read_csv(file_path, header=None).values
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None
    
    data = data[1:]
    data =data.float()
    # 修复字节序问题
    data = fix_byte_order(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    data = data.astype(np.float64)
    return data

if __name__ == "__main__":
    from utils import test_reader
    # 测试数据读取
    test_reader(metadata_path = '/home/user/data/PHMbenchdata/metadata_5_29.xlsx',
                 data_dir = '/home/user/data/PHMbenchdata/raw/',
                 name = 'RM_011_pump',
                 output_dir = '/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/src/data_factory/reader/output',
                 read=read)