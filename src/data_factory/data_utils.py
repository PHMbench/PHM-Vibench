import os
import pandas as pd
from typing import Union, Optional
import subprocess
import sys


def download_data(data_file: Optional[str] = "metadata_6_1.xlsx",
                       save_path: Optional[str] = './data/',
                       source: str = 'auto') -> bool:
    """
    下载元数据文件，支持 ModelScope 和 Hugging Face 两种源
    Args:
        data_file: 数据文件名，默认为 "metadata_6_1.xlsx"
        save_path: 保存路径，默认为当前目录
        source: 下载源，'modelscope', 'huggingface' 或 'auto'
        
    Returns:
        bool: 下载是否成功
    """

    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    check_exists = os.path.exists(os.path.join(save_path, data_file))
    if check_exists:
        print(f"[INFO] 检查缓存文件: 数据文件已存在: {os.path.join(save_path, data_file)} ")
        return True
    
    success = False

    if source == 'auto' or source == 'modelscope':
        print(f"[INFO] 检查缓存文件, 尝试从 ModelScope 下载 {data_file}...")
        try:
            # 使用 modelscope 下载
            cmd = f"modelscope download --dataset PHMbench/PHM-Vibench {data_file} --local_dir {save_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"[SUCCESS] 从 ModelScope 成功下载 {data_file}")
                success = True
            else:
                print(f"[WARNING] ModelScope 下载失败: {result.stderr}")

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            print(f"[WARNING] ModelScope 下载失败: {e}")
    
    # 如果 ModelScope 失败，尝试 Hugging Face
    if not success and (source == 'auto' or source == 'huggingface'):
        print(f"[INFO] 尝试从 Hugging Face 下载 {data_file}...")
        try:
            
            # 使用 huggingface_hub 下载
            from huggingface_hub import hf_hub_download
            
            downloaded_file = hf_hub_download(
                repo_id="PHMbench/PHM-Vibench",
                filename=data_file,
                repo_type="dataset",
                local_dir=save_path,
                local_dir_use_symlinks=False
            )
            print(f"[SUCCESS] 从 Hugging Face 成功下载 {data_file}")
            success = True
            
        except ImportError:
            print("[WARNING] huggingface_hub 未安装，请运行: pip install huggingface_hub")
        except Exception as e:
            print(f"[WARNING] Hugging Face 下载失败: {e}")
    
    if not success:
        print(f"[ERROR] 所有下载源都失败，无法下载 {data_file}")
        print("[INFO] 请手动下载元数据文件或检查网络连接")
    
    return success

if __name__ == "__main__":
    # 测试下载功能
    download_data(data_file=" 'RM_006_THU.h5'", save_path="./data/", source='auto')
    print("测试完成")




def smart_read_csv(file_path, auto_detect=True):
    """智能读取CSV/Excel文件，自动尝试不同的分隔符和编码"""
    # 检查文件扩展名
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # 如果是Excel文件，直接使用pandas读取
    if file_ext in ['.xlsx', '.xls']:
        try:
            return pd.read_excel(file_path)
        except Exception as e:
            print(f"读取Excel文件失败: {e}")
            raise Exception(f"无法读取Excel文件 {file_path}: {e}")
    
    # CSV读取逻辑
    if auto_detect:
        # 先尝试检测文件前几行来判断可能的分隔符
        try:
            with open(file_path, 'rb') as f:
                sample = f.read(4096)  # 读取前4KB判断格式
                
            # 尝试用UTF-8解码
            sample_text = sample.decode('utf-8', errors='ignore')
            
            # 根据文件内容推测分隔符
            comma_count = sample_text.count(',')
            tab_count = sample_text.count('\t')
            
            # 根据分隔符频率选择解析策略
            if comma_count > tab_count:
                # 更可能是逗号分隔的文件
                try:
                    return pd.read_csv(file_path)
                except UnicodeDecodeError:
                    return pd.read_csv(file_path, encoding='gbk')
            else:
                # 更可能是制表符分隔的文件
                try:
                    return pd.read_csv(file_path, sep='\t')
                except UnicodeDecodeError:
                    return pd.read_csv(file_path, sep='\t', encoding='gbk', low_memory=False)
                
        except Exception as e:
            print(f"自动检测格式失败: {e}，尝试默认方法")
        
        # 如果检测失败，按照优先级尝试不同组合
        encodings = ['utf-8', 'gbk', 'latin1']
        separators = [',', '\t']
        
        for encoding in encodings:
            for sep in separators:
                try:
                    return pd.read_csv(file_path, encoding=encoding, sep=sep)
                except Exception as e:
                    continue
                    
        # 最后的后备方案，使用最宽松的参数
        try:
            return pd.read_csv(file_path, encoding='latin1', sep=None, engine='python')
        except Exception as e:
            raise Exception(f"无法读取文件 {file_path}，尝试了所有常见格式: {e}")


class MetadataAccessor:
    """提供类似字典的接口访问DataFrame数据的类"""
    
    def __init__(self, dataframe, key_column='Id'):
        """初始化元数据访问器
        
        Args:
            dataframe: pandas DataFrame包含元数据
            key_column: 用作键的列名，默认为'Id'
        """
        self.df = dataframe
        self.key_column = key_column
        # 为了加速查询，将索引设置为key_column
        self.df.set_index(key_column, inplace=True, drop=False)
    
    def __getitem__(self, key):
        """通过键获取元数据行，返回一个字典
        
        Args:
            key: 要查找的键值
            
        Returns:
            dict: 包含该行所有数据的字典
        """
        try:
            return self.df.loc[key].to_dict()
        except KeyError:
            raise KeyError(f"找不到ID为{key}的记录")
    
    def __contains__(self, key):
        """检查键是否存在
        
        Args:
            key: 要检查的键值
            
        Returns:
            bool: 键是否存在
        """
        return key in self.df.index
    
    def keys(self):
        """获取所有键的列表
        
        Returns:
            列表: 所有键的列表
        """
        return list(self.df[self.key_column])
    
    def items(self):
        """获取(key, value)对的列表，类似字典的items方法
        
        Returns:
            列表: (key, value)元组的列表
        """
        for key in self.keys():
            yield (key, self[key])
    
    def values(self):
        """获取所有值的列表
        
        Returns:
            列表: 所有行数据字典的列表
        """
        return [row.to_dict() for _, row in self.df.iterrows()]
    
    def get(self, key, default=None):
        """获取键对应的值，如果不存在返回默认值
        
        Args:
            key: 要查找的键
            default: 键不存在时返回的默认值
            
        Returns:
            字典或默认值
        """
        try:
            return self[key]
        except KeyError:
            return default
    
    def __len__(self):
        """返回元数据条目数量"""
        return len(self.df)
    
    def query(self, query_str):
        """使用pandas的query功能直接查询数据
        
        Args:
            query_str: pandas query语法的查询字符串
            
        Returns:
            查询结果DataFrame
        """
        return self.df.query(query_str)

