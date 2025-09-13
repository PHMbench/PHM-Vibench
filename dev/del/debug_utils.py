"""
调试工具模块
提供各种调试辅助功能
"""
import os
import sys
import time
import inspect
import traceback
import logging
import warnings
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from functools import wraps
import torch
import numpy as np

from src.utils.logging_config import get_logger, LOG_LEVELS

logger = get_logger("vbench.debug")

# 全局调试配置
DEBUG_CONFIG = {
    "enabled": False,
    "verbose": False,
    "save_artifacts": True,
    "artifacts_dir": None,
    "profile_enabled": False
}

# 尝试导入可选依赖
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import torch.autograd.profiler as profiler
    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False

try:
    from IPython.core.display import display, HTML
    from IPython import get_ipython
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


def trace_execution(func: Callable) -> Callable:
    """
    跟踪函数执行的装饰器
    记录函数调用、参数、返回值和执行时间

    Args:
        func: 要跟踪的函数

    Returns:
        包装后的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 获取调用信息
        caller_frame = inspect.currentframe().f_back
        caller_info = inspect.getframeinfo(caller_frame)
        
        # 记录函数调用
        logger.debug(f"TRACE: 调用 {func.__name__} 从 {caller_info.filename}:{caller_info.lineno}")
        
        # 记录参数 (过滤掉过大的参数)
        debug_args = [
            str(arg) if sys.getsizeof(arg) < 1000 else f"<大型对象: {type(arg).__name__}>" 
            for arg in args
        ]
        debug_kwargs = {
            k: (str(v) if sys.getsizeof(v) < 1000 else f"<大型对象: {type(v).__name__}>") 
            for k, v in kwargs.items()
        }
        logger.debug(f"TRACE: 参数: args={debug_args}, kwargs={debug_kwargs}")
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 执行原始函数
            result = func(*args, **kwargs)
            
            # 记录执行时间
            elapsed = time.time() - start_time
            logger.debug(f"TRACE: {func.__name__} 执行完成，耗时 {elapsed:.4f}秒")
            
            # 记录返回值 (过滤掉过大的返回值)
            if result is not None:
                if sys.getsizeof(result) < 1000:
                    logger.debug(f"TRACE: 返回值: {result}")
                else:
                    logger.debug(f"TRACE: 返回值: <大型对象: {type(result).__name__}>")
            
            return result
        
        except Exception as e:
            # 记录异常
            logger.error(f"TRACE: 在执行 {func.__name__} 时发生异常: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    return wrapper


def memory_profile(func: Callable) -> Callable:
    """
    内存分析装饰器
    跟踪函数执行前后的内存使用情况

    Args:
        func: 要分析的函数

    Returns:
        包装后的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 记录 GPU 内存情况（如果可用）
        gpu_mem_before = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_mem_before[i] = torch.cuda.memory_allocated(i) / (1024**2)
        
        # 执行函数
        result = func(*args, **kwargs)
        
        # 记录 GPU 内存情况（如果可用）
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_mem_after = torch.cuda.memory_allocated(i) / (1024**2)
                gpu_mem_diff = gpu_mem_after - gpu_mem_before[i]
                logger.debug(f"MEM: GPU:{i} 内存使用变化: {gpu_mem_diff:.2f} MB, 当前: {gpu_mem_after:.2f} MB")
        
        return result
    
    return wrapper


def inspect_tensor(tensor: Union[torch.Tensor, np.ndarray], name: str = "tensor") -> Dict[str, Any]:
    """
    检查张量的基本属性和统计信息

    Args:
        tensor: 要检查的张量
        name: 张量名称

    Returns:
        包含张量信息的字典
    """
    info = {}
    
    # 基本信息
    if isinstance(tensor, torch.Tensor):
        info["类型"] = "PyTorch Tensor"
        info["形状"] = tensor.shape
        info["数据类型"] = tensor.dtype
        info["设备"] = tensor.device
        
        # 转换为 CPU 上的 NumPy 数组以进行统计分析
        if tensor.requires_grad:
            tensor_np = tensor.detach().cpu().numpy()
        else:
            tensor_np = tensor.cpu().numpy()
    
    elif isinstance(tensor, np.ndarray):
        info["类型"] = "NumPy Array"
        info["形状"] = tensor.shape
        info["数据类型"] = tensor.dtype
        tensor_np = tensor
    
    else:
        raise TypeError(f"不支持的类型: {type(tensor)}")
    
    # 统计信息
    info["最小值"] = float(np.min(tensor_np))
    info["最大值"] = float(np.max(tensor_np))
    info["均值"] = float(np.mean(tensor_np))
    info["标准差"] = float(np.std(tensor_np))
    
    # 检查是否包含 NaN 或 Inf 值
    info["NaN 数量"] = int(np.isnan(tensor_np).sum())
    info["Inf 数量"] = int(np.isinf(tensor_np).sum())
    
    # 打印信息到日志
    logger.debug(f"TENSOR[{name}]: {info}")
    
    return info


class ModelDebugger:
    """模型调试器，用于调试模型输入输出和梯度"""
    
    def __init__(self, model: torch.nn.Module, log_dir: Optional[str] = None):
        """
        初始化模型调试器

        Args:
            model: 要调试的模型
            log_dir: 日志目录，None则使用默认目录
        """
        self.model = model
        self.logger = get_logger("vbench.model_debugger")
        self.hooks = []
        
        # 设置日志目录
        if log_dir is None:
            self.log_dir = os.path.join(
                os.environ.get("VBENCH_HOME", os.path.abspath(".")),
                "logs", "model_debug"
            )
        else:
            self.log_dir = log_dir
        
        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
    
    def register_hooks(self):
        """注册各种钩子函数来收集信息"""
        # 保存各层输出
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # 只注册叶节点模块
                def hook_fn(module, input, output, name=name):
                    try:
                        self.logger.debug(f"Layer[{name}] output shape: {output.shape}")
                        # 统计信息
                        if torch.is_tensor(output):
                            inspect_tensor(output, f"layer_{name}")
                    except Exception as e:
                        self.logger.error(f"Error in hook for {name}: {e}")
                
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)
        
        # 注册梯度钩子
        if self.model.training:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    def grad_hook(grad, name=name):
                        try:
                            self.logger.debug(f"Grad[{name}] shape: {grad.shape}")
                            inspect_tensor(grad, f"grad_{name}")
                        except Exception as e:
                            self.logger.error(f"Error in grad hook for {name}: {e}")
                    
                    hook = param.register_hook(grad_hook)
                    self.hooks.append(hook)
        
        return self
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        return self
    
    def __enter__(self):
        """上下文管理器入口"""
        self.register_hooks()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.remove_hooks()
        return False  # 不抑制异常


def debug_data_batch(data_batch: Any, logger_name: str = "vbench.data_debug") -> Dict[str, Any]:
    """
    调试数据批次，打印结构和统计信息

    Args:
        data_batch: 数据批次，可以是各种类型
        logger_name: 日志记录器名称

    Returns:
        包含数据批次信息的字典
    """
    logger = get_logger(logger_name)
    info = {}
    
    # 处理字典类型
    if isinstance(data_batch, dict):
        logger.debug("数据批次为字典类型")
        info["类型"] = "字典"
        info["键"] = list(data_batch.keys())
        
        # 处理每个键值对
        for key, value in data_batch.items():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                logger.debug(f"键 '{key}' 是张量")
                info[f"'{key}'"] = inspect_tensor(value, key)
            else:
                try:
                    logger.debug(f"键 '{key}' 是 {type(value).__name__}")
                    info[f"'{key}' 类型"] = type(value).__name__
                    if hasattr(value, "shape"):
                        info[f"'{key}' 形状"] = value.shape
                except Exception as e:
                    logger.error(f"处理键 '{key}' 时出错: {e}")
    
    # 处理列表或元组类型
    elif isinstance(data_batch, (list, tuple)):
        container_type = "列表" if isinstance(data_batch, list) else "元组"
        logger.debug(f"数据批次为{container_type}类型")
        info["类型"] = container_type
        info["长度"] = len(data_batch)
        
        # 处理前几个元素
        for i, item in enumerate(data_batch[:5]):  # 只检查前5个元素
            if isinstance(item, (torch.Tensor, np.ndarray)):
                logger.debug(f"索引 {i} 是张量")
                info[f"[{i}]"] = inspect_tensor(item, f"item_{i}")
            else:
                try:
                    logger.debug(f"索引 {i} 是 {type(item).__name__}")
                    info[f"[{i}] 类型"] = type(item).__name__
                    if hasattr(item, "shape"):
                        info[f"[{i}] 形状"] = item.shape
                except Exception as e:
                    logger.error(f"处理索引 {i} 时出错: {e}")
    
    # 处理张量类型
    elif isinstance(data_batch, (torch.Tensor, np.ndarray)):
        logger.debug("数据批次为张量类型")
        info = inspect_tensor(data_batch, "data_batch")
    
    # 其他类型
    else:
        logger.debug(f"数据批次为其他类型: {type(data_batch).__name__}")
        info["类型"] = type(data_batch).__name__
        try:
            info["字符串表示"] = str(data_batch)[:100]  # 截断过长的字符串
        except Exception as e:
            logger.error(f"处理数据批次时出错: {e}")
    
    return info


def setup_debug_mode(log_level: str = "debug", save_dir: Optional[str] = None,
                    enable_profiling: bool = False, verbose: bool = False):
    """
    设置调试模式，包括：
    1. 设置详细日志级别
    2. 启用 PyTorch 异常详细信息
    3. 禁用 CUDA 异步执行 (如果可用)
    4. 配置调试工件保存路径
    5. 启用性能分析器(可选)

    Args:
        log_level: 日志级别，默认为debug
        save_dir: 保存调试工件的目录，None则使用默认目录
        enable_profiling: 是否启用性能分析
        verbose: 是否启用详细输出
    """
    global DEBUG_CONFIG
    
    # 设置详细日志级别
    logger = get_logger("vbench")
    log_level = log_level.lower()
    if log_level in LOG_LEVELS:
        logger.setLevel(LOG_LEVELS[log_level])
    else:
        logger.setLevel(logging.DEBUG)
        logger.warning(f"未知的日志级别: {log_level}，使用DEBUG级别")
    
    # 启用 PyTorch 详细异常
    torch.autograd.set_detect_anomaly(True)
    
    # 如果有 CUDA，则禁用异步执行以便于调试
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("CUDA异步执行已禁用，以便调试")
    
    # 配置调试工件保存路径
    if save_dir is None:
        save_dir = os.path.join(
            os.environ.get("VBENCH_HOME", os.path.abspath(".")),
            "logs", "debug_artifacts", 
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 更新全局调试配置
    DEBUG_CONFIG.update({
        "enabled": True,
        "verbose": verbose,
        "save_artifacts": True,
        "artifacts_dir": save_dir,
        "profile_enabled": enable_profiling and HAS_PROFILER
    })
    
    # 启用Python警告
    warnings.filterwarnings('always')
    
    # 记录调试模式已启用
    logger.info(f"调试模式已启用: 日志级别={log_level}, 工件保存目录={save_dir}, 性能分析={enable_profiling}")
    logger.info(f"调试工具可用性: matplotlib={HAS_MATPLOTLIB}, profiler={HAS_PROFILER}, ipython={HAS_IPYTHON}")


def check_nan_gradients(model: torch.nn.Module) -> bool:
    """
    检查模型是否有 NaN 梯度

    Args:
        model: 要检查的模型

    Returns:
        如果发现 NaN 梯度则返回 True，否则返回 False
    """
    has_nan = False
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                logger.error(f"参数 {name} 的梯度包含 NaN 值")
                has_nan = True
            if torch.isinf(param.grad).any():
                logger.error(f"参数 {name} 的梯度包含 Inf 值")
                has_nan = True
    
    return has_nan