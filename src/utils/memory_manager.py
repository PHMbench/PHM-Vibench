"""
内存管理工具模块

提供统一的内存清理功能，包括GPU缓存清理、Python垃圾回收和系统追踪器清理。
适用于深度学习训练过程中的内存优化，防止内存泄漏和累积。

Usage:
    from src.utils.memory_manager import clear_all_memory
    
    # 在epoch结束时调用
    clear_all_memory()
    
    # 或分别调用
    clear_gpu_memory()
    clear_python_memory()
"""

import gc
import torch
from typing import Optional, Any
import warnings


def clear_gpu_memory(synchronize: bool = True, verbose: bool = False) -> None:
    """
    清理GPU内存缓存
    
    Args:
        synchronize (bool): 是否等待CUDA操作完成，默认True
        verbose (bool): 是否打印内存清理信息，默认False
    """
    if not torch.cuda.is_available():
        return
        
    try:
        # 获取清理前的内存使用情况（如果需要打印）
        if verbose:
            allocated_before = torch.cuda.memory_allocated()
            cached_before = torch.cuda.memory_reserved()
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        # 等待CUDA操作完成
        if synchronize:
            torch.cuda.synchronize()
        
        if verbose:
            allocated_after = torch.cuda.memory_allocated()
            cached_after = torch.cuda.memory_reserved()
            freed_allocated = (allocated_before - allocated_after) / 1024**2  # MB
            freed_cached = (cached_before - cached_after) / 1024**2  # MB
            
            print(f"GPU Memory cleared - Allocated: {freed_allocated:.1f}MB, "
                  f"Cached: {freed_cached:.1f}MB freed")
                  
    except Exception as e:
        warnings.warn(f"Failed to clear GPU memory: {e}", UserWarning)


def clear_python_memory(verbose: bool = False) -> None:
    """
    执行Python垃圾回收
    
    Args:
        verbose (bool): 是否打印垃圾回收统计，默认False
    """
    try:
        # 执行垃圾回收
        collected = gc.collect()
        
        if verbose:
            print(f"Python GC collected {collected} objects")
            
    except Exception as e:
        warnings.warn(f"Failed to run Python garbage collection: {e}", UserWarning)


def clear_tracker_memory(tracker: Any) -> None:
    """
    清理追踪器内存（如SystemMetricsTracker）
    
    Args:
        tracker: 具有clear()方法的追踪器对象
    """
    try:
        if hasattr(tracker, 'clear') and callable(getattr(tracker, 'clear')):
            tracker.clear()
    except Exception as e:
        warnings.warn(f"Failed to clear tracker memory: {e}", UserWarning)


def clear_all_memory(trackers: Optional[list] = None, 
                    synchronize: bool = True, 
                    verbose: bool = False) -> None:
    """
    执行完整的内存清理：GPU缓存 + Python垃圾回收 + 追踪器清理
    
    Args:
        trackers (list, optional): 需要清理的追踪器列表
        synchronize (bool): 是否等待CUDA操作完成，默认True
        verbose (bool): 是否打印详细清理信息，默认False
    """
    if verbose:
        print("Starting comprehensive memory cleanup...")
    
    # 清理GPU内存
    clear_gpu_memory(synchronize=synchronize, verbose=verbose)
    
    # 清理Python内存
    clear_python_memory(verbose=verbose)
    
    # 清理追踪器内存
    if trackers:
        for tracker in trackers:
            if tracker is not None:
                clear_tracker_memory(tracker)
                if verbose:
                    print(f"Cleared tracker: {type(tracker).__name__}")
    
    if verbose:
        print("Memory cleanup completed")


def get_memory_info() -> dict:
    """
    获取当前内存使用信息
    
    Returns:
        dict: 包含GPU和CPU内存信息的字典
    """
    info = {
        'gpu_available': torch.cuda.is_available(),
        'cpu_memory_objects': len(gc.get_objects())
    }
    
    if torch.cuda.is_available():
        info.update({
            'gpu_allocated_mb': torch.cuda.memory_allocated() / 1024**2,
            'gpu_reserved_mb': torch.cuda.memory_reserved() / 1024**2,
            'gpu_max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
            'gpu_device_count': torch.cuda.device_count(),
            'gpu_current_device': torch.cuda.current_device()
        })
    
    return info


def memory_cleanup_context(trackers: Optional[list] = None, 
                          verbose: bool = False):
    """
    内存清理上下文管理器
    
    Usage:
        with memory_cleanup_context([tracker1, tracker2]):
            # 训练代码
            pass
        # 自动清理内存
    """
    class MemoryCleanupContext:
        def __init__(self, trackers, verbose):
            self.trackers = trackers
            self.verbose = verbose
            
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            clear_all_memory(trackers=self.trackers, verbose=self.verbose)
    
    return MemoryCleanupContext(trackers, verbose)


if __name__ == '__main__':
    """测试内存管理功能"""
    print("=== Memory Manager Test ===")
    
    # 获取初始内存信息
    print("Initial memory info:")
    initial_info = get_memory_info()
    for key, value in initial_info.items():
        print(f"  {key}: {value}")
    
    # 创建一些测试数据
    if torch.cuda.is_available():
        test_tensor = torch.randn(1000, 1000).cuda()
        print(f"\nCreated test tensor: {test_tensor.shape}")
    
    # 执行内存清理
    print("\nExecuting memory cleanup...")
    clear_all_memory(verbose=True)
    
    # 获取清理后内存信息
    print("\nFinal memory info:")
    final_info = get_memory_info()
    for key, value in final_info.items():
        print(f"  {key}: {value}")
    
    # 测试上下文管理器
    print("\nTesting context manager...")
    with memory_cleanup_context(verbose=True):
        if torch.cuda.is_available():
            temp_tensor = torch.randn(500, 500).cuda()
            print(f"Created temp tensor: {temp_tensor.shape}")
    
    print("✓ Memory manager test completed!")