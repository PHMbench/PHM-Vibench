"""
Flow自测试基础设施 (Flow Self-Testing Infrastructure)

这个模块提供了Flow预训练模块自测试的基础设施类，包括结果验证、配置管理、性能指标、
测试编排和资源管理。遵循PHM-Vibench的工厂模式和PyTorch Lightning约定。

Author: PHM-Vibench Team
Date: 2025-09-05
"""

import torch
import time
import tempfile
import shutil
import os
import gc
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
import signal
import logging

# Configure logging for self-testing
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    验证结果数据类 (Validation Result Data Class)
    
    存储单个测试的验证结果，包括状态、指标、错误信息等。
    """
    test_name: str
    success: bool
    execution_time: float
    memory_usage: Optional[float] = None
    gpu_memory_usage: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.execution_time < 0:
            raise ValueError("执行时间不能为负数 (Execution time cannot be negative)")
        if self.memory_usage is not None and self.memory_usage < 0:
            raise ValueError("内存使用量不能为负数 (Memory usage cannot be negative)")


@dataclass 
class TestConfiguration:
    """
    测试配置数据类 (Test Configuration Data Class)
    
    定义测试参数和设置，支持灵活的测试场景配置。
    """
    test_name: str
    device: Union[str, torch.device] = "auto"
    batch_size: int = 8
    sequence_length: int = 64
    input_dim: int = 3
    timeout_seconds: float = 30.0
    memory_limit_mb: Optional[float] = None
    gpu_memory_limit_mb: Optional[float] = None
    tolerance: float = 1e-6
    enable_performance_tracking: bool = True
    enable_memory_tracking: bool = True
    random_seed: int = 42
    test_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization setup."""
        if isinstance(self.device, str) and self.device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(self.device, str):
            self.device = torch.device(self.device)
        
        if self.timeout_seconds <= 0:
            raise ValueError("超时时间必须大于0 (Timeout must be greater than 0)")


@dataclass
class PerformanceMetrics:
    """
    性能指标数据类 (Performance Metrics Data Class)
    
    跟踪和存储测试过程中的性能指标，用于性能分析和优化。
    """
    forward_time: Optional[float] = None
    backward_time: Optional[float] = None
    total_time: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    peak_gpu_memory_mb: Optional[float] = None
    throughput_samples_per_sec: Optional[float] = None
    model_parameters: Optional[int] = None
    model_size_mb: Optional[float] = None
    flops: Optional[int] = None
    additional_metrics: Dict[str, float] = field(default_factory=dict)
    
    def compute_derived_metrics(self, batch_size: int = None):
        """计算派生指标 (Compute derived metrics)."""
        if self.total_time and batch_size:
            self.throughput_samples_per_sec = batch_size / self.total_time
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式 (Convert to dictionary format)."""
        result = {}
        for field_name, value in self.__dict__.items():
            if value is not None:
                result[field_name] = value
        return result


class TimeoutError(Exception):
    """测试超时异常 (Test timeout exception)."""
    pass


class SelfTestOrchestrator:
    """
    自测试编排器基类 (Self-Test Orchestrator Base Class)
    
    提供测试执行的统一框架，包括超时管理、结果聚合和错误处理。
    遵循PHM-Vibench的测试模式，支持30秒超时管理。
    """
    
    def __init__(
        self,
        name: str,
        default_config: Optional[TestConfiguration] = None,
        resource_manager: Optional['ResourceManager'] = None
    ):
        """
        初始化测试编排器
        
        Args:
            name: 测试编排器名称
            default_config: 默认测试配置
            resource_manager: 资源管理器实例
        """
        self.name = name
        self.default_config = default_config or TestConfiguration(test_name=name)
        self.resource_manager = resource_manager or ResourceManager()
        self.results: List[ValidationResult] = []
        self.test_registry: Dict[str, Callable] = {}
        
    def register_test(self, test_name: str, test_func: Callable):
        """
        注册测试函数 (Register test function)
        
        Args:
            test_name: 测试名称
            test_func: 测试函数，应接受TestConfiguration作为参数
        """
        self.test_registry[test_name] = test_func
        logger.info(f"注册测试: {test_name} (Registered test: {test_name})")
    
    def run_single_test(
        self,
        test_name: str,
        config: Optional[TestConfiguration] = None
    ) -> ValidationResult:
        """
        运行单个测试，包含超时管理 (Run single test with timeout management)
        
        Args:
            test_name: 测试名称
            config: 测试配置，如果为None则使用默认配置
            
        Returns:
            ValidationResult: 测试结果
        """
        if test_name not in self.test_registry:
            return ValidationResult(
                test_name=test_name,
                success=False,
                execution_time=0.0,
                error_message=f"测试未注册: {test_name} (Test not registered: {test_name})"
            )
        
        test_config = config or self.default_config
        test_func = self.test_registry[test_name]
        
        # 设置随机种子确保可重现性
        torch.manual_seed(test_config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(test_config.random_seed)
        
        start_time = time.time()
        
        try:
            with self.resource_manager.managed_execution():
                result = self._run_with_timeout(test_func, test_config)
                
            execution_time = time.time() - start_time
            
            # 收集性能指标
            performance_metrics = PerformanceMetrics(total_time=execution_time)
            if test_config.enable_memory_tracking:
                self._collect_memory_metrics(performance_metrics)
            
            return ValidationResult(
                test_name=test_name,
                success=True,
                execution_time=execution_time,
                memory_usage=performance_metrics.peak_memory_mb,
                gpu_memory_usage=performance_metrics.peak_gpu_memory_mb,
                metrics=performance_metrics.to_dict(),
                metadata={"config": test_config.test_parameters}
            )
            
        except TimeoutError as e:
            return ValidationResult(
                test_name=test_name,
                success=False,
                execution_time=time.time() - start_time,
                error_message=f"测试超时: {str(e)} (Test timeout: {str(e)})"
            )
            
        except Exception as e:
            return ValidationResult(
                test_name=test_name,
                success=False,
                execution_time=time.time() - start_time,
                error_message=f"测试执行失败: {str(e)} (Test execution failed: {str(e)})"
            )
    
    def _run_with_timeout(self, test_func: Callable, config: TestConfiguration):
        """
        在指定超时时间内运行测试函数 (Run test function within timeout)
        
        Args:
            test_func: 要执行的测试函数
            config: 测试配置
            
        Returns:
            测试函数的返回结果
            
        Raises:
            TimeoutError: 如果测试超时
        """
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = test_func(config)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=config.timeout_seconds)
        
        if thread.is_alive():
            # 尝试清理资源
            self.resource_manager.cleanup()
            raise TimeoutError(f"测试在{config.timeout_seconds}秒后超时 (Test timed out after {config.timeout_seconds} seconds)")
        
        if exception[0]:
            raise exception[0]
            
        return result[0]
    
    def run_test_suite(
        self,
        test_names: Optional[List[str]] = None,
        config: Optional[TestConfiguration] = None
    ) -> List[ValidationResult]:
        """
        运行测试套件 (Run test suite)
        
        Args:
            test_names: 要运行的测试名称列表，如果为None则运行所有注册的测试
            config: 测试配置
            
        Returns:
            List[ValidationResult]: 所有测试的结果列表
        """
        if test_names is None:
            test_names = list(self.test_registry.keys())
        
        results = []
        for test_name in test_names:
            logger.info(f"运行测试: {test_name} (Running test: {test_name})")
            result = self.run_single_test(test_name, config)
            results.append(result)
            self.results.append(result)
            
            # 在测试之间清理资源
            self.resource_manager.cleanup()
        
        return results
    
    def _collect_memory_metrics(self, metrics: PerformanceMetrics):
        """收集内存使用指标 (Collect memory usage metrics)."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            metrics.peak_memory_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            logger.warning("psutil未安装，无法收集内存指标 (psutil not installed, cannot collect memory metrics)")
        
        if torch.cuda.is_available():
            metrics.peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取测试总结 (Get test summary)
        
        Returns:
            包含测试统计信息的字典
        """
        if not self.results:
            return {"total_tests": 0, "message": "没有运行的测试 (No tests run)"}
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        total_time = sum(r.execution_time for r in self.results)
        
        return {
            "orchestrator_name": self.name,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            "total_execution_time": total_time,
            "average_test_time": total_time / total_tests if total_tests > 0 else 0.0,
            "failed_test_names": [r.test_name for r in self.results if not r.success]
        }


class ResourceManager:
    """
    资源管理器类 (Resource Manager Class)
    
    负责适当的资源清理，包括GPU内存、临时文件和设备管理。
    遵循现有的设备管理模式，参考test/conftest.py的ModelTestHelper。
    """
    
    def __init__(self):
        """初始化资源管理器."""
        self.temp_dirs: List[str] = []
        self.temp_files: List[str] = []
        self.cuda_contexts: List[Any] = []
        
    @contextmanager
    def managed_execution(self):
        """
        管理执行上下文 (Managed execution context)
        
        确保在测试执行前后正确管理资源。
        """
        try:
            # 执行前清理
            self.cleanup()
            yield
        finally:
            # 执行后清理
            self.cleanup()
    
    def cleanup(self):
        """
        执行完整的资源清理 (Perform comprehensive resource cleanup)
        
        按照test/conftest.py的模式进行GPU内存和资源清理。
        """
        try:
            # 清理GPU内存，遵循test/conftest.py模式
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # 强制同步以确保清理完成
                torch.cuda.synchronize()
            
            # 清理临时文件
            self._cleanup_temp_files()
            
            # 清理临时目录
            self._cleanup_temp_dirs()
            
            # Python垃圾回收
            gc.collect()
            
        except Exception as e:
            logger.warning(f"资源清理过程中发生错误: {e} (Error during resource cleanup: {e})")
    
    def _cleanup_temp_files(self):
        """清理临时文件 (Clean up temporary files)."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"无法删除临时文件 {temp_file}: {e} (Cannot remove temp file {temp_file}: {e})")
        self.temp_files.clear()
    
    def _cleanup_temp_dirs(self):
        """清理临时目录 (Clean up temporary directories)."""
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"无法删除临时目录 {temp_dir}: {e} (Cannot remove temp dir {temp_dir}: {e})")
        self.temp_dirs.clear()
    
    def create_temp_file(self, suffix: str = "", prefix: str = "phm_test_") -> str:
        """
        创建临时文件 (Create temporary file)
        
        Args:
            suffix: 文件后缀
            prefix: 文件前缀
            
        Returns:
            str: 临时文件路径
        """
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        os.close(fd)  # 关闭文件描述符，但保留文件
        self.temp_files.append(temp_path)
        return temp_path
    
    def create_temp_dir(self, prefix: str = "phm_test_") -> str:
        """
        创建临时目录 (Create temporary directory)
        
        Args:
            prefix: 目录前缀
            
        Returns:
            str: 临时目录路径
        """
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        获取设备信息 (Get device information)
        
        Returns:
            包含设备信息的字典
        """
        info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated": torch.cuda.memory_allocated() / 1024 / 1024,  # MB
                "memory_reserved": torch.cuda.memory_reserved() / 1024 / 1024,    # MB
            })
        
        return info
    
    def monitor_memory_usage(self) -> Dict[str, float]:
        """
        监控当前内存使用情况 (Monitor current memory usage)
        
        Returns:
            包含内存使用信息的字典(单位: MB)
        """
        memory_info = {}
        
        # 系统内存
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info["system_memory_mb"] = process.memory_info().rss / 1024 / 1024
        except ImportError:
            logger.warning("psutil未安装，无法监控系统内存 (psutil not installed, cannot monitor system memory)")
        
        # GPU内存
        if torch.cuda.is_available():
            memory_info["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            memory_info["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        
        return memory_info


# 导出的类和函数
__all__ = [
    'ValidationResult',
    'TestConfiguration', 
    'PerformanceMetrics',
    'SelfTestOrchestrator',
    'ResourceManager',
    'TimeoutError',
]