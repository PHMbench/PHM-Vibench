"""
Flow自测试基础设施使用示例 (Flow Self-Testing Infrastructure Usage Example)

这个文件展示了如何使用自测试基础设施为Flow模块创建自测试。

Author: PHM-Vibench Team
Date: 2025-09-05
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from . import (
    ValidationResult,
    TestConfiguration,
    PerformanceMetrics,
    SelfTestOrchestrator,
    ResourceManager
)


class ExampleFlowModule(nn.Module):
    """示例Flow模块 (Example Flow module)"""
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def create_flow_module_orchestrator() -> SelfTestOrchestrator:
    """
    创建Flow模块测试编排器 (Create Flow module test orchestrator)
    
    Returns:
        配置好的测试编排器
    """
    orchestrator = SelfTestOrchestrator(
        name="ExampleFlowModuleTest",
        default_config=TestConfiguration(
            test_name="flow_module_default",
            batch_size=8,
            sequence_length=64,
            input_dim=3,
            timeout_seconds=30.0
        )
    )
    
    # 注册测试函数
    orchestrator.register_test("forward_pass", test_forward_pass)
    orchestrator.register_test("backward_pass", test_backward_pass)
    orchestrator.register_test("output_shape", test_output_shape)
    orchestrator.register_test("device_compatibility", test_device_compatibility)
    orchestrator.register_test("memory_efficiency", test_memory_efficiency)
    
    return orchestrator


def test_forward_pass(config: TestConfiguration) -> Dict[str, Any]:
    """测试前向传播 (Test forward pass)"""
    model = ExampleFlowModule(input_dim=config.input_dim)
    model.to(config.device)
    model.eval()
    
    # 创建测试输入
    x = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.input_dim,
        device=config.device
    )
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    # 验证输出
    assert output.shape == x.shape, f"输出形状不匹配: {output.shape} vs {x.shape}"
    assert torch.isfinite(output).all(), "输出包含NaN或Inf值"
    
    return {
        "input_shape": x.shape,
        "output_shape": output.shape,
        "output_mean": output.mean().item(),
        "output_std": output.std().item()
    }


def test_backward_pass(config: TestConfiguration) -> Dict[str, Any]:
    """测试反向传播 (Test backward pass)"""
    model = ExampleFlowModule(input_dim=config.input_dim)
    model.to(config.device)
    model.train()
    
    # 创建测试输入和目标
    x = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.input_dim,
        device=config.device
    )
    target = torch.randn_like(x)
    
    # 前向传播
    output = model(x)
    loss = nn.MSELoss()(output, target)
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            assert torch.isfinite(param.grad).all(), f"参数 {name} 的梯度包含NaN或Inf"
        else:
            raise AssertionError(f"参数 {name} 没有梯度")
    
    return {
        "loss": loss.item(),
        "grad_norms": grad_norms,
        "avg_grad_norm": sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    }


def test_output_shape(config: TestConfiguration) -> Dict[str, Any]:
    """测试输出形状 (Test output shape)"""
    model = ExampleFlowModule(input_dim=config.input_dim)
    model.to(config.device)
    
    # 测试不同的输入形状
    test_cases = [
        (config.batch_size, config.sequence_length, config.input_dim),
        (1, config.sequence_length, config.input_dim),
        (config.batch_size, 32, config.input_dim),
        (config.batch_size, 128, config.input_dim),
    ]
    
    results = {}
    for i, shape in enumerate(test_cases):
        x = torch.randn(*shape, device=config.device)
        output = model(x)
        assert output.shape == x.shape, f"测试用例 {i}: 形状不匹配"
        results[f"test_case_{i}"] = {
            "input_shape": shape,
            "output_shape": output.shape
        }
    
    return results


def test_device_compatibility(config: TestConfiguration) -> Dict[str, Any]:
    """测试设备兼容性 (Test device compatibility)"""
    results = {}
    
    # 测试CPU
    model_cpu = ExampleFlowModule(input_dim=config.input_dim)
    x_cpu = torch.randn(config.batch_size, config.sequence_length, config.input_dim)
    output_cpu = model_cpu(x_cpu)
    results["cpu_test"] = {
        "success": True,
        "output_device": str(output_cpu.device)
    }
    
    # 测试GPU（如果可用）
    if torch.cuda.is_available():
        model_gpu = ExampleFlowModule(input_dim=config.input_dim).cuda()
        x_gpu = torch.randn(config.batch_size, config.sequence_length, config.input_dim).cuda()
        output_gpu = model_gpu(x_gpu)
        results["gpu_test"] = {
            "success": True,
            "output_device": str(output_gpu.device)
        }
    else:
        results["gpu_test"] = {"success": False, "reason": "CUDA不可用"}
    
    return results


def test_memory_efficiency(config: TestConfiguration) -> Dict[str, Any]:
    """测试内存效率 (Test memory efficiency)"""
    if not torch.cuda.is_available():
        return {"success": False, "reason": "需要GPU进行内存测试"}
    
    # 清理GPU内存
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()
    
    model = ExampleFlowModule(input_dim=config.input_dim).cuda()
    model_memory = torch.cuda.memory_allocated() - initial_memory
    
    x = torch.randn(
        config.batch_size,
        config.sequence_length, 
        config.input_dim
    ).cuda()
    data_memory = torch.cuda.memory_allocated() - initial_memory - model_memory
    
    # 前向传播
    output = model(x)
    forward_memory = torch.cuda.memory_allocated() - initial_memory - model_memory - data_memory
    
    # 反向传播
    loss = output.sum()
    loss.backward()
    backward_memory = torch.cuda.memory_allocated() - initial_memory - model_memory - data_memory - forward_memory
    
    return {
        "model_memory_mb": model_memory / 1024 / 1024,
        "data_memory_mb": data_memory / 1024 / 1024,
        "forward_memory_mb": forward_memory / 1024 / 1024,
        "backward_memory_mb": backward_memory / 1024 / 1024,
        "total_memory_mb": torch.cuda.memory_allocated() / 1024 / 1024
    }


def run_example():
    """运行示例测试 (Run example tests)"""
    print("=" * 60)
    print("Flow自测试基础设施使用示例")
    print("Flow Self-Testing Infrastructure Usage Example")
    print("=" * 60)
    
    # 创建测试编排器
    orchestrator = create_flow_module_orchestrator()
    
    # 运行所有测试
    results = orchestrator.run_test_suite()
    
    # 打印结果
    print("\n测试结果 (Test Results):")
    print("-" * 40)
    for result in results:
        status = "✅" if result.success else "❌"
        print(f"{status} {result.test_name}: {result.execution_time:.3f}s")
        if not result.success:
            print(f"    错误: {result.error_message}")
    
    # 打印总结
    summary = orchestrator.get_summary()
    print(f"\n总结 (Summary):")
    print(f"  总测试数: {summary['total_tests']}")
    print(f"  通过: {summary['passed_tests']}")
    print(f"  失败: {summary['failed_tests']}")
    print(f"  成功率: {summary['success_rate']:.2%}")
    print(f"  总耗时: {summary['total_execution_time']:.3f}s")
    
    return summary['success_rate'] == 1.0


if __name__ == "__main__":
    success = run_example()
    if not success:
        exit(1)