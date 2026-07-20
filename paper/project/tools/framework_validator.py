"""
Neural-Symbolic Framework Validator
神经-符号框架验证器

本工具用于验证模型是否符合四层神经-符号架构，
并提供架构合规性评估。
"""

import torch
import torch.nn as nn
import inspect
import ast
import re
from typing import Dict, List, Tuple, Optional, Any
import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


class LayerType:
    """层类型枚举"""
    SIGNAL_PROCESSING = "signal_processing"
    FEATURE_EXTRACTION = "feature_extraction"
    SYMBOLIC_REASONING = "symbolic_reasoning"
    LINGUISTIC_EXPLANATION = "linguistic_explanation"


class ComponentValidator:
    """组件验证器基类"""

    def __init__(self, layer_type: str):
        self.layer_type = layer_type

    def validate(self, model: nn.Module, component: Any) -> Dict[str, Any]:
        """验证组件是否符合层类型要求"""
        raise NotImplementedError


class SignalProcessingValidator(ComponentValidator):
    """信号处理层验证器"""

    def __init__(self):
        super().__init__(LayerType.SIGNAL_PROCESSING)
        self.expected_operations = ['FFT', 'HT', 'WF', 'LNO', 'Conv', 'Linear', 'Identity']
        self.expected_properties = ['input_shape', 'output_shape', 'operation_type']

    def validate(self, model: nn.Module, component: Any) -> Dict[str, Any]:
        """验证信号处理组件"""
        validation_result = {
            'layer_type': self.layer_type,
            'component_name': component.__class__.__name__,
            'is_valid': False,
            'issues': [],
            'properties': {}
        }

        # 检查是否有明确的信号处理操作
        if hasattr(component, 'operation'):
            if component.operation in self.expected_operations:
                validation_result['properties']['operation_type'] = component.operation
            else:
                validation_result['issues'].append(f"未知操作类型: {component.operation}")

        # 检查是否是PyTorch层
        if isinstance(component, nn.Module):
            validation_result['properties']['is_pytorch_layer'] = True

            # 分析参数
            total_params = sum(p.numel() for p in component.parameters())
            validation_result['properties']['parameter_count'] = total_params

        # 检查命名模式
        name = component.__class__.__name__.lower()
        for op in self.expected_operations:
            if op.lower() in name:
                validation_result['properties']['inferred_operation'] = op
                break

        # 如果找到至少一个信号处理特征，认为有效
        validation_result['is_valid'] = len(validation_result['properties']) > 0

        return validation_result


class FeatureExtractionValidator(ComponentValidator):
    """特征提取层验证器"""

    def __init__(self):
        super().__init__(LayerType.FEATURE_EXTRACTION)
        self.expected_methods = ['forward', 'extract_features']
        self.expected_features = ['statistical', 'frequency', 'temporal', 'attention']

    def validate(self, model: nn.Module, component: Any) -> Dict[str, Any]:
        """验证特征提取组件"""
        validation_result = {
            'layer_type': self.layer_type,
            'component_name': component.__class__.__name__,
            'is_valid': False,
            'issues': [],
            'properties': {}
        }

        # 检查方法
        methods = [method for method in dir(component) if not method.startswith('_')]
        validation_result['properties']['methods'] = methods

        # 查找特征相关的方法
        feature_methods = [m for m in methods if 'feature' in m.lower()]
        if feature_methods:
            validation_result['properties']['feature_methods'] = feature_methods

        # 检查是否有统计特征相关
        name = component.__class__.__name__.lower()
        if any(feat in name for feat in ['statistic', 'feature', 'extractor']):
            validation_result['properties']['feature_type'] = 'statistical'

        # 检查输出维度
        if hasattr(component, 'output_dim'):
            validation_result['properties']['output_dim'] = component.output_dim

        # 有效性判断
        validation_result['is_valid'] = (
            len(feature_methods) > 0 or
            'feature_type' in validation_result['properties'] or
            'extractor' in name
        )

        return validation_result


class SymbolicReasoningValidator(ComponentValidator):
    """符号推理层验证器"""

    def __init__(self):
        super().__init__(LayerType.SYMBOLIC_REASONING)
        self.expected_components = ['rule', 'fuzzy', 'logic', 'moe', 'expert', 'attention']

    def validate(self, model: nn.Module, component: Any) -> Dict[str, Any]:
        """验证符号推理组件"""
        validation_result = {
            'layer_type': self.layer_type,
            'component_name': component.__class__.__name__,
            'is_valid': False,
            'issues': [],
            'properties': {}
        }

        name = component.__class__.__name__.lower()

        # 检查符号推理关键词
        found_components = []
        for comp in self.expected_components:
            if comp in name:
                found_components.append(comp)

        if found_components:
            validation_result['properties']['reasoning_type'] = found_components

        # 检查规则系统特征
        if hasattr(component, 'rules'):
            validation_result['properties']['has_rules'] = True
            if isinstance(component.rules, (list, tuple)):
                validation_result['properties']['rule_count'] = len(component.rules)

        # 检查模糊系统特征
        if any(keyword in name for keyword in ['fuzzy', 'membership']):
            validation_result['properties']['is_fuzzy_system'] = True

        # 检查MoE特征
        if any(keyword in name for keyword in ['moe', 'expert', 'mixture']):
            validation_result['properties']['is_moe'] = True
            if hasattr(component, 'num_experts'):
                validation_result['properties']['expert_count'] = component.num_experts

        # 有效性判断
        validation_result['is_valid'] = (
            len(found_components) > 0 or
            'has_rules' in validation_result['properties'] or
            'is_fuzzy_system' in validation_result['properties'] or
            'is_moe' in validation_result['properties']
        )

        return validation_result


class LinguisticExplanationValidator(ComponentValidator):
    """语言解释层验证器"""

    def __init__(self):
        super().__init__(LayerType.LINGUISTIC_EXPLANATION)
        self.expected_methods = ['explain', 'generate', 'interpret', 'text']

    def validate(self, model: nn.Module, component: Any) -> Dict[str, Any]:
        """验证语言解释组件"""
        validation_result = {
            'layer_type': self.layer_type,
            'component_name': component.__class__.__name__,
            'is_valid': False,
            'issues': [],
            'properties': {}
        }

        # 检查解释相关方法
        methods = [method for method in dir(component) if not method.startswith('_')]
        explanation_methods = []
        for method in self.expected_methods:
            if any(method in m.lower() for m in methods):
                explanation_methods.append(method)

        if explanation_methods:
            validation_result['properties']['explanation_methods'] = explanation_methods

        # 检查语言模型特征
        name = component.__class__.__name__.lower()
        if any(keyword in name for keyword in ['llm', 'language', 'text', 'explain']):
            validation_result['properties']['is_language_model'] = True

        # 检查模板系统
        if hasattr(component, 'templates'):
            validation_result['properties']['has_templates'] = True

        # 检查生成器特征
        if 'generator' in name or hasattr(component, 'generate'):
            validation_result['properties']['is_generator'] = True

        # 有效性判断
        validation_result['is_valid'] = (
            len(explanation_methods) > 0 or
            'is_language_model' in validation_result['properties'] or
            'has_templates' in validation_result['properties'] or
            'is_generator' in validation_result['properties']
        )

        return validation_result


class FrameworkValidator:
    """神经-符号框架验证器"""

    def __init__(self):
        self.validators = {
            LayerType.SIGNAL_PROCESSING: SignalProcessingValidator(),
            LayerType.FEATURE_EXTRACTION: FeatureExtractionValidator(),
            LayerType.SYMBOLIC_REASONING: SymbolicReasoningValidator(),
            LayerType.LINGUISTIC_EXPLANATION: LinguisticExplanationValidator()
        }

    def validate_model(self, model: nn.Module, input_sample: torch.Tensor) -> Dict[str, Any]:
        """
        验证模型是否符合四层架构

        Args:
            model: 待验证的模型
            input_sample: 输入样本用于前向传播

        Returns:
            validation_result: 验证结果
        """
        validation_result = {
            'model_name': model.__class__.__name__,
            'is_compliant': False,
            'layer_mapping': {},
            'validation_details': {},
            'completeness_score': 0.0,
            'consistency_score': 0.0,
            'issues': []
        }

        # 收集所有组件
        components = self._collect_components(model)

        # 验证每个组件
        layer_counts = defaultdict(int)
        for name, component in components.items():
            # 尝试每个验证器
            for layer_type, validator in self.validators.items():
                result = validator.validate(model, component)
                if result['is_valid']:
                    validation_result['validation_details'][name] = result
                    validation_result['layer_mapping'][name] = layer_type
                    layer_counts[layer_type] += 1
                    break
            else:
                # 未识别的组件
                validation_result['issues'].append(f"未识别的组件: {name}")

        # 计算完整性分数
        expected_layers = len(self.validators)
        found_layers = len(layer_counts)
        validation_result['completeness_score'] = found_layers / expected_layers

        # 计算一致性分数
        validation_result['consistency_score'] = self._calculate_consistency(
            model, input_sample, validation_result['layer_mapping']
        )

        # 总体合规性判断
        validation_result['is_compliant'] = (
            validation_result['completeness_score'] >= 0.75 and  # 至少75%的层
            validation_result['consistency_score'] >= 0.5       # 一致性分数至少50%
        )

        return validation_result

    def _collect_components(self, model: nn.Module) -> Dict[str, Any]:
        """收集模型的所有组件"""
        components = {}

        # 直接子模块
        for name, module in model.named_children():
            components[name] = module

        # 递归收集嵌套组件
        for name, module in model.named_modules():
            if name:  # 跳过根模块
                parts = name.split('.')
                if len(parts) > 2:  # 深度嵌套的模块也收集
                    components[name] = module

        return components

    def _calculate_consistency(self,
                             model: nn.Module,
                             input_sample: torch.Tensor,
                             layer_mapping: Dict[str, str]) -> float:
        """计算层间一致性分数"""
        try:
            model.eval()
            with torch.no_grad():
                # 执行前向传播并捕获中间结果
                layer_outputs = {}
                hooks = []

                def create_hook(name):
                    def hook(module, input, output):
                        layer_outputs[name] = output
                    return hook

                # 注册钩子
                for name, module in model.named_modules():
                    if name in layer_mapping:
                        hooks.append(module.register_forward_hook(create_hook(name)))

                # 前向传播
                _ = model(input_sample)

                # 移除钩子
                for hook in hooks:
                    hook.remove()

                # 检查维度一致性
                consistency_scores = []
                prev_shape = None
                prev_layer_type = None

                # 按预期顺序处理层
                expected_order = [
                    LayerType.SIGNAL_PROCESSING,
                    LayerType.FEATURE_EXTRACTION,
                    LayerType.SYMBOLIC_REASONING,
                    LayerType.LINGUISTIC_EXPLANATION
                ]

                for layer_type in expected_order:
                    layer_names = [n for n, t in layer_mapping.items() if t == layer_type]
                    for name in layer_names:
                        if name in layer_outputs:
                            output = layer_outputs[name]
                            if hasattr(output, 'shape'):
                                # 简单的维度检查
                                if prev_shape is not None:
                                    # 检查维度合理性（简化版）
                                    if prev_layer_type == LayerType.SIGNAL_PROCESSING and \
                                       layer_type == LayerType.FEATURE_EXTRACTION:
                                        # 信号->特征：应该降维
                                        if output.shape[-1] <= prev_shape[-1]:
                                            consistency_scores.append(1.0)
                                        else:
                                            consistency_scores.append(0.5)
                                    else:
                                        consistency_scores.append(0.8)
                                prev_shape = output.shape
                                prev_layer_type = layer_type

                return np.mean(consistency_scores) if consistency_scores else 0.5

        except Exception as e:
            print(f"一致性计算错误: {e}")
            return 0.0

    def generate_architecture_diagram(self,
                                    model: nn.Module,
                                    validation_result: Dict[str, Any],
                                    save_path: Optional[str] = None):
        """生成架构图"""
        G = nx.DiGraph()

        # 添加节点
        for name, layer_type in validation_result['layer_mapping'].items():
            G.add_node(name, layer_type=layer_type)

        # 添加边（基于命名层次）
        for name in validation_result['layer_mapping']:
            parts = name.split('.')
            if len(parts) > 1:
                parent = '.'.join(parts[:-1])
                if parent in G.nodes:
                    G.add_edge(parent, name)

        # 绘制图
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)

        # 根据层类型着色
        colors = {
            LayerType.SIGNAL_PROCESSING: 'lightblue',
            LayerType.FEATURE_EXTRACTION: 'lightgreen',
            LayerType.SYMBOLIC_REASONING: 'lightyellow',
            LayerType.LINGUISTIC_EXPLANATION: 'lightcoral'
        }

        node_colors = []
        for node in G.nodes():
            layer_type = validation_result['layer_mapping'].get(node, 'unknown')
            node_colors.append(colors.get(layer_type, 'gray'))

        nx.draw(G, pos,
                with_labels=True,
                node_color=node_colors,
                node_size=1000,
                font_size=10,
                font_weight='bold',
                arrows=True,
                arrowsize=20)

        # 添加图例
        legend_elements = []
        for layer_type, color in colors.items():
            legend_elements.append(plt.scatter([], [], c=color, label=layer_type))
        plt.legend(handles=legend_elements, loc='upper right')

        plt.title(f"Neural-Symbolic Architecture: {model.__class__.__name__}")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


def validate_multiple_models(models: Dict[str, nn.Module],
                          input_sample: torch.Tensor) -> Dict[str, Dict[str, Any]]:
    """
    批量验证多个模型

    Args:
        models: 模型字典 {name: model}
        input_sample: 输入样本

    Returns:
        results: 所有模型的验证结果
    """
    validator = FrameworkValidator()
    results = {}

    for name, model in models.items():
        print(f"\n验证模型: {name}")
        result = validator.validate_model(model, input_sample)
        results[name] = result

        print(f"  合规性: {'✓' if result['is_compliant'] else '✗'}")
        print(f"  完整性分数: {result['completeness_score']:.2f}")
        print(f"  一致性分数: {result['consistency_score']:.2f}")

        if result['issues']:
            print(f"  问题: {'; '.join(result['issues'][:3])}")

    return results


def generate_comparison_report(results: Dict[str, Dict[str, Any]], save_path: str):
    """生成对比报告"""
    report = {
        'summary': {
            'total_models': len(results),
            'compliant_models': sum(1 for r in results.values() if r['is_compliant']),
            'average_completeness': np.mean([r['completeness_score'] for r in results.values()]),
            'average_consistency': np.mean([r['consistency_score'] for r in results.values()])
        },
        'model_details': results
    }

    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n对比报告已保存到: {save_path}")


# 测试函数
def test_framework_validator():
    """测试框架验证器"""
    # 创建测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.signal_layer = nn.Sequential(
                nn.Conv1d(1, 4, 3),
                nn.ReLU()
            )
            self.feature_extractor = nn.Linear(1000, 50)
            self.reasoning_layer = nn.Sequential(
                nn.Linear(50, 20),
                nn.ReLU(),
                nn.Linear(20, 5)
            )
            self.explanation_generator = nn.ModuleDict({
                'templates': ['template1', 'template2']
            })

        def forward(self, x):
            x = self.signal_layer(x)
            x = x.view(x.size(0), -1)
            features = self.feature_extractor(x)
            logits = self.reasoning_layer(features)
            return logits

    # 创建验证器
    validator = FrameworkValidator()
    model = TestModel()
    input_sample = torch.randn(1, 1, 100)

    # 验证模型
    result = validator.validate_model(model, input_sample)

    print("框架验证结果:")
    print(f"  模型: {result['model_name']}")
    print(f"  合规性: {'✓' if result['is_compliant'] else '✗'}")
    print(f"  完整性分数: {result['completeness_score']:.2f}")
    print(f"  一致性分数: {result['consistency_score']:.2f}")
    print(f"  层映射: {result['layer_mapping']}")

    # 生成架构图
    validator.generate_architecture_diagram(
        model, result,
        save_path='./Paper/Neuralsymbolic_theory/results/test_architecture.png'
    )

    return result


if __name__ == "__main__":
    import numpy as np

    # 运行测试
    test_framework_validator()