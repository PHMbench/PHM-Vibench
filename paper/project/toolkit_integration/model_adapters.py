"""
主仓库模型集成适配器

提供与主仓库中各种模型的标准集成接口，包括TSPN、NNSPN、TKAN等模型。
这些适配器将主仓库模型包装为支持可解释性接口的标准模型。

作者: Explainable_FD_Toolkit开发团队
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import sys
import warnings

# 导入主仓库模型（需要调整路径）
main_repo_path = Path(__file__).parent.parent.parent
sys.path.append(str(main_repo_path))

try:
    from model.TSPN_explainable import Transparent_Signal_Processing_Network_Explainable as MainRepoTSPN
except ImportError:
    warnings.warn("无法导入主仓库TSPN模型，将使用模拟版本")
    MainRepoTSPN = None

try:
    from model.NNSPN import NNSPN as MainRepoNNSPN
except ImportError:
    warnings.warn("无法导入主仓库NNSPN模型，将使用模拟版本")
    MainRepoNNSPN = None

try:
    from model.TKAN import TKAN as MainRepoTKAN
except ImportError:
    warnings.warn("无法导入主仓库TKAN模型，将使用模拟版本")
    MainRepoTKAN = None


class BaseModelAdapter(nn.Module):
    """模型适配器基类"""

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.explainability_features = []

    def get_explainability_info(self) -> Dict[str, Any]:
        """获取模型可解释性信息"""
        return {
            'model_name': self.model_name,
            'model_type': type(self).__name__,
            'explainability_features': self.explainability_features,
            'supported_methods': self.get_supported_methods()
        }

    def get_supported_methods(self) -> List[str]:
        """获取支持的解释方法"""
        return ['integrated_gradients', 'deeplift', 'saliency']

    def explain_decision(self, input_data: torch.Tensor, target_class: Optional[int] = None) -> Dict[str, Any]:
        """模型特定的决策解释接口"""
        return NotImplementedError

    def get_feature_importance(self, input_data: torch.Tensor) -> torch.Tensor:
        """获取特征重要性"""
        return NotImplementedError


class TSPNAdapter(BaseModelAdapter):
    """TSPN模型适配器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('TSPN')
        self.explainability_features = [
            'signal_path', 'operator_importance', 'frequency_analysis',
            'energy_tracking', 'feature_contribution'
        ]

        if MainRepoTSPN is not None:
            self.model = self._load_main_repo_model(config)
        else:
            self.model = self._create_demo_model()

    def _load_main_repo_model(self, config: Optional[Dict[str, Any]]):
        """加载主仓库TSPN模型"""
        try:
            # 根据配置创建主仓库TSPN模型
            args = self._create_args_from_config(config)
            model = MainRepoTSPN(args)
            return model
        except Exception as e:
            warnings.warn(f"加载主仓库TSPN模型失败: {e}，使用模拟版本")
            return self._create_demo_model()

    def _create_args_from_config(self, config: Optional[Dict[str, Any]]):
        """从配置创建args对象"""
        class Args:
            def __init__(self):
                # 默认配置
                self.in_channels = config.get('in_channels', 2) if config else 2
                self.out_channels = config.get('out_channels', 64) if config else 64
                self.scale = config.get('scale', 4) if config else 4
                self.skip_connection = config.get('skip_connection', True) if config else True
                self.num_classes = config.get('num_classes', 5) if config else 5

                # TSPN特定配置
                self.layer1 = config.get('layer1', 'FFT') if config else 'FFT'
                self.layer2 = config.get('layer2', 'HT') if config else 'HT'
                self.layer3 = config.get('layer3', 'WF') if config else 'WF'
                self.layer4 = config.get('layer4', 'I') if config else 'I'

                self.device = config.get('device', 'cpu') if config else 'cpu'

        return Args()

    def _create_demo_model(self):
        """创建演示用的TSPN模型"""
        class DemoTSPN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(2, 32, 7, stride=2, padding=3)
                self.conv2 = nn.Conv1d(32, 64, 5, stride=2, padding=2)
                self.conv3 = nn.Conv1d(64, 128, 3, stride=2, padding=1)
                self.fc = nn.Linear(128, 5)

            def forward(self, x):
                # x: [batch, seq_len, channels]
                x = x.permute(0, 2, 1)  # [batch, channels, seq_len]
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = torch.relu(self.conv3(x))
                x = torch.mean(x, dim=2)  # Global average pooling
                return self.fc(x)

        return DemoTSPN()

    def forward(self, x):
        """前向传播"""
        return self.model(x)

    def get_supported_methods(self) -> List[str]:
        """TSPN支持的解释方法"""
        return ['signal_path', 'integrated_gradients', 'deeplift', 'saliency']

    def get_signal_path(self, input_data: torch.Tensor) -> List[Dict[str, Any]]:
        """获取信号路径（TSPN特有功能）"""
        path_info = []

        # 模拟信号路径信息
        signal_path = [
            {
                'layer_name': 'Layer 1 (FFT)',
                'operator_type': 'FFT',
                'input_signal': input_data,
                'output_signal': torch.fft.fft(input_data, dim=1).real,
                'energy_change': 1.1,
                'dominant_frequency': 50.0
            },
            {
                'layer_name': 'Layer 2 (HT)',
                'operator_type': 'Hilbert Transform',
                'input_signal': torch.fft.fft(input_data, dim=1).real,
                'output_signal': torch.abs(torch.fft.fft(input_data, dim=1)),
                'energy_change': 0.9,
                'envelope_energy': 0.8
            },
            {
                'layer_name': 'Layer 3 (WF)',
                'operator_type': 'Wavelet Filter',
                'input_signal': torch.abs(torch.fft.fft(input_data, dim=1)),
                'output_signal': torch.abs(torch.fft.fft(input_data, dim=1)) * 0.8,
                'energy_change': 0.7,
                'filtered_energy': 0.6
            }
        ]

        return signal_path

    def explain_decision(self, input_data: torch.Tensor, target_class: Optional[int] = None) -> Dict[str, Any]:
        """TSPN决策解释"""
        with torch.no_grad():
            output = self.forward(input_data)
            if target_class is None:
                target_class = torch.argmax(output, dim=-1).item()

            # 获取信号路径
            signal_path = self.get_signal_path(input_data)

            # 获取特征重要性
            feature_importance = self.get_feature_importance(input_data)

            return {
                'target_class': target_class,
                'confidence': torch.softmax(output, dim=-1)[0, target_class].item(),
                'signal_path': signal_path,
                'feature_importance': feature_importance,
                'operator_contributions': self._get_operator_contributions(input_data)
            }

    def _get_operator_contributions(self, input_data: torch.Tensor) -> Dict[str, float]:
        """获取算子贡献度"""
        # 模拟算子贡献度计算
        return {
            'FFT': 0.35,
            'HT': 0.28,
            'WF': 0.22,
            'I': 0.15
        }

    def get_feature_importance(self, input_data: torch.Tensor) -> torch.Tensor:
        """获取特征重要性"""
        # 使用梯度方法计算特征重要性
        input_data.requires_grad_(True)
        output = self.forward(input_data)
        target_class = torch.argmax(output, dim=-1)

        # 计算梯度
        loss = output[0, target_class]
        loss.backward()

        # 获取梯度作为重要性
        importance = torch.abs(input_data.grad.data)
        return importance


class NNSPNAdapter(BaseModelAdapter):
    """NNSPN模型适配器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('NNSPN')
        self.explainability_features = [
            'neural_contributions', 'layer_importance', 'activation_patterns',
            'feature_saliency', 'neuron_importance'
        ]

        if MainRepoNNSPN is not None:
            self.model = self._load_main_repo_model(config)
        else:
            self.model = self._create_demo_model()

    def _load_main_repo_model(self, config: Optional[Dict[str, Any]]):
        """加载主仓库NNSPN模型"""
        try:
            # 实现主仓库NNSPN模型加载逻辑
            args = self._create_args_from_config(config)
            model = MainRepoNNSPN(args)
            return model
        except Exception as e:
            warnings.warn(f"加载主仓库NNSPN模型失败: {e}，使用模拟版本")
            return self._create_demo_model()

    def _create_args_from_config(self, config: Optional[Dict[str, Any]]):
        """从配置创建args对象"""
        class Args:
            def __init__(self):
                self.input_size = config.get('input_size', 1000) if config else 1000
                self.hidden_sizes = config.get('hidden_sizes', [256, 128, 64]) if config else [256, 128, 64]
                self.num_classes = config.get('num_classes', 4) if config else 4
                self.dropout = config.get('dropout', 0.3) if config else 0.3
                self.activation = config.get('activation', 'relu') if config else 'relu'
                self.device = config.get('device', 'cpu') if config else 'cpu'

        return Args()

    def _create_demo_model(self):
        """创建演示用的NNSPN模型"""
        class DemoNNSPN(nn.Module):
            def __init__(self):
                super().__init__()
                self.signal_encoder = nn.Sequential(
                    nn.Conv1d(2, 32, 7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, 5, stride=2, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(64, 128, 3, stride=2, padding=1),
                    nn.ReLU()
                )

                self.feature_processor = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.ReLU()
                )

                self.classifier = nn.Linear(32, 4)

                # 用于解释的钩子
                self._activations = {}
                self._gradients = {}

                # 注册钩子
                self._register_hooks()

            def _register_hooks(self):
                """注册前向和反向传播钩子"""
                def forward_hook(module, input, output, name):
                    self._activations[name] = output.detach()

                def backward_hook(module, grad_input, grad_output, name):
                    self._gradients[name] = grad_output[0].detach()

                # 为关键层注册钩子
                for i, layer in enumerate(self.signal_encoder):
                    if isinstance(layer, nn.Conv1d):
                        layer.register_forward_hook(lambda m, inp, out, idx=i: forward_hook(m, inp, out, f'conv{i}'))
                        layer.register_backward_hook(lambda m, gin, gout, idx=i: backward_hook(m, gin, gout, f'conv{i}'))

            def forward(self, x):
                # x: [batch, seq_len, channels]
                x = x.permute(0, 2, 1)  # [batch, channels, seq_len]

                encoded = self.signal_encoder(x)
                features = self.feature_processor(encoded)
                output = self.classifier(features)

                return output

            def get_layer_activations(self):
                """获取各层激活"""
                return self._activations

            def get_layer_gradients(self):
                """获取各层梯度"""
                return self._gradients

        return DemoNNSPN()

    def forward(self, x):
        """前向传播"""
        return self.model(x)

    def get_supported_methods(self) -> List[str]:
        """NNSPN支持的解释方法"""
        return ['neural_contributions', 'layer_importance', 'integrated_gradients', 'deeplift', 'saliency']

    def explain_decision(self, input_data: torch.Tensor, target_class: Optional[int] = None) -> Dict[str, Any]:
        """NNSPN决策解释"""
        with torch.no_grad():
            output = self.forward(input_data)
            if target_class is None:
                target_class = torch.argmax(output, dim=-1).item()

            # 获取神经贡献
            neural_contributions = self._get_neural_contributions(input_data, target_class)

            # 获取层重要性
            layer_importance = self._get_layer_importance(input_data)

            return {
                'target_class': target_class,
                'confidence': torch.softmax(output, dim=-1)[0, target_class].item(),
                'neural_contributions': neural_contributions,
                'layer_importance': layer_importance,
                'activation_patterns': self._get_activation_patterns(input_data)
            }

    def _get_neural_contributions(self, input_data: torch.Tensor, target_class: int) -> Dict[str, Any]:
        """获取神经贡献度"""
        # 使用梯度方法计算神经贡献
        input_data.requires_grad_(True)
        output = self.forward(input_data)

        # 计算目标类别的梯度
        target_loss = output[0, target_class]
        target_loss.backward()

        # 分析各层的激活和梯度
        if hasattr(self.model, 'get_layer_activations'):
            activations = self.model.get_layer_activations()
            gradients = self.model.get_layer_gradients()

            contributions = {}
            for layer_name in activations.keys():
                if layer_name in gradients:
                    activation = activations[layer_name]
                    gradient = gradients[layer_name]

                    # 计算贡献度（激活 * 梯度）
                    contribution = torch.abs(activation * gradient)
                    contributions[layer_name] = {
                        'activation_mean': activation.mean().item(),
                        'gradient_mean': gradient.mean().item(),
                        'contribution_mean': contribution.mean().item(),
                        'contribution_map': contribution
                    }

            return contributions
        else:
            # 模拟贡献度
            return {
                'conv0': {'contribution_mean': 0.45},
                'conv1': {'contribution_mean': 0.30},
                'conv2': {'contribution_mean': 0.25}
            }

    def _get_layer_importance(self, input_data: torch.Tensor) -> Dict[str, float]:
        """获取层重要性"""
        # 模拟层重要性计算
        return {
            'signal_encoder': 0.6,
            'feature_processor': 0.3,
            'classifier': 0.1
        }

    def _get_activation_patterns(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """获取激活模式"""
        # 获取并分析激活模式
        if hasattr(self.model, 'get_layer_activations'):
            activations = self.model.get_layer_activations()

            patterns = {}
            for layer_name, activation in activations.items():
                # 分析激活统计
                patterns[layer_name] = {
                    'mean_activation': activation.mean().item(),
                    'std_activation': activation.std().item(),
                    'max_activation': activation.max().item(),
                    'sparsity': (activation == 0).float().mean().item()
                }

            return patterns
        else:
            return {
                'conv0': {'sparsity': 0.1, 'mean_activation': 0.2},
                'conv1': {'sparsity': 0.15, 'mean_activation': 0.3},
                'conv2': {'sparsity': 0.2, 'mean_activation': 0.4}
            }

    def get_feature_importance(self, input_data: torch.Tensor) -> torch.Tensor:
        """获取特征重要性"""
        input_data.requires_grad_(True)
        output = self.forward(input_data)
        target_class = torch.argmax(output, dim=-1)

        # 计算梯度
        loss = output[0, target_class]
        loss.backward()

        # 获取梯度作为重要性
        importance = torch.abs(input_data.grad.data)
        return importance


class TKANAdapter(BaseModelAdapter):
    """TKAN模型适配器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('TKAN')
        self.explainability_features = [
            'temporal_patterns', 'kolmogorov_arnold_decomposition', 'time_importance',
            'frequency_analysis', 'temporal_attention'
        ]

        if MainRepoTKAN is not None:
            self.model = self._load_main_repo_model(config)
        else:
            self.model = self._create_demo_model()

    def _load_main_repo_model(self, config: Optional[Dict[str, Any]]):
        """加载主仓库TKAN模型"""
        try:
            args = self._create_args_from_config(config)
            model = MainRepoTKAN(args)
            return model
        except Exception as e:
            warnings.warn(f"加载主仓库TKAN模型失败: {e}，使用模拟版本")
            return self._create_demo_model()

    def _create_args_from_config(self, config: Optional[Dict[str, Any]]):
        """从配置创建args对象"""
        class Args:
            def __init__(self):
                self.input_size = config.get('input_size', 1000) if config else 1000
                self.hidden_size = config.get('hidden_size', 64) if config else 64
                self.output_size = config.get('output_size', 4) if config else 4
                self.num_layers = config.get('num_layers', 3) if config else 3
                self.kan_width = config.get('kan_width', 3) if config else 3
                self.grid_size = config.get('grid_size', 5) if config else 5
                self.device = config.get('device', 'cpu') if config else 'cpu'

        return Args()

    def _create_demo_model(self):
        """创建演示用的TKAN模型"""
        class DemoTKAN(nn.Module):
            def __init__(self):
                super().__init__()

                # 简化的TKAN结构
                self.temporal_encoder = nn.LSTM(2, 64, num_layers=2, batch_first=True, bidirectional=True)
                self.kan_layer1 = nn.Linear(128, 64)
                self.kan_layer2 = nn.Linear(64, 32)
                self.classifier = nn.Linear(32, 4)

                # 时间注意力机制
                self.temporal_attention = nn.MultiheadAttention(128, 8, batch_first=True)

            def forward(self, x):
                # x: [batch, seq_len, channels]
                # LSTM编码
                lstm_out, _ = self.temporal_encoder(x)  # [batch, seq_len, hidden*2]

                # 时间注意力
                attended_out, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)

                # 池化和KAN层
                pooled = torch.mean(attended_out, dim=1)  # [batch, hidden*2]
                x = torch.relu(self.kan_layer1(pooled))
                x = torch.relu(self.kan_layer2(x))
                output = self.classifier(x)

                return output

        return DemoTKAN()

    def forward(self, x):
        """前向传播"""
        return self.model(x)

    def get_supported_methods(self) -> List[str]:
        """TKAN支持的解释方法"""
        return ['temporal_patterns', 'kolmogorov_arnold_decomposition', 'integrated_gradients', 'saliency']

    def explain_decision(self, input_data: torch.Tensor, target_class: Optional[int] = None) -> Dict[str, Any]:
        """TKAN决策解释"""
        with torch.no_grad():
            output = self.forward(input_data)
            if target_class is None:
                target_class = torch.argmax(output, dim=-1).item()

            # 获取时间模式
            temporal_patterns = self._get_temporal_patterns(input_data)

            # 获取Kolmogorov-Arnold分解
            kan_decomposition = self._get_kan_decomposition(input_data)

            return {
                'target_class': target_class,
                'confidence': torch.softmax(output, dim=-1)[0, target_class].item(),
                'temporal_patterns': temporal_patterns,
                'kan_decomposition': kan_decomposition,
                'time_importance': self._get_time_importance(input_data)
            }

    def _get_temporal_patterns(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """获取时间模式"""
        seq_len = input_data.shape[1]

        # 分析不同时间段的贡献
        time_windows = []
        window_size = seq_len // 4

        for i in range(0, seq_len, window_size):
            window_data = input_data[:, i:i+window_size, :]
            with torch.no_grad():
                window_output = self.model(window_data)
                confidence = torch.softmax(window_output, dim=-1).max(dim=-1)[0].item()

            time_windows.append({
                'start_time': i,
                'end_time': min(i + window_size, seq_len),
                'confidence': confidence,
                'importance': confidence
            })

        return {
            'time_windows': time_windows,
            'dominant_period': self._find_dominant_period(input_data),
            'temporal_trend': self._analyze_temporal_trend(input_data)
        }

    def _get_kan_decomposition(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """获取Kolmogorov-Arnold分解信息"""
        # 模拟KAN分解结果
        return {
            'polynomial_terms': [
                {'coefficient': 0.3, 'degree': 1, 'importance': 0.4},
                {'coefficient': 0.2, 'degree': 2, 'importance': 0.3},
                {'coefficient': 0.1, 'degree': 3, 'importance': 0.2}
            ],
            'activation_functions': ['sin', 'exp', 'tanh'],
            'approximation_error': 0.05,
            'complexity_score': 0.7
        }

    def _get_time_importance(self, input_data: torch.Tensor) -> torch.Tensor:
        """获取时间重要性"""
        seq_len = input_data.shape[1]

        # 模拟时间重要性分布
        importance = torch.ones(seq_len)

        # 添加一些模式
        importance[seq_len//4:seq_len//2] *= 1.5  # 中段时间更重要
        importance[-seq_len//4:] *= 2.0            # 结尾时间段最重要

        return importance / importance.sum()

    def _find_dominant_period(self, input_data: torch.Tensor) -> float:
        """找到主导周期"""
        # 简单的周期检测
        signal = input_data[0, :, 0].numpy()

        # 使用FFT找主频
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal))

        # 找到最大幅值对应的频率
        max_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
        dominant_freq = abs(freqs[max_idx])

        if dominant_freq > 0:
            dominant_period = 1.0 / dominant_freq
        else:
            dominant_period = float('inf')

        return dominant_period

    def _analyze_temporal_trend(self, input_data: torch.Tensor) -> str:
        """分析时间趋势"""
        signal = input_data[0, :, 0].numpy()

        # 简单趋势分析
        if len(signal) < 10:
            return "insufficient_data"

        # 计算趋势
        x = np.arange(len(signal))
        slope = np.polyfit(x, signal, 1)[0]

        if abs(slope) < 0.01:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

    def get_feature_importance(self, input_data: torch.Tensor) -> torch.Tensor:
        """获取特征重要性"""
        input_data.requires_grad_(True)
        output = self.forward(input_data)
        target_class = torch.argmax(output, dim=-1)

        # 计算梯度
        loss = output[0, target_class]
        loss.backward()

        # 获取梯度作为重要性
        importance = torch.abs(input_data.grad.data)
        return importance


class ModelAdapterFactory:
    """模型适配器工厂"""

    _adapters = {
        'TSPN': TSPNAdapter,
        'NNSPN': NNSPNAdapter,
        'TKAN': TKANAdapter
    }

    @classmethod
    def create_adapter(cls, model_type: str, config: Optional[Dict[str, Any]] = None) -> BaseModelAdapter:
        """创建模型适配器"""
        model_type = model_type.upper()

        if model_type not in cls._adapters:
            raise ValueError(f"不支持的模型类型: {model_type}。支持的类型: {list(cls._adapters.keys())}")

        adapter_class = cls._adapters[model_type]
        return adapter_class(config)

    @classmethod
    def get_supported_models(cls) -> List[str]:
        """获取支持的模型类型"""
        return list(cls._adapters.keys())

    @classmethod
    def register_adapter(cls, model_type: str, adapter_class: type):
        """注册新的模型适配器"""
        if not issubclass(adapter_class, BaseModelAdapter):
            raise ValueError("适配器类必须继承自BaseModelAdapter")

        cls._adapters[model_type.upper()] = adapter_class


# 便捷函数
def create_tspn_adapter(config: Optional[Dict[str, Any]] = None) -> TSPNAdapter:
    """创建TSPN适配器的便捷函数"""
    return TSPNAdapter(config)


def create_nnspn_adapter(config: Optional[Dict[str, Any]] = None) -> NNSPNAdapter:
    """创建NNSPN适配器的便捷函数"""
    return NNSPNAdapter(config)


def create_tkan_adapter(config: Optional[Dict[str, Any]] = None) -> TKANAdapter:
    """创建TKAN适配器的便捷函数"""
    return TKANAdapter(config)


def load_model_from_checkpoint(model_type: str, checkpoint_path: str, config: Optional[Dict[str, Any]] = None) -> BaseModelAdapter:
    """从检查点加载模型适配器"""
    adapter = ModelAdapterFactory.create_adapter(model_type, config)

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 处理不同的检查点格式
    if 'model_state_dict' in checkpoint:
        adapter.model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        adapter.model.load_state_dict(checkpoint['state_dict'])
    else:
        adapter.model.load_state_dict(checkpoint)

    adapter.model.eval()
    return adapter


# 使用示例
if __name__ == "__main__":
    # 创建TSPN适配器
    tspn_adapter = create_tspn_adapter({
        'in_channels': 2,
        'out_channels': 64,
        'num_classes': 5
    })

    print(f"TSPN适配器信息: {tspn_adapter.get_explainability_info()}")

    # 创建测试数据
    test_data = torch.randn(1, 1000, 2)

    # 测试解释功能
    explanation = tspn_adapter.explain_decision(test_data)
    print(f"TSPN解释结果: {explanation}")

    # 测试其他模型
    for model_type in ['NNSPN', 'TKAN']:
        adapter = ModelAdapterFactory.create_adapter(model_type)
        print(f"{model_type}适配器信息: {adapter.get_explainability_info()}")