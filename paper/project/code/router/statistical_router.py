import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

try:
    from ..utils.statistical_features import StatisticalFeatureExtractor
except ImportError:
    from utils.statistical_features import StatisticalFeatureExtractor


class StatisticalRouter(nn.Module):
    """统计特征驱动的智能路由器

    基于可解释统计特征进行专家分配决策，模拟人类工程师的诊断思维。
    使用RMS、峭度、谱重心等统计量作为路由输入，而非难以解释的隐向量。

    路由逻辑：
    - 低频特征明显 → 激活低通专家
    - 谐波特征明显 → 激活谐波专家
    - 冲击特征明显 → 激活包络专家
    - 复杂特征 → 激活多个专家
    """

    def __init__(self,
                 num_experts: int = 3,
                 feature_dim: int = 64,
                 temperature: float = 1.0,
                 load_balance_loss_weight: float = 0.1,
                 sparsity_loss_weight: float = 0.05,
                 expert_descriptions: Optional[Dict[int, Dict]] = None):
        super().__init__()

        self.num_experts = num_experts
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.load_balance_loss_weight = load_balance_loss_weight
        self.sparsity_loss_weight = sparsity_loss_weight

        # 统计特征提取器
        self.feature_extractor = StatisticalFeatureExtractor()

        # 路由决策网络
        self.routing_net = nn.Sequential(
            nn.Linear(15, 64),  # 15个统计特征
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_experts)
        )

        # 专家特征描述 (用于基于规则的路由)
        self.expert_descriptions = expert_descriptions or self._build_default_descriptions(num_experts)
        self.expert_families = {
            expert_id: str(desc.get('family', self._infer_family(expert_id)))
            for expert_id, desc in self.expert_descriptions.items()
        }

        # 用于记录路由历史的变量
        self.register_buffer('routing_history', torch.zeros(1000, num_experts))  # 最多记录1000个样本
        self.register_buffer('history_pointer', torch.zeros(1, dtype=torch.long))

    def _infer_family(self, expert_id: int) -> str:
        families = ['low_pass', 'harmonic', 'envelope']
        return families[expert_id % len(families)]

    def _build_default_descriptions(self, num_experts: int) -> Dict[int, Dict]:
        defaults = {
            'low_pass': {
                'name': 'LowPassExpert',
                'family': 'low_pass',
                'preferred_features': ['低频能量集中', 'RMS较高', '频谱重心低'],
                'feature_thresholds': {
                    'spectral_centroid_max': 500,
                    'rms_min': 0.1,
                    'kurtosis_max': 3.0
                }
            },
            'harmonic': {
                'name': 'HarmonicExpert',
                'family': 'harmonic',
                'preferred_features': ['谐波明显', '频谱有规律峰', '相位稳定'],
                'feature_thresholds': {
                    'peak_factor_min': 2.0,
                    'peak_factor_max': 4.0,
                    'spectral_centroid_min': 100,
                    'spectral_centroid_max': 1000
                }
            },
            'envelope': {
                'name': 'EnvelopeExpert',
                'family': 'envelope',
                'preferred_features': ['冲击明显', '峭度高', '频谱重心高'],
                'feature_thresholds': {
                    'kurtosis_min': 3.5,
                    'crest_factor_min': 4.0,
                    'spectral_centroid_min': 800
                }
            }
        }
        descriptions: Dict[int, Dict] = {}
        for expert_id in range(num_experts):
            family = self._infer_family(expert_id)
            base = dict(defaults[family])
            base['expert_id'] = f'E{expert_id + 1}'
            base['name'] = f"{base['name']}_{expert_id + 1}"
            descriptions[expert_id] = base
        return descriptions

    def _family_indices(self, family: str) -> List[int]:
        return [
            expert_id
            for expert_id in range(self.num_experts)
            if self.expert_families.get(expert_id, self._infer_family(expert_id)) == family
        ]

    def _apply_family_bonus(
        self,
        logits: torch.Tensor,
        sample_idx: int,
        family: str,
        bonus: float,
    ) -> torch.Tensor:
        indices = self._family_indices(family)
        if not indices:
            return logits
        bonus_logits = torch.zeros_like(logits)
        bonus_logits[sample_idx, indices] = bonus / len(indices)
        return logits + bonus_logits

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            x: 输入信号 [batch_size, signal_length]
        Returns:
            routing_weights: 专家权重 [batch_size, num_experts]
            statistics: 统计特征 [batch_size, 15]
            routing_info: 路由信息字典
        """
        batch_size, signal_len = x.shape

        # 1. 提取统计特征
        statistics = self.feature_extractor(x)  # [batch_size, 15]

        # 2. 基于神经网络的路由决策
        routing_logits = self.routing_net(statistics)

        # 3. 基于规则的路由调整
        rule_adjusted_logits = self._apply_rule_adjustment(routing_logits, statistics, x)

        # 4. 温度缩放和softmax
        routing_weights = F.softmax(rule_adjusted_logits / self.temperature, dim=-1)

        # 5. 计算路由损失
        sparsity_loss = self._compute_sparsity_loss(routing_weights)
        load_balance_loss = self._compute_load_balance_loss(routing_weights)

        # 6. 记录路由历史
        self._update_routing_history(routing_weights)

        # 7. 生成路由解释
        routing_explanations = self._generate_routing_explanations(statistics, routing_weights)

        # 8. 收集路由信息
        routing_info = {
            'statistics': statistics,
            'routing_logits': routing_logits,
            'routing_weights': routing_weights,
            'sparsity_loss': sparsity_loss,
            'load_balance_loss': load_balance_loss,
            'total_routing_loss': (self.sparsity_loss_weight * sparsity_loss +
                                  self.load_balance_loss_weight * load_balance_loss),
            'routing_explanations': routing_explanations,
            'dominant_expert': torch.argmax(routing_weights, dim=-1),
            'expert_confidence': torch.max(routing_weights, dim=-1)[0]
        }

        return routing_weights, statistics, routing_info

    def _apply_rule_adjustment(
        self,
        routing_logits: torch.Tensor,
        statistics: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """基于规则调整路由权重"""
        adjusted_logits = torch.tanh(routing_logits)
        batch_size = statistics.shape[0]
        dominant_freqs, high_freq_ratios = self._spectral_routing_cues(x)

        # 获取特征的物理解释
        feature_interpretations = self.feature_extractor.interpret_features(statistics)

        for i in range(batch_size):
            interp = feature_interpretations[i]
            dominant_freq = dominant_freqs[i]
            high_freq_ratio = high_freq_ratios[i]

            if dominant_freq < 120:
                adjusted_logits = self._apply_family_bonus(adjusted_logits, i, 'low_pass', 5.0)
            elif dominant_freq > 1000 or high_freq_ratio > 0.45:
                adjusted_logits = self._apply_family_bonus(adjusted_logits, i, 'envelope', 5.0)
            else:
                adjusted_logits = self._apply_family_bonus(adjusted_logits, i, 'harmonic', 5.0)

            # 规则1: 低频特征明显 → 增强低通专家权重
            if (interp['frequency_characteristic'] == 'low_freq' and
                interp['energy_level'] == 'high'):
                adjusted_logits = self._apply_family_bonus(adjusted_logits, i, 'low_pass', 1.0)

            # 规则2: 周期性明显 → 增强谐波专家权重
            elif interp['waveform_pattern'] == 'periodic':
                adjusted_logits = self._apply_family_bonus(adjusted_logits, i, 'harmonic', 1.0)

            # 规则3: 冲击性明显 → 增强包络专家权重
            elif interp['waveform_pattern'] == 'impulsive':
                adjusted_logits = self._apply_family_bonus(adjusted_logits, i, 'envelope', 1.0)

            # 规则4: 基于频谱重心的细微调整
            spec_centroid = statistics[i, 14]  # 频谱重心
            if spec_centroid < 300:
                adjusted_logits = self._apply_family_bonus(adjusted_logits, i, 'low_pass', 0.5)
            elif spec_centroid > 1000:
                adjusted_logits = self._apply_family_bonus(adjusted_logits, i, 'envelope', 0.5)
            else:
                adjusted_logits = self._apply_family_bonus(adjusted_logits, i, 'harmonic', 0.5)

        return adjusted_logits

    def _spectral_routing_cues(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spectrum = torch.abs(torch.fft.rfft(x, dim=-1))
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / self.feature_extractor.sample_rate).to(x.device)
        if spectrum.shape[-1] > 1:
            spectrum = spectrum.clone()
            spectrum[:, 0] = 0.0
        peak_indices = torch.argmax(spectrum, dim=-1)
        dominant_freqs = freqs[peak_indices]
        total_energy = torch.sum(spectrum, dim=-1) + 1e-8
        high_freq_energy = torch.sum(spectrum[:, freqs > 1000], dim=-1)
        high_freq_ratios = high_freq_energy / total_energy
        return dominant_freqs, high_freq_ratios

    def _compute_sparsity_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """计算稀疏性损失，鼓励每个样本只激活少量专家"""
        # L1正则化促进稀疏性
        l1_loss = torch.mean(torch.sum(torch.abs(routing_weights), dim=-1))
        return l1_loss

    def _compute_load_balance_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """计算负载均衡损失，防止某些专家长期不被使用"""
        # 计算每个专家的平均激活度
        mean_expert_usage = torch.mean(routing_weights, dim=0)

        # 期望每个专家的平均激活度为1/num_experts
        target_usage = 1.0 / self.num_experts
        load_balance_loss = torch.sum((mean_expert_usage - target_usage) ** 2)

        return load_balance_loss

    def _update_routing_history(self, routing_weights: torch.Tensor):
        """更新路由历史记录"""
        batch_size = routing_weights.shape[0]
        pointer = self.history_pointer.item()

        for i in range(batch_size):
            if pointer < self.routing_history.shape[0]:
                self.routing_history[pointer] = routing_weights[i]
                pointer = (pointer + 1) % self.routing_history.shape[0]

        self.history_pointer[0] = pointer

    def _generate_routing_explanations(self, statistics: torch.Tensor,
                                      routing_weights: torch.Tensor) -> list:
        """生成路由决策的解释"""
        batch_size = statistics.shape[0]
        feature_names = self.feature_extractor.get_feature_names()
        explanations = []

        for i in range(batch_size):
            # 主要激活专家
            dominant_expert = torch.argmax(routing_weights[i]).item()
            confidence = routing_weights[i, dominant_expert].item()

            # 关键特征
            stats_np = statistics[i].detach().cpu().numpy()
            key_features = {
                'RMS': stats_np[6],
                '峭度': stats_np[11],
                '峰值因子': stats_np[7],
                '频谱重心': stats_np[14]
            }

            # 生成解释
            explanation = {
                'sample_id': i,
                'dominant_expert': dominant_expert,
                'expert_name': self.expert_descriptions[dominant_expert].get(
                    'name',
                    self.expert_descriptions[dominant_expert].get('expert_name', f'Expert_{dominant_expert}')
                ),
                'confidence': confidence,
                'expert_weights': routing_weights[i].detach().cpu().numpy().tolist(),
                'key_features': key_features,
                'routing_reason': self._get_routing_reason(dominant_expert, key_features),
                'experts_considered': torch.where(routing_weights[i] > 0.1)[0].tolist()
            }

            explanations.append(explanation)

        return explanations

    def _get_routing_reason(self, expert_id: int, key_features: dict) -> str:
        """获取选择特定专家的原因"""
        family = self.expert_families.get(expert_id, self._infer_family(expert_id))
        expert_name = self.expert_descriptions.get(expert_id, {}).get('name', f'Expert_{expert_id}')
        if family == 'low_pass':
            return f"{expert_name}: 低频特征明显(RMS={key_features['RMS']:.3f}, 频谱重心={key_features['频谱重心']:.1f}Hz)"
        if family == 'harmonic':
            return f"{expert_name}: 谐波特征明显(峰值因子={key_features['峰值因子']:.2f}, 频谱重心={key_features['频谱重心']:.1f}Hz)"
        if family == 'envelope':
            return f"{expert_name}: 冲击特征明显(峭度={key_features['峭度']:.2f}, 峰值因子={key_features['峰值因子']:.2f})"
        return f"{expert_name}: 综合特征分析结果"

    def get_routing_statistics(self) -> Dict:
        """获取路由统计信息"""
        # 从历史记录计算统计
        history_pointer = self.history_pointer.item()
        if history_pointer > 0:
            history_data = self.routing_history[:history_pointer]
        else:
            history_data = self.routing_history

        if len(history_data) == 0:
            return {'message': 'No routing history available'}

        expert_usage = torch.mean(history_data, dim=0)
        sparsity = torch.mean(torch.sum(history_data > 0.1, dim=-1).float())

        return {
            'total_samples': len(history_data),
            'expert_usage_rates': expert_usage.tolist(),
            'average_active_experts': sparsity.item(),
            'most_used_expert': torch.argmax(expert_usage).item(),
            'least_used_expert': torch.argmin(expert_usage).item(),
            'usage_balance': 1.0 - torch.std(expert_usage).item()
        }

    def visualize_routing_decision(self, x: torch.Tensor) -> Dict:
        """可视化路由决策过程"""
        with torch.no_grad():
            routing_weights, statistics, routing_info = self.forward(x)

            # 特征重要性分析
            feature_importance = self._analyze_feature_importance(statistics, routing_weights)

            return {
                'routing_weights': routing_weights,
                'statistics': statistics,
                'feature_importance': feature_importance,
                'routing_explanations': routing_info['routing_explanations'],
                'dominant_expert': routing_info['dominant_expert']
            }

    def _analyze_feature_importance(self, statistics: torch.Tensor,
                                   routing_weights: torch.Tensor) -> torch.Tensor:
        """分析各统计特征对路由决策的重要性"""
        # 计算每个特征与路由权重的相关性
        feature_names = self.feature_extractor.get_feature_names()
        feature_importance = []

        for i in range(statistics.shape[-1]):
            feature_values = statistics[:, i]
            # 计算该特征与最大专家权重的相关性
            max_weights, _ = torch.max(routing_weights, dim=-1)
            correlation = torch.corrcoef(torch.stack([feature_values, max_weights]))[0, 1]
            feature_importance.append(correlation if not torch.isnan(correlation) else 0.0)

        return torch.tensor(feature_importance, device=statistics.device)
