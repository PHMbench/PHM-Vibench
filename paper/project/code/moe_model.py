import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

try:
    from .experts import LowPassExpert, HarmonicExpert, EnvelopeExpert
    from .router import StatisticalRouter
except ImportError:
    from experts.low_pass_expert import LowPassExpert
    from experts.harmonic_expert import HarmonicExpert
    from experts.envelope_expert import EnvelopeExpert
    from router.statistical_router import StatisticalRouter


class NNSPNMoE(nn.Module):
    """NNSPN-MoE: 基于物理机理约束的内在可解释故障诊断模型

    整合多个物理专家和智能路由器的完整MoE模型，实现从"黑盒决策"到"物理解释"的转变。

    架构：
    - 路由器：基于统计特征进行专家分配
    - 专家库：低通专家、谐波专家、包络专家
    - 分类器：最终故障诊断决策
    - 解释模块：生成多层次可解释性分析
    """

    def __init__(self,
                 num_classes: int = 10,
                 feature_dim: int = 64,
                 num_experts: Optional[int] = None,
                 expert_pool: Optional[List[str]] = None,
                 use_load_balance: bool = True,
                 use_sparsity: bool = True,
                 routing_temperature: float = 1.0,
                 dropout_rate: float = 0.1):
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.use_load_balance = use_load_balance
        self.use_sparsity = use_sparsity

        # 初始化专家
        spec_ids = self._resolve_expert_pool(num_experts=num_experts, expert_pool=expert_pool)
        self.expert_spec_ids = list(spec_ids)
        self.experts = nn.ModuleList([
            self._build_configured_expert(spec_id, expert_idx)
            for expert_idx, spec_id in enumerate(self.expert_spec_ids)
        ])

        self.num_experts = len(self.experts)
        self.expert_descriptions = [expert.get_expert_description() for expert in self.experts]

        # 路由器
        self.router = StatisticalRouter(
            num_experts=self.num_experts,
            feature_dim=feature_dim,
            temperature=routing_temperature,
            expert_descriptions={idx: desc for idx, desc in enumerate(self.expert_descriptions)}
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

        # 专家正则化参数
        self.expert_orthogonal_weight = 0.01
        self.expert_diversity_weight = 0.01

        # 物理约束参数
        self.frequency_constraint_weight = 0.1    # 频域约束权重
        self.orthogonal_constraint_weight = 0.05  # 正交约束权重
        self.physics_constraint_weight = 0.1      # 总物理约束权重

        # 记录训练信息
        self.training_stats = {
            'expert_activations': torch.zeros(self.num_experts),
            'routing_entropy': 0.0,
            'total_samples': 0
        }

    @staticmethod
    def _available_expert_specs() -> List[str]:
        return [
            'low_pass_500',
            'harmonic_50',
            'envelope_2000_5000',
            'envelope_1500_4500',
            'harmonic_120',
            'low_pass_800',
            'envelope_2500_5500',
            'harmonic_75',
        ]

    def _resolve_expert_pool(
        self,
        num_experts: Optional[int] = None,
        expert_pool: Optional[List[str]] = None,
    ) -> List[str]:
        if expert_pool:
            return list(expert_pool)

        if num_experts is None:
            num_experts = 3

        available = self._available_expert_specs()
        if num_experts < 1 or num_experts > len(available):
            raise ValueError(f"num_experts must be between 1 and {len(available)}, got {num_experts}")
        return available[:num_experts]

    def _build_configured_expert(self, spec_id: str, expert_idx: int) -> nn.Module:
        base_specs: Dict[str, Dict[str, Any]] = {
            'low_pass_500': {
                'cls': LowPassExpert,
                'kwargs': {'cutoff_freq': 500.0},
                'expert_name': 'LowPassRotorExpert',
                'family': 'low_pass',
                'target_faults': ['转子不平衡', '基础振动', '低频机械松动'],
                'physical_mechanism': '500Hz 低通强调低频转频倍频与基础振动',
            },
            'harmonic_50': {
                'cls': HarmonicExpert,
                'kwargs': {'fundamental_freq': 50.0, 'num_harmonics': 5},
                'expert_name': 'HarmonicMisalignmentExpert',
                'family': 'harmonic',
                'target_faults': ['转子不对中', '谐波类故障', '周期性结构偏差'],
                'physical_mechanism': '50Hz 基频及其倍频谐波强调周期性失配',
            },
            'envelope_2000_5000': {
                'cls': EnvelopeExpert,
                'kwargs': {'band_freq': (2000.0, 5000.0)},
                'expert_name': 'EnvelopeOuterRaceExpert',
                'family': 'envelope',
                'target_faults': ['轴承外圈故障', '宽带冲击', '高频共振'],
                'physical_mechanism': '2000-5000Hz 高频包络用于提取宽带冲击共振',
            },
            'envelope_1500_4500': {
                'cls': EnvelopeExpert,
                'kwargs': {'band_freq': (1500.0, 4500.0)},
                'expert_name': 'EnvelopeInnerRaceExpert',
                'family': 'envelope',
                'target_faults': ['轴承内圈故障', '转频调制冲击', '中高频包络'],
                'physical_mechanism': '1500-4500Hz 包络更关注受转频调制的冲击成分',
            },
            'harmonic_120': {
                'cls': HarmonicExpert,
                'kwargs': {'fundamental_freq': 120.0, 'num_harmonics': 6},
                'expert_name': 'HarmonicGearMeshExpert',
                'family': 'harmonic',
                'target_faults': ['齿轮故障', '边带调制', '啮合频率异常'],
                'physical_mechanism': '120Hz 啮合相关谐波用于捕获中频边带模式',
            },
            'low_pass_800': {
                'cls': LowPassExpert,
                'kwargs': {'cutoff_freq': 800.0},
                'expert_name': 'LowPassLoosenessExpert',
                'family': 'low_pass',
                'target_faults': ['机械松动', '低中频结构振动', '复合低频漂移'],
                'physical_mechanism': '800Hz 低通覆盖更宽的低中频结构振动区间',
            },
            'envelope_2500_5500': {
                'cls': EnvelopeExpert,
                'kwargs': {'band_freq': (2500.0, 5500.0)},
                'expert_name': 'EnvelopeHighResonanceExpert',
                'family': 'envelope',
                'target_faults': ['高频冲击故障', '局部剥落', '高频共振增强'],
                'physical_mechanism': '2500-5500Hz 包络强调更强的高频冲击共振',
            },
            'harmonic_75': {
                'cls': HarmonicExpert,
                'kwargs': {'fundamental_freq': 75.0, 'num_harmonics': 8},
                'expert_name': 'HarmonicDenseSidebandExpert',
                'family': 'harmonic',
                'target_faults': ['复杂谐波故障', '密集边带', '多倍频调制'],
                'physical_mechanism': '75Hz 稠密谐波用于补足复杂边带与多倍频调制模式',
            },
        }
        if spec_id not in base_specs:
            raise KeyError(f"Unknown expert spec: {spec_id}")

        spec = base_specs[spec_id]
        expert = spec['cls'](feature_dim=self.feature_dim, **spec['kwargs'])
        description = expert.get_expert_description()
        description.update(
            {
                'expert_id': f'E{expert_idx + 1}',
                'expert_name': spec['expert_name'],
                'name': spec['expert_name'],
                'family': spec['family'],
                'spec_id': spec_id,
                'target_faults': spec['target_faults'],
                'physical_mechanism': spec['physical_mechanism'],
            }
        )

        class ConfiguredExpert(nn.Module):
            def __init__(self, base_expert: nn.Module, base_description: Dict[str, Any]):
                super().__init__()
                self.base_expert = base_expert
                self.base_description = base_description

            def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
                return self.base_expert(x)

            def get_expert_description(self) -> Dict[str, Any]:
                return dict(self.base_description)

        return ConfiguredExpert(expert, description)

    def forward(self, x: torch.Tensor, return_explanations: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播
        Args:
            x: 输入信号 [batch_size, signal_length]
            return_explanations: 是否返回解释信息
        Returns:
            logits: 分类logits [batch_size, num_classes]
            metadata: 包含专家激活、路由信息等的元数据
        """
        batch_size = x.shape[0]

        # 1. 路由决策
        routing_weights, statistics, routing_info = self.router(x)

        # 2. 专家并行处理
        expert_outputs = []
        expert_metadata = []

        for i, expert in enumerate(self.experts):
            expert_output, expert_meta = expert(x)
            expert_outputs.append(expert_output)
            expert_metadata.append(expert_meta)

        # 3. 专家输出加权和
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, feature_dim]
        routing_weights = routing_weights.unsqueeze(-1)      # [batch_size, num_experts, 1]

        # 加权融合
        fused_features = torch.sum(expert_outputs * routing_weights, dim=1)  # [batch_size, feature_dim]

        # 4. 分类预测
        logits = self.classifier(fused_features)

        # 5. 计算正则化损失
        regularization_losses = self._compute_regularization_losses(
            expert_outputs, routing_weights, statistics, expert_metadata
        )

        # 6. 收集元数据
        metadata = {
            'routing_weights': routing_weights.squeeze(-1),  # [batch_size, num_experts]
            'expert_outputs': expert_outputs,
            'fused_features': fused_features,
            'statistics': statistics,
            'routing_info': routing_info,
            'expert_metadata': expert_metadata,
            'regularization_losses': regularization_losses,
            'logits': logits
        }

        # 7. 生成解释信息
        if return_explanations:
            explanations = self._generate_explanations(x, metadata)
            metadata['explanations'] = explanations

        # 8. 更新训练统计
        self._update_training_stats(routing_weights)

        return logits, metadata

    def _compute_regularization_losses(self,
                                     expert_outputs: torch.Tensor,
                                     routing_weights: torch.Tensor,
                                     statistics: torch.Tensor,
                                     expert_metadata: List = None) -> Dict[str, torch.Tensor]:
        """计算正则化损失"""
        losses = {}

        # 1. 路由稀疏性损失 (由路由器计算)
        if self.use_sparsity:
            losses['routing_sparsity'] = self.router.sparsity_loss_weight * \
                                       self.router._compute_sparsity_loss(routing_weights.squeeze(-1))

        # 2. 负载均衡损失 (由路由器计算)
        if self.use_load_balance:
            losses['load_balance'] = self.router.load_balance_loss_weight * \
                                    self.router._compute_load_balance_loss(routing_weights.squeeze(-1))

        # 3. 专家正交性损失
        losses['expert_orthogonal'] = self.orthogonal_constraint_weight * \
                                    self._compute_expert_orthogonal_loss(expert_outputs)

        # 4. 专家多样性损失
        losses['expert_diversity'] = self.expert_diversity_weight * \
                                    self._compute_expert_diversity_loss(expert_outputs)

        # 5. 频域约束损失
        if expert_metadata is not None:
            losses['frequency_constraint'] = self.frequency_constraint_weight * \
                                           self._compute_frequency_constraint_loss(expert_metadata)

        # 6. 总物理约束损失
        physics_loss = 0.0
        if 'frequency_constraint' in losses:
            physics_loss += losses['frequency_constraint']
        if 'expert_orthogonal' in losses:
            physics_loss += losses['expert_orthogonal']

        losses['physics_constraint'] = self.physics_constraint_weight * physics_loss

        return losses

    def _compute_expert_orthogonal_loss(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        """计算专家输出正交性损失"""
        # expert_outputs: [batch_size, num_experts, feature_dim]
        num_experts = expert_outputs.shape[1]

        orthogonal_loss = 0.0
        count = 0

        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                # 计算专家i和j输出的相似度
                output_i = expert_outputs[:, i, :]  # [batch_size, feature_dim]
                output_j = expert_outputs[:, j, :]  # [batch_size, feature_dim]

                # 归一化
                output_i_norm = F.normalize(output_i, dim=-1)
                output_j_norm = F.normalize(output_j, dim=-1)

                # 余弦相似度
                similarity = torch.mean(torch.sum(output_i_norm * output_j_norm, dim=-1))
                orthogonal_loss += similarity
                count += 1

        return orthogonal_loss / (count + 1e-8)

    def _compute_expert_diversity_loss(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        """计算专家多样性损失"""
        # 计算每个专家输出的方差，鼓励专家有不同的激活模式
        expert_variances = torch.var(expert_outputs, dim=0)  # [num_experts, feature_dim]
        diversity_loss = -torch.mean(expert_variances)  # 最大化方差
        return diversity_loss

    def _compute_frequency_constraint_loss(self, expert_metadata: List[Dict]) -> torch.Tensor:
        """计算频域约束损失

        确保每个专家的频率响应集中在预期频带内，实现物理机理约束
        """
        frequency_loss = 0.0

        for i, meta in enumerate(expert_metadata):
            expert_type = meta.get('expert_type', '')

            if expert_type == 'low_pass':
                # 低通专家：能量应该集中在低频段
                low_freq_energy = meta.get('low_freq_energy', torch.zeros(1))
                # 鼓励低频能量高，惩罚高频泄漏
                freq_loss = torch.mean(torch.relu(1.0 - low_freq_energy))  # 低频能量应该大于1

            elif expert_type == 'harmonic':
                # 谐波专家：应该在谐波频率处有明显响应
                spectrum_magnitude = meta.get('spectrum_magnitude', torch.zeros(1))
                # 计算谐波峰值集中度（简化版本）
                if hasattr(spectrum_magnitude, 'shape') and len(spectrum_magnitude.shape) > 1:
                    # 计算频谱的峰值因子
                    spectrum_mean = torch.mean(spectrum_magnitude, dim=-1, keepdim=True)
                    spectrum_std = torch.std(spectrum_magnitude, dim=-1, keepdim=True)
                    peak_factor = spectrum_std / (spectrum_mean + 1e-8)
                    # 鼓励明显的峰值（高峰值因子）
                    freq_loss = torch.mean(torch.relu(0.5 - peak_factor))
                else:
                    freq_loss = torch.tensor(0.0, device=spectrum_magnitude.device)

            elif expert_type == 'envelope':
                # 包络专家：应该对冲击信号敏感
                envelope_power = meta.get('envelope_power', torch.zeros(1))
                # 鼓励包络能量，说明有冲击成分
                freq_loss = torch.mean(torch.relu(0.1 - envelope_power))

            else:
                # 未知专家类型
                freq_loss = torch.tensor(0.0)

            frequency_loss += freq_loss

        return frequency_loss / len(expert_metadata)

    def compute_frequency_response_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """计算频率响应矩阵

        Returns:
            freq_matrix: [num_experts, num_freq_bins] 专家的频率响应特性
        """
        batch_size = min(x.shape[0], 10)  # 取前10个样本计算
        sample_signals = x[:batch_size]

        # 计算输入信号的频谱
        fft_signals = torch.fft.fft(sample_signals, dim=-1)
        spectrum_magnitude = torch.abs(fft_signals[:, :x.shape[-1]//2])
        freq_bins = x.shape[-1]//2

        freq_matrix = []

        for expert in self.experts:
            with torch.no_grad():
                expert_output, expert_meta = expert(sample_signals)

                # 如果专家有频谱信息，使用专家的频谱
                if 'spectrum_magnitude' in expert_meta:
                    expert_spectrum = expert_meta['spectrum_magnitude']
                else:
                    # 否则计算专家输出的频谱
                    expert_spectrum = torch.abs(torch.fft.fft(expert_output, dim=-1))
                    expert_spectrum = expert_spectrum[:, :freq_bins]

                # 确保维度匹配
                if expert_spectrum.shape[-1] != spectrum_magnitude.shape[-1]:
                    min_bins = min(expert_spectrum.shape[-1], spectrum_magnitude.shape[-1])
                    expert_spectrum = expert_spectrum[:, :min_bins]
                    spectrum_magnitude_slice = spectrum_magnitude[:, :min_bins]
                else:
                    spectrum_magnitude_slice = spectrum_magnitude

                # 计算相关性系数
                correlation = torch.mean(
                    expert_spectrum * spectrum_magnitude_slice, dim=0
                ) / (torch.std(spectrum_magnitude_slice, dim=0) + 1e-8)

                freq_matrix.append(correlation)

        if not freq_matrix:
            return torch.zeros(0, freq_bins, device=x.device)

        min_bins = min(t.shape[-1] for t in freq_matrix)
        aligned = [t[..., :min_bins] for t in freq_matrix]
        freq_matrix = torch.stack(aligned, dim=0)  # [num_experts, num_freq_bins]
        return freq_matrix

    def _generate_explanations(self, x: torch.Tensor, metadata: Dict) -> Dict:
        """生成可解释性分析"""
        batch_size = x.shape[0]
        routing_weights = metadata['routing_weights']  # [batch_size, num_experts]
        statistics = metadata['statistics']          # [batch_size, 15]

        explanations = {
            'path_signatures': [],
            'expert_activations': [],
            'feature_contributions': [],
            'sample_explanations': []
        }

        # 1. 路径签名分析
        for i in range(batch_size):
            # 当前样本的路径签名
            path_signature = {
                'sample_id': i,
                'expert_weights': routing_weights[i].detach().cpu().numpy().tolist(),
                'dominant_expert': torch.argmax(routing_weights[i]).item(),
                'expert_confidence': torch.max(routing_weights[i]).item(),
                'active_experts': (routing_weights[i] > 0.1).nonzero(as_tuple=True)[0].tolist(),
                'routing_entropy': self._compute_entropy(routing_weights[i])
            }
            explanations['path_signatures'].append(path_signature)

        # 2. 专家激活统计
        expert_activation_means = torch.mean(routing_weights, dim=0)
        expert_activation_stds = torch.std(routing_weights, dim=0)
        explanations['expert_activations'] = {
            'mean_weights': expert_activation_means.detach().cpu().numpy().tolist(),
            'std_weights': expert_activation_stds.detach().cpu().numpy().tolist(),
            'most_active_expert': torch.argmax(expert_activation_means).item(),
            'least_active_expert': torch.argmin(expert_activation_means).item()
        }

        # 3. 特征贡献分析
        # 分析统计特征对路由决策的贡献
        feature_importance = self.router._analyze_feature_importance(statistics, routing_weights)
        explanations['feature_contributions'] = {
            'importance_scores': feature_importance.detach().cpu().numpy().tolist(),
            'feature_names': self.router.feature_extractor.get_feature_names(),
            'most_important_features': torch.topk(feature_importance, 5)[1].tolist()
        }

        # 4. 样本级解释
        for i in range(min(5, batch_size)):  # 只解释前5个样本
            sample_exp = self._generate_sample_explanation(x[i], metadata, i)
            explanations['sample_explanations'].append(sample_exp)

        return explanations

    def _compute_entropy(self, weights: torch.Tensor) -> torch.Tensor:
        """计算权重分布的熵"""
        weights = weights.detach()
        weights = weights[weights > 1e-8]  # 过滤掉接近0的权重
        entropy = -torch.sum(weights * torch.log(weights + 1e-8))
        return entropy

    def _generate_sample_explanation(self, x: torch.Tensor, metadata: Dict, sample_idx: int) -> Dict:
        """生成单样本的详细解释"""
        routing_weights = metadata['routing_weights'][sample_idx]
        statistics = metadata['statistics'][sample_idx]
        dominant_expert = torch.argmax(routing_weights).item()

        # 获取专家的描述信息
        expert_descriptions = self.expert_descriptions

        # 生成解释
        explanation = {
            'sample_id': sample_idx,
            'signal_statistics': {
                'rms': statistics[6].item(),
                'kurtosis': statistics[11].item(),
                'peak_factor': statistics[7].item(),
                'spectral_centroid': statistics[14].item()
            },
            'routing_decision': {
                'selected_expert': expert_descriptions[dominant_expert].get('expert_name', f'Expert_{dominant_expert}'),
                'expert_confidence': routing_weights[dominant_expert].item(),
                'expert_target_faults': expert_descriptions[dominant_expert].get('target_faults', [])
            },
            'all_expert_weights': {
                expert_descriptions[i].get('expert_name', f'Expert_{i}'): routing_weights[i].item()
                for i in range(len(self.experts))
            },
            'physical_explanation': self._generate_physical_explanation(
                statistics, dominant_expert, expert_descriptions
            )
        }

        return explanation

    def _generate_physical_explanation(self, statistics: torch.Tensor,
                                      expert_id: int,
                                      expert_descriptions: List[Dict]) -> str:
        """生成物理解释"""
        rms = statistics[6].item()
        kurtosis = statistics[11].item()
        peak_factor = statistics[7].item()
        spectral_centroid = statistics[14].item()

        expert_desc = expert_descriptions[expert_id]
        expert_name = expert_desc.get('expert_name', f'Expert_{expert_id}')
        target_faults = expert_desc.get('target_faults', ['未知故障'])
        family = expert_desc.get('family', '')

        if family == 'low_pass':
            if spectral_centroid < 500:
                return f"信号主要集中在低频段({spectral_centroid:.1f}Hz)，能量较强(RMS={rms:.3f})，" \
                       f"符合{target_faults[0]}等低频故障特征。"
        elif family == 'harmonic':
            if 2.0 < peak_factor < 4.0:
                return f"信号呈现明显的周期性特征(峰值因子={peak_factor:.2f})，" \
                       f"频谱分布中等({spectral_centroid:.1f}Hz)，" \
                       f"符合{target_faults[0]}等谐波故障特征。"
        elif family == 'envelope':
            if kurtosis > 3.5 and peak_factor > 4.0:
                return f"信号具有强冲击特征(峭度={kurtosis:.2f}，峰值因子={peak_factor:.2f})，" \
                       f"高频成分丰富({spectral_centroid:.1f}Hz)，" \
                       f"符合{target_faults[0]}等冲击故障特征。"

        return f"信号特征激活了{expert_name}，主要针对{target_faults}等故障类型。"

    def _update_training_stats(self, routing_weights: torch.Tensor):
        """更新训练统计信息"""
        # 累积专家激活统计（使用运行平均）
        current_mean = torch.mean(routing_weights, dim=0).cpu()
        total_samples = self.training_stats['total_samples'] + routing_weights.shape[0]

        # 计算新的累积平均值
        if self.training_stats['total_samples'] == 0:
            self.training_stats['expert_activations'] = current_mean
        else:
            prev_weight = self.training_stats['total_samples'] / total_samples
            new_weight = routing_weights.shape[0] / total_samples
            self.training_stats['expert_activations'] = (
                prev_weight * self.training_stats['expert_activations'] +
                new_weight * current_mean
            )

        # 更新路由熵
        current_entropy = torch.mean(
            torch.stack([self._compute_entropy(w) for w in routing_weights])
        )
        if self.training_stats['total_samples'] == 0:
            self.training_stats['routing_entropy'] = current_entropy
        else:
            self.training_stats['routing_entropy'] = (
                prev_weight * self.training_stats['routing_entropy'] +
                new_weight * current_entropy
            )

        self.training_stats['total_samples'] = total_samples

    def get_training_summary(self) -> Dict:
        """获取训练统计摘要"""
        if self.training_stats['total_samples'] == 0:
            return {'message': 'No training data available'}

        total_samples = self.training_stats['total_samples']
        avg_activations = self.training_stats['expert_activations'] / total_samples
        avg_entropy = self.training_stats['routing_entropy'] / total_samples

        return {
            'total_samples': total_samples,
            'average_expert_activations': avg_activations.tolist(),
            'average_routing_entropy': avg_entropy,
            'most_used_expert': torch.argmax(avg_activations).item(),
            'routing_balance': 1.0 - torch.std(avg_activations).item()
        }

    def switch_to_blackbox_mode(self):
        """切换到黑盒模式（移除物理约束和规则调整）"""
        # 临时保存原始参数
        self.original_rule_adjustment = self.router._apply_rule_adjustment

        # 禁用规则调整
        def no_rule_adjustment(logits, stats):
            return logits
        self.router._apply_rule_adjustment = no_rule_adjustment

        # 设置路由温度为更高值（更均匀的分布）
        self.router.temperature = 5.0

        return "Switched to blackbox mode"

    def switch_to_physics_mode(self):
        """切换到物理同构模式"""
        # 恢复规则调整
        if hasattr(self, 'original_rule_adjustment'):
            self.router._apply_rule_adjustment = self.original_rule_adjustment

        # 恢复正常温度
        self.router.temperature = 1.0

        return "Switched to physics-constrained mode"

    def get_model_description(self) -> Dict:
        """获取模型描述"""
        return {
            'model_name': 'NNSPN-MoE',
            'version': '0.1.0',
            'num_experts': self.num_experts,
            'experts': [expert.get_expert_description() for expert in self.experts],
            'router_type': 'StatisticalRouter',
            'feature_dim': self.feature_dim,
            'num_classes': self.num_classes,
            'regularization': {
                'load_balance': self.use_load_balance,
                'sparsity': self.use_sparsity,
                'expert_orthogonal': True,
                'expert_diversity': True
            }
        }
