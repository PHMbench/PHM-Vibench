"""
Theoretical Validation Experiments for Neural-Symbolic Theory
神经-符号理论验证实验

本脚本实现了对三个核心命题的验证实验：
1. 符号约束提升可靠性
2. 物理同构增强鲁棒性
3. 可解释性-性能权衡的帕累托边界
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import os
import sys

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from theory.neural_symbolic_constraints import NeuralSymbolicConstraints
from theory.interpretability_metrics import evaluate_model_interpretability


class SyntheticFaultDiagnosisModel(nn.Module):
    """合成的故障诊断模型，用于理论验证"""

    def __init__(self,
                 input_dim: int = 50,
                 hidden_dim: int = 100,
                 num_classes: int = 5,
                 use_symbolic_constraints: bool = False,
                 use_physics_informed: bool = False):
        """
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            num_classes: 类别数
            use_symbolic_constraints: 是否使用符号约束
            use_physics_informed: 是否使用物理信息
        """
        super(SyntheticFaultDiagnosisModel, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_symbolic_constraints = use_symbolic_constraints
        self.use_physics_informed = use_physics_informed

        # 基础网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 分类器
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

        # 物理信息层
        if use_physics_informed:
            self.physics_layer = self._create_physics_layer()

        # 符号约束
        if use_symbolic_constraints:
            self.constraints = NeuralSymbolicConstraints({
                'logical': {
                    'rules': [
                        "IF feature_1 > 0.5 THEN class == 0",
                        "IF feature_2 < -0.3 THEN class == 1",
                        "IF feature_3 > 0.2 AND feature_4 < 0.1 THEN class == 2"
                    ],
                    'weight': 0.1
                }
            })

    def _create_physics_layer(self) -> nn.Module:
        """创建物理信息层"""
        return nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5, padding=2),  # 模拟FFT
            nn.ReLU(),
            nn.Conv1d(4, 8, kernel_size=3, padding=1),  # 模拟滤波
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor, return_explanation: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入张量 (batch_size, input_dim)
            return_explanation: 是否返回解释信息

        Returns:
            outputs: 包含logits和解释的字典
        """
        # 特征提取
        features = self.feature_extractor(x)

        # 物理信息处理
        if self.use_physics_informed:
            x_reshaped = x.unsqueeze(1)  # (batch, 1, input_dim)
            physics_features = self.physics_layer(x_reshaped)
            physics_features = physics_features.view(physics_features.size(0), -1)
            # 动态调整分类器输入维度
            combined_dim = features.shape[1] + physics_features.shape[1]
            if not hasattr(self, 'adaptive_classifier'):
                self.adaptive_classifier = nn.Linear(combined_dim, self.num_classes)
            # 融合特征
            features = torch.cat([features, physics_features], dim=1)
            logits = self.adaptive_classifier(features)
        else:
            # 分类
            logits = self.classifier(features)

        outputs = {'logits': logits}

        # 生成解释
        if return_explanation:
            explanation = self._generate_explanation(x, features, logits)
            outputs.update(explanation)

        return outputs

    def _generate_explanation(self,
                            x: torch.Tensor,
                            features: torch.Tensor,
                            logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """生成解释信息"""
        # 特征重要性（基于梯度）
        if x.requires_grad:
            grad_outputs = torch.ones_like(logits)
            gradients = torch.autograd.grad(
                outputs=logits,
                inputs=x,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0]
            feature_importance = torch.mean(torch.abs(gradients), dim=0)
        else:
            # 简化的特征重要性
            feature_importance = torch.abs(features).mean(dim=0)

        # 规则激活
        rule_activations = self._compute_rule_activations(x)

        return {
            'feature_importance': feature_importance,
            'rule_activations': rule_activations
        }

    def _compute_rule_activations(self, x: torch.Tensor) -> torch.Tensor:
        """计算规则激活度"""
        # 简化的规则激活计算
        activations = torch.zeros(x.size(0), 3, device=x.device)

        # 规则1: feature_1 > 0.5
        activations[:, 0] = F.relu(x[:, 0] - 0.5)

        # 规则2: feature_2 < -0.3
        activations[:, 1] = F.relu(-0.3 - x[:, 1])

        # 规则3: feature_3 > 0.2 AND feature_4 < 0.1
        activations[:, 2] = F.relu(x[:, 2] - 0.2) * F.relu(0.1 - x[:, 3])

        return activations


def generate_synthetic_data(num_samples: int = 1000,
                           input_dim: int = 50,
                           num_classes: int = 5,
                           noise_level: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成合成故障诊断数据"""
    np.random.seed(42)
    torch.manual_seed(42)

    # 生成基础特征
    X = torch.randn(num_samples, input_dim)

    # 生成标签（基于特征的简单规则）
    y = torch.zeros(num_samples, dtype=torch.long)

    # 类别0: feature_1 > 0.5
    mask0 = X[:, 0] > 0.5
    y[mask0] = 0

    # 类别1: feature_2 < -0.3
    mask1 = X[:, 1] < -0.3
    y[mask1] = 1

    # 类别2: feature_3 > 0.2 AND feature_4 < 0.1
    mask2 = (X[:, 2] > 0.2) & (X[:, 3] < 0.1)
    y[mask2] = 2

    # 剩余样本分配到其他类别
    remaining = ~(mask0 | mask1 | mask2)
    y[remaining] = torch.randint(3, num_classes, (remaining.sum(),))

    # 添加噪声
    if noise_level > 0:
        X += torch.randn_like(X) * noise_level

    return X, y


class PropositionValidator:
    """命题验证器"""

    def __init__(self):
        self.results = {}

    def validate_proposition_1(self,
                              num_experiments: int = 10) -> Dict[str, float]:
        """
        验证命题1：符号约束提升可靠性

        实验：对比有/无符号约束模型的可靠性
        """
        print("\n=== 验证命题1：符号约束提升可靠性 ===")

        reliability_scores = {'with_constraints': [], 'without_constraints': []}

        for exp in range(num_experiments):
            # 生成数据
            train_data, train_labels = generate_synthetic_data(num_samples=800)
            test_data, test_labels = generate_synthetic_data(num_samples=200)

            # 训练无约束模型
            model_without = SyntheticFaultDiagnosisModel(
                use_symbolic_constraints=False,
                use_physics_informed=False
            )
            acc_without = self._train_and_evaluate(
                model_without, train_data, train_labels, test_data, test_labels
            )
            reliability_scores['without_constraints'].append(acc_without)

            # 训练有符号约束模型
            model_with = SyntheticFaultDiagnosisModel(
                use_symbolic_constraints=True,
                use_physics_informed=False
            )
            acc_with = self._train_and_evaluate(
                model_with, train_data, train_labels, test_data, test_labels
            )
            reliability_scores['with_constraints'].append(acc_with)

        # 计算统计结果
        mean_without = np.mean(reliability_scores['without_constraints'])
        mean_with = np.mean(reliability_scores['with_constraints'])
        improvement = mean_with - mean_without

        results = {
            'reliability_without': mean_without,
            'reliability_with': mean_with,
            'improvement': improvement,
            'improvement_percentage': (improvement / mean_without * 100) if mean_without > 0 else 0
        }

        print(f"  无约束模型可靠性: {mean_without:.4f}")
        print(f"  有约束模型可靠性: {mean_with:.4f}")
        print(f"  提升幅度: {improvement:.4f} ({results['improvement_percentage']:.2f}%)")

        self.results['proposition_1'] = results
        return results

    def validate_proposition_2(self,
                              num_experiments: int = 10,
                              noise_levels: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5]) -> Dict[str, List[float]]:
        """
        验证命题2：物理同构增强鲁棒性

        实验：对比物理同构模型在噪声下的性能下降
        """
        print("\n=== 验证命题2：物理同构增强鲁棒性 ===")

        performance_drop = {'physics_informed': [], 'standard': []}

        for noise_level in noise_levels:
            print(f"\n噪声水平: {noise_level}")

            # 生成噪声数据
            train_data, train_labels = generate_synthetic_data(noise_level=0.0)
            test_data, test_labels = generate_synthetic_data(noise_level=noise_level)

            # 训练标准模型
            model_standard = SyntheticFaultDiagnosisModel(
                use_symbolic_constraints=False,
                use_physics_informed=False
            )
            acc_standard = self._train_and_evaluate(
                model_standard, train_data, train_labels, test_data, test_labels
            )

            # 训练物理信息模型
            model_physics = SyntheticFaultDiagnosisModel(
                use_symbolic_constraints=False,
                use_physics_informed=True
            )
            acc_physics = self._train_and_evaluate(
                model_physics, train_data, train_labels, test_data, test_labels
            )

            performance_drop['standard'].append(acc_standard)
            performance_drop['physics_informed'].append(acc_physics)

            print(f"  标准模型准确率: {acc_standard:.4f}")
            print(f"  物理模型准确率: {acc_physics:.4f}")

        # 计算性能下降斜率
        std_drops = [performance_drop['standard'][0] - p for p in performance_drop['standard']]
        phy_drops = [performance_drop['physics_informed'][0] - p for p in performance_drop['physics_informed']]

        results = {
            'performance_standard': performance_drop['standard'],
            'performance_physics': performance_drop['physics_informed'],
            'drop_rate_standard': np.mean(std_drops[1:]) / noise_levels[-1] if noise_levels[-1] > 0 else 0,
            'drop_rate_physics': np.mean(phy_drops[1:]) / noise_levels[-1] if noise_levels[-1] > 0 else 0,
            'noise_levels': noise_levels
        }

        print(f"\n性能下降率:")
        print(f"  标准模型: {results['drop_rate_standard']:.4f}")
        print(f"  物理模型: {results['drop_rate_physics']:.4f}")

        self.results['proposition_2'] = results
        return results

    def validate_proposition_3(self,
                              model_configs: List[Dict]) -> Dict[str, List[float]]:
        """
        验证命题3：可解释性-性能权衡的帕累托边界

        实验：评估不同配置模型的性能与可解释性
        """
        print("\n=== 验证命题3：可解释性-性能权衡的帕累托边界 ===")

        performance_scores = []
        interpretability_scores = []

        # 生成测试数据
        _, test_data = generate_synthetic_data(num_samples=200)

        for i, config in enumerate(model_configs):
            print(f"\n评估配置 {i+1}/{len(model_configs)}")

            # 创建模型
            model = SyntheticFaultDiagnosisModel(**config)

            # 简化的可解释性评分
            interp_score = self._compute_interpretability_score(config)
            interpretability_scores.append(interp_score)

            # 简化的性能评估（不训练）
            model.eval()
            with torch.no_grad():
                outputs = model(test_data.float())
                pred = torch.argmax(outputs['logits'], dim=1)
                accuracy = (pred == test_labels).float().mean().item()
            performance_scores.append(accuracy)

            print(f"  配置: {config}")
            print(f"  性能: {accuracy:.4f}")
            print(f"  可解释性: {interp_score:.4f}")

        # 识别帕累托前沿
        pareto_indices = self._find_pareto_front(performance_scores, interpretability_scores)

        results = {
            'performance': performance_scores,
            'interpretability': interpretability_scores,
            'pareto_front': pareto_indices,
            'configurations': model_configs
        }

        print(f"\n帕累托前沿包含 {len(pareto_indices)} 个配置")

        self.results['proposition_3'] = results
        return results

    def _train_and_evaluate(self,
                           model: nn.Module,
                           train_data: torch.Tensor,
                           train_labels: torch.Tensor,
                           test_data: torch.Tensor,
                           test_labels: torch.Tensor,
                           num_epochs: int = 20) -> float:
        """训练并评估模型"""
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # 训练
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(train_data)
            loss = criterion(outputs['logits'], train_labels)

            # 添加约束损失
            if hasattr(model, 'constraints'):
                constraint_loss = model.constraints(train_data, train_data, outputs['logits'])
                loss += constraint_loss['total']

            loss.backward()
            optimizer.step()

        # 评估
        model.eval()
        with torch.no_grad():
            outputs = model(test_data)
            pred = torch.argmax(outputs['logits'], dim=1)
            accuracy = (pred == test_labels).float().mean().item()

        return accuracy

    def _compute_interpretability_score(self, config: Dict) -> float:
        """计算可解释性分数"""
        score = 5.0  # 基础分数

        # 符号约束增加可解释性
        if config.get('use_symbolic_constraints', False):
            score += 1.5

        # 物理信息略微增加可解释性
        if config.get('use_physics_informed', False):
            score += 0.5

        # 模型大小降低可解释性
        if config.get('hidden_dim', 100) > 150:
            score -= 1.0
        elif config.get('hidden_dim', 100) < 50:
            score += 0.5

        return max(1.0, min(5.0, score))

    def _find_pareto_front(self,
                          performance: List[float],
                          interpretability: List[float]) -> List[int]:
        """找到帕累托前沿的索引"""
        pareto_indices = []
        n = len(performance)

        for i in range(n):
            dominated = False
            for j in range(n):
                if i != j:
                    # 如果j在两个维度上都优于或等于i，且至少一个严格优于
                    if (performance[j] >= performance[i] and
                        interpretability[j] >= interpretability[i] and
                        (performance[j] > performance[i] or interpretability[j] > interpretability[i])):
                        dominated = True
                        break

            if not dominated:
                pareto_indices.append(i)

        return pareto_indices

    def generate_report(self, output_dir: str = './results'):
        """生成验证报告"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存结果
        with open(os.path.join(output_dir, 'validation_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)

        # 生成可视化
        self._plot_proposition_1_results(output_dir)
        self._plot_proposition_2_results(output_dir)
        self._plot_proposition_3_results(output_dir)

        print(f"\n验证报告已保存到: {output_dir}")

    def _plot_proposition_1_results(self, output_dir: str):
        """绘制命题1结果"""
        if 'proposition_1' not in self.results:
            return

        plt.figure(figsize=(8, 6))
        results = self.results['proposition_1']

        models = ['无约束模型', '有约束模型']
        scores = [results['reliability_without'], results['reliability_with']]

        bars = plt.bar(models, scores, color=['lightcoral', 'lightblue'])
        plt.title('命题1验证：符号约束对可靠性的影响')
        plt.ylabel('可靠性（准确率）')
        plt.ylim(0, 1)

        # 添加数值标签
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.4f}', ha='center', va='bottom')

        plt.savefig(os.path.join(output_dir, 'proposition_1_validation.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_proposition_2_results(self, output_dir: str):
        """绘制命题2结果"""
        if 'proposition_2' not in self.results:
            return

        plt.figure(figsize=(8, 6))
        results = self.results['proposition_2']

        noise_levels = results['noise_levels']
        perf_std = results['performance_standard']
        perf_phy = results['performance_physics_informed']

        plt.plot(noise_levels, perf_std, 'o-', label='标准模型', color='red')
        plt.plot(noise_levels, perf_phy, 's-', label='物理信息模型', color='blue')

        plt.title('命题2验证：物理同构对噪声鲁棒性的影响')
        plt.xlabel('噪声水平')
        plt.ylabel('性能（准确率）')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(os.path.join(output_dir, 'proposition_2_validation.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_proposition_3_results(self, output_dir: str):
        """绘制命题3结果"""
        if 'proposition_3' not in self.results:
            return

        plt.figure(figsize=(10, 8))
        results = self.results['proposition_3']

        performance = results['performance']
        interpretability = results['interpretability']
        pareto_indices = results['pareto_front']

        # 所有点
        plt.scatter(performance, interpretability, c='gray', alpha=0.5, s=100, label='所有配置')

        # 帕累托前沿
        pareto_perf = [performance[i] for i in pareto_indices]
        pareto_interp = [interpretability[i] for i in pareto_indices]
        plt.scatter(pareto_perf, pareto_interp, c='red', s=150, label='帕累托最优', marker='*')

        # 拟合帕累托边界
        if len(pareto_perf) > 2:
            z = np.polyfit(pareto_perf, pareto_interp, 2)
            p = np.poly1d(z)
            x_fit = np.linspace(min(performance), max(performance), 100)
            plt.plot(x_fit, p(x_fit), 'r--', alpha=0.5, label='拟合边界')

        plt.title('命题3验证：性能-可解释性权衡与帕累托边界')
        plt.xlabel('性能（准确率）')
        plt.ylabel('可解释性评分')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(os.path.join(output_dir, 'proposition_3_validation.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """主函数：运行所有验证实验"""
    print("开始神经-符号理论验证实验...")

    # 创建验证器
    validator = PropositionValidator()

    # 验证命题1
    validator.validate_proposition_1(num_experiments=10)

    # 验证命题2
    validator.validate_proposition_2(num_experiments=5, noise_levels=[0.0, 0.1, 0.2, 0.3, 0.5])

    # 验证命题3
    model_configs = [
        {'use_symbolic_constraints': False, 'use_physics_informed': False, 'hidden_dim': 200},
        {'use_symbolic_constraints': True, 'use_physics_informed': False, 'hidden_dim': 100},
        {'use_symbolic_constraints': False, 'use_physics_informed': True, 'hidden_dim': 150},
        {'use_symbolic_constraints': True, 'use_physics_informed': True, 'hidden_dim': 100},
        {'use_symbolic_constraints': True, 'use_physics_informed': False, 'hidden_dim': 50},
        {'use_symbolic_constraints': True, 'use_physics_informed': True, 'hidden_dim': 50},
    ]
    validator.validate_proposition_3(model_configs)

    # 生成报告
    validator.generate_report('./Paper/Neuralsymbolic_theory/results/theory_validation')

    print("\n所有验证实验完成！")


if __name__ == "__main__":
    main()