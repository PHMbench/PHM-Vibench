# 可解释性保持方法详细设计

## 可解释性保持核心原则

### 设计目标
在1D-2D融合过程中保持模型决策的透明度，使故障诊断结果不仅准确，而且可理解、可信任、可追溯。

### 可解释性层次划分
1. **数据可解释性** - 输入数据的理解和可视化
2. **特征可解释性** - 中间特征的表达和重要性
3. **决策可解释性** - 最终决策的归因和解释
4. **系统可解释性** - 整体架构的透明度

## 数据层面可解释性

### 1. **多模态数据可视化**
```python
class DataVisualizer:
    def __init__(self):
        self.color_map = 'viridis'

    def visualize_fusion_process(self, signal_1d, spectrogram_2d, attention_weights):
        """
        可视化1D-2D融合过程
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))

        # 1. 原始1D信号
        axes[0].plot(signal_1d)
        axes[0].set_title('1D时序信号')
        axes[0].set_xlabel('时间采样点')
        axes[0].set_ylabel('幅值')

        # 2. 2D频谱图
        im = axes[1].imshow(spectrogram_2d.T, aspect='auto', origin='lower',
                           cmap=self.color_map)
        axes[1].set_title('2D时频谱图')
        axes[1].set_xlabel('时间帧')
        axes[1].set_ylabel('频率bin')
        plt.colorbar(im, ax=axes[1])

        # 3. 融合注意力权重
        attention_map = attention_weights.reshape(-1, 1)
        im2 = axes[2].imshow(attention_map.T, aspect='auto', cmap='Reds')
        axes[2].set_title('融合注意力权重')
        axes[2].set_xlabel('时间/频率索引')
        axes[2].set_ylabel('注意力强度')
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        return fig
```

### 2. **数据增强可解释性**
```python
class ExplainableDataAugmentation:
    def __init__(self):
        self.noise_levels = [0.1, 0.2, 0.3]
        self.augment_types = ['time_warp', 'freq_mask', 'time_mask']

    def augment_with_explanation(self, signal_1d, spectrogram_2d):
        """
        带解释的数据增强
        """
        augmentations = []
        explanations = []

        # 时间扭曲
        warped_signal, warp_explanation = self.time_warp_with_explanation(signal_1d)
        augmentations.append(warped_signal)
        explanations.append({
            'type': 'time_warp',
            'explanation': warp_explanation,
            'purpose': '模拟转速变化对信号的影响'
        })

        # 频率掩码
        masked_spec, mask_explanation = self.freq_mask_with_explanation(spectrogram_2d)
        augmentations.append(masked_spec)
        explanations.append({
            'type': 'freq_mask',
            'explanation': mask_explanation,
            'purpose': '模拟传感器故障或频带丢失'
        })

        return augmentations, explanations
```

## 特征层面可解释性

### 1. **特征重要性可视化**
```python
class FeatureImportanceVisualizer:
    def __init__(self):
        self.attention_rollout = AttentionRollout()
        self.gradient_cam = GradientCAM()

    def compute_feature_importance(self, model, input_1d, input_2d, target_class):
        """
        计算多模态特征重要性
        """
        # 1D特征重要性
        feat_imp_1d = self.gradient_cam.compute_1d_importance(
            model, input_1d, target_class
        )

        # 2D特征重要性
        feat_imp_2d = self.gradient_cam.compute_2d_importance(
            model, input_2d, target_class
        )

        # 跨模态注意力重要性
        cross_attention = self.attention_rollout.get_cross_attention(
            model, input_1d, input_2d
        )

        return {
            '1d_importance': feat_imp_1d,
            '2d_importance': feat_imp_2d,
            'cross_modal_attention': cross_attention
        }

    def visualize_feature_importance(self, importance_dict):
        """
        可视化特征重要性
        """
        fig = plt.figure(figsize=(20, 10))

        # 1D特征重要性
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(importance_dict['1d_importance'])
        ax1.set_title('1D时域特征重要性')
        ax1.set_xlabel('时间位置')
        ax1.set_ylabel('重要性分数')

        # 2D特征重要性热力图
        ax2 = plt.subplot(2, 3, 2)
        im = ax2.imshow(importance_dict['2d_importance'], cmap='hot')
        ax2.set_title('2D频域特征重要性')
        plt.colorbar(im, ax=ax2)

        # 跨模态注意力
        ax3 = plt.subplot(2, 3, 3)
        sns.heatmap(importance_dict['cross_modal_attention'],
                   ax=ax3, cmap='Blues')
        ax3.set_title('跨模态注意力权重')

        # 融合特征重要性
        ax4 = plt.subplot(2, 3, 4)
        fused_importance = self.compute_fused_importance(importance_dict)
        ax4.bar(range(len(fused_importance)), fused_importance)
        ax4.set_title('融合特征重要性排序')
        ax4.set_xlabel('特征索引')
        ax4.set_ylabel('重要性')

        return fig
```

### 2. **特征演化可视化**
```python
class FeatureEvolutionTracker:
    def __init__(self):
        self.feature_history = []
        self.layer_names = []

    def track_features(self, model, input_data):
        """
        追踪特征在网络各层的演化
        """
        features = {}

        def hook_fn(module, input, output, name):
            features[name] = output.detach().clone()

        # 注册钩子
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                hook = module.register_forward_hook(
                    lambda m, i, o, n=name: hook_fn(m, i, o, n)
                )
                hooks.append(hook)

        # 前向传播
        with torch.no_grad():
            model(input_data)

        # 移除钩子
        for hook in hooks:
            hook.remove()

        return features

    def visualize_evolution(self, features_dict):
        """
        可视化特征演化过程
        """
        fig, axes = plt.subplots(2, len(features_dict), figsize=(20, 8))

        for idx, (layer_name, features) in enumerate(features_dict.items()):
            # 特征统计信息
            feat_mean = features.mean(dim=0).cpu().numpy()
            feat_std = features.std(dim=0).cpu().numpy()

            # 上层：特征均值演化
            axes[0, idx].plot(feat_mean)
            axes[0, idx].set_title(f'{layer_name} - 特征均值')
            axes[0, idx].set_xlabel('特征维度')

            # 下层：特征方差演化
            axes[1, idx].plot(feat_std)
            axes[1, idx].set_title(f'{layer_name} - 特征方差')
            axes[1, idx].set_xlabel('特征维度')

        plt.tight_layout()
        return fig
```

## 决策层面可解释性

### 1. **决策归因分析**
```python
class DecisionAttribution:
    def __init__(self):
        self.shap_explainer = None
        self.lime_explainer = None

    def explain_decision(self, model, input_1d, input_2d, target_class):
        """
        解释模型决策过程
        """
        explanations = {}

        # SHAP解释
        shap_values_1d = self.explain_with_shap_1d(model, input_1d, target_class)
        shap_values_2d = self.explain_with_shap_2d(model, input_2d, target_class)

        # LIME解释
        lime_exp = self.explain_with_lime(model, input_1d, input_2d)

        # 梯度解释
        grad_exp = self.explain_with_gradients(model, input_1d, input_2d)

        explanations['shap_1d'] = shap_values_1d
        explanations['shap_2d'] = shap_values_2d
        explanations['lime'] = lime_exp
        explanations['gradients'] = grad_exp

        return explanations

    def generate_text_explanation(self, explanations, prediction, confidence):
        """
        生成自然语言解释
        """
        # 提取关键特征
        top_features_1d = np.argsort(np.abs(explanations['shap_1d']))[-5:]
        top_features_2d = np.unravel_index(
            np.argsort(np.abs(explanations['shap_2d']).ravel())[-10:],
            explanations['shap_2d'].shape
        )

        # 生成解释文本
        explanation_text = f"""
        模型预测为故障类型: {prediction} (置信度: {confidence:.2%})

        关键决策依据:
        1. 时域信号中，位置 {top_features_1d} 附近的异常振动模式最为关键
        2. 频域中，频率范围 {top_features_2d[1]} 在时间段 {top_features_2d[0]} 表现出典型故障特征
        3. 跨模态融合权重显示，时域和频域信息的贡献比为 {self.get_modal_ratio(explanations)}

        决策可信度分析:
        - SHAP值一致性: {self.consistency_check(explanations)}
        - 与历史案例相似度: {self.similarity_to_history(explanations)}
        - 不确定性量化: {self.uncertainty_quantification(explanations)}
        """

        return explanation_text
```

### 2. **反事实解释**
```python
class CounterfactualExplanation:
    def __init__(self):
        self.cf_generator = CFGenerator()

    def generate_counterfactual(self, model, input_data, original_prediction, target_class):
        """
        生成反事实解释：什么样的输入会导致不同的预测？
        """
        # 寻找最小改变量
        cf_input = self.cf_generator.find_minimal_change(
            model, input_data, target_class
        )

        # 计算变化量
        delta_1d = cf_input['1d'] - input_data['1d']
        delta_2d = cf_input['2d'] - input_data['2d']

        # 生成解释
        explanation = {
            'original_prediction': original_prediction,
            'counterfactual_prediction': target_class,
            'minimal_changes': {
                '1d_changes': self.describe_1d_changes(delta_1d),
                '2d_changes': self.describe_2d_changes(delta_2d)
            },
            'change_magnitude': {
                '1d_norm': torch.norm(delta_1d).item(),
                '2d_norm': torch.norm(delta_2d).item()
            }
        }

        return explanation

    def describe_1d_changes(self, delta_1d):
        """
        描述1D信号的变化
        """
        changes = []

        # 找出最大变化区域
        threshold = torch.std(delta_1d) * 2
        significant_changes = torch.where(torch.abs(delta_1d) > threshold)[0]

        if len(significant_changes) > 0:
            changes.append(f"在时间点 {significant_changes[0]}-{significant_changes[-1]} 附近")
            changes.append(f"幅值变化约为 {torch.max(torch.abs(delta_1d)):.3f}")

        return changes
```

## 系统层面可解释性

### 1. **模型架构可视化**
```python
class ArchitectureVisualizer:
    def __init__(self):
        self.graphviz = GraphViz()

    def visualize_fusion_architecture(self, model):
        """
        可视化1D-2D融合架构
        """
        # 创建计算图
        graph = self.graphviz.Digraph()

        # 输入节点
        graph.node('input_1d', '1D时序输入', shape='box')
        graph.node('input_2d', '2D频谱输入', shape='box')

        # 特征提取层
        graph.node('feat_1d', '1D特征提取器', shape='ellipse')
        graph.node('feat_2d', '2D特征提取器', shape='ellipse')

        # 融合层
        graph.node('fusion', '特征融合模块', shape='diamond')

        # 分类器
        graph.node('classifier', '分类器', shape='ellipse')

        # 输出
        graph.node('output', '故障预测', shape='box')

        # 连接
        graph.edge('input_1d', 'feat_1d')
        graph.edge('input_2d', 'feat_2d')
        graph.edge('feat_1d', 'fusion')
        graph.edge('feat_2d', 'fusion')
        graph.edge('fusion', 'classifier')
        graph.edge('classifier', 'output')

        return graph
```

### 2. **可解释性评估指标**
```python
class ExplainabilityMetrics:
    def __init__(self):
        self.metrics = {
            'fidelity': self.compute_fidelity,
            'stability': self.compute_stability,
            'comprehensibility': self.compute_comprehensibility,
            'completeness': self.compute_completeness
        }

    def compute_fidelity(self, explanations, predictions):
        """
        保真度：解释与模型预测的一致性
        """
        # 通过扰动验证解释的准确性
        consistency_scores = []
        for exp, pred in zip(explanations, predictions):
            # 基于解释构造新输入
            perturbed_input = self.apply_explanation(exp)

            # 验证预测是否改变
            new_pred = self.model(perturbed_input)
            consistency = (new_pred == pred).float()
            consistency_scores.append(consistency)

        return torch.stack(consistency_scores).mean()

    def compute_stability(self, explanations):
        """
        稳定性：相似输入的解释相似性
        """
        n_explanations = len(explanations)
        stability_scores = []

        for i in range(n_explanations):
            for j in range(i+1, n_explanations):
                # 计算解释相似度
                similarity = self.explanation_similarity(
                    explanations[i], explanations[j]
                )
                stability_scores.append(similarity)

        return np.mean(stability_scores)

    def compute_comprehensibility(self, explanations):
        """
        可理解性：解释的复杂度和直观性
        """
        scores = []
        for exp in explanations:
            # 特征数量
            n_features = len(exp['important_features'])
            # 解释长度
            exp_length = len(exp['text_explanation'])
            # 可视化复杂度
            viz_complexity = exp['visualization_complexity']

            # 综合评分
            score = 1 / (1 + 0.1*n_features + 0.01*exp_length + 0.2*viz_complexity)
            scores.append(score)

        return np.mean(scores)
```

## 交互式可解释性系统

### 1. **用户界面设计**
```python
class InteractiveExplainer:
    def __init__(self):
        self.explainers = {
            'shap': SHAPExplainer(),
            'lime': LIMEExplainer(),
            'attention': AttentionExplainer(),
            'prototype': PrototypeExplainer()
        }

    def create_explanation_dashboard(self, model, input_data):
        """
        创建交互式解释仪表板
        """
        dashboard = Dashboard()

        # 1. 原始数据展示
        dashboard.add_section('原始数据', self.visualize_raw_data(input_data))

        # 2. 模型预测结果
        prediction = model.predict(input_data)
        dashboard.add_section('预测结果', self.show_prediction(prediction))

        # 3. 多种解释方法
        explanations = {}
        for method_name, explainer in self.explainers.items():
            exp = explainer.explain(model, input_data)
            explanations[method_name] = exp
            dashboard.add_section(f'{method_name}解释', exp)

        # 4. 对比分析
        dashboard.add_section('解释对比', self.compare_explanations(explanations))

        # 5. 交互式探索
        dashboard.add_section('探索模式', self.interactive_explore(model, input_data))

        return dashboard
```

---

*更新时间: 2025-11-19*
*状态: 可解释性方法设计完成*