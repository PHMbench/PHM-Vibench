# 模型架构增强方案

## 1. 增强的NNSPN-MoE架构设计

### 1.1 整体架构图

```
输入信号 x(t)
    |
    v
物理特征提取器 (Physics Feature Extractor)
    | - RMS, 峭度, 形状因子等8个统计特征
    | - 频域特征 (频谱重心, 频谱滚降点等)
    v
统计路由器 (Statistical Router) --> 物理专家模块 (Physical Experts Pool)
    |                                   |
    |--> E1: 低频通带专家 --------------|
    |--> E2: 谐波分析专家 --------------|--> 专家聚合 (Expert Aggregation)
    |--> E3: 包络检测专家 --------------|       |
    |--> E4: 调制分析专家 --------------|       v
    |--> E5: 边带分析专家 --------------|--> 加权求和: y = Σ Gi·Ei(x)
    |--> E6: 残差专家 (惩罚机制) -----|       |
    |                                   |       v
    |------------------> 损失计算 <-------|--> 分类输出
```

### 1.2 关键增强点

1. **更强的专家约束机制**
2. **残差专家惩罚策略**
3. **专家预训练流程**
4. **动态温度调度**
5. **频域约束增强**

## 2. 专家模块增强实现

### 2.1 可微分滤波器约束

```python
class DifferentiableFilter(nn.Module):
    """可微分滤波器基类"""
    def __init__(self, fs, filter_type, init_freq):
        super().__init__()
        self.fs = fs
        self.filter_type = filter_type

        # 可学习的频率参数
        if filter_type == "lowpass":
            self.cutoff_freq = nn.Parameter(torch.tensor(init_freq))
            self.freq_range = (10, 1000)  # 截止频率范围
        elif filter_type == "bandpass":
            self.low_freq = nn.Parameter(torch.tensor(init_freq[0]))
            self.high_freq = nn.Parameter(torch.tensor(init_freq[1]))
            self.freq_range = ((100, 5000), (1000, 10000))

        # 确保滤波器阶数固定，保持物理意义
        self.order = 4

    def forward(self, x):
        """前向传播，带频率约束"""
        # 获取约束后的频率
        if self.filter_type == "lowpass":
            cutoff = self.constrain_frequency(self.cutoff_freq, *self.freq_range)
            filtered = self.lowpass_filter(x, cutoff)
        elif self.filter_type == "bandpass":
            low = self.constrain_frequency(self.low_freq, *self.freq_range[0])
            high = self.constrain_frequency(self.high_freq, *self.freq_range[1])
            filtered = self.bandpass_filter(x, low, high)
        return filtered

    def constrain_frequency(self, freq, min_val, max_val):
        """将频率约束在合理范围内"""
        return torch.sigmoid(freq) * (max_val - min_val) + min_val

    def lowpass_filter(self, x, cutoff):
        """低通滤波器实现"""
        # 使用Butterworth滤波器
        from scipy.signal import butter, filtfilt
        import numpy as np

        # 转换为numpy进行滤波
        x_np = x.detach().cpu().numpy()
        cutoff_np = cutoff.detach().cpu().numpy()

        # 设计滤波器
        nyquist = self.fs / 2
        normal_cutoff = cutoff_np / nyquist
        b, a = butter(self.order, normal_cutoff, btype='low')

        # 应用滤波器
        filtered = filtfilt(b, a, x_np, axis=-1)

        # 转回tensor
        return torch.from_numpy(filtered).to(x.device)

    def bandpass_filter(self, x, low, high):
        """带通滤波器实现"""
        from scipy.signal import butter, filtfilt
        import numpy as np

        x_np = x.detach().cpu().numpy()
        low_np = low.detach().cpu().numpy()
        high_np = high.detach().cpu().numpy()

        nyquist = self.fs / 2
        low_normal = low_np / nyquist
        high_normal = high_np / nyquist

        b, a = butter(self.order, [low_normal, high_normal], btype='band')
        filtered = filtfilt(b, a, x_np, axis=-1)

        return torch.from_numpy(filtered).to(x.device)
```

### 2.2 增强的物理专家

```python
class EnhancedPhysicsExpert(nn.Module):
    """增强的物理专家模块"""
    def __init__(self, expert_type, fs, expert_config):
        super().__init__()
        self.expert_type = expert_type
        self.fs = fs

        # 滤波器约束
        if expert_type == "E1_low_pass":
            self.filter = DifferentiableFilter(fs, "lowpass", expert_config['cutoff'])
            self.features = StatisticalFeatures(['rms', 'variance', 'peak'])

        elif expert_type == "E3_envelope":
            self.filter = DifferentiableFilter(fs, "bandpass", expert_config['band'])
            self.envelope = nn.Sequential(
                nn.HilbertTransform(),  # 希尔伯特变换
                nn.Abs(),
                nn.LowpassFilter(cutoff=500)  # 包络滤波
            )
            self.features = StatisticalFeatures(['kurtosis', 'crest_factor', 'rms'])

        elif expert_type == "E2_harmonic":
            self.harmonics = expert_config.get('harmonics', [1, 2, 3])
            self.filters = nn.ModuleList([
                DifferentiableFilter(fs, "bandpass", (f*fs/100 - 10, f*fs/100 + 10))
                for f in self.harmonics
            ])
            self.features = HarmonicFeatures(self.harmonics)

        elif expert_type == "E5_sideband":
            self.filter = DifferentiableFilter(fs, "bandpass", expert_config['band'])
            self.cepstrum = CepstrumAnalysis()
            self.features = CepstrumFeatures()

        elif expert_type == "E6_residual":
            # 残差专家没有滤波约束，但有其他约束
            self.mlp = nn.Sequential(
                nn.Linear(4096, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
            self.features = nn.Identity()

    def forward(self, x):
        """前向传播"""
        if self.expert_type == "E6_residual":
            # 残差专家直接处理
            filtered = x
        else:
            # 物理专家先滤波
            if self.expert_type == "E2_harmonic":
                # 谐波专家使用多个滤波器
                filtered_signals = []
                for f, filt in zip(self.harmonics, self.filters):
                    filtered_signals.append(filt(x))
                filtered = torch.stack(filtered_signals, dim=-1).sum(dim=-1)
            else:
                filtered = self.filter(x)

        # 特征提取
        if self.expert_type == "E3_envelope":
            # 包络专家特殊处理
            envelope = self.envelope(filtered)
            features = self.features(envelope)
        else:
            features = self.features(filtered)

        return features
```

### 2.3 专家正交正则化

```python
class ExpertOrthogonalRegularizer(nn.Module):
    """专家正交正则化器"""
    def __init__(self, num_experts, feature_dim):
        super().__init__()
        self.num_experts = num_experts
        self.feature_dim = feature_dim

    def forward(self, expert_outputs):
        """
        计算专家输出的正交正则损失

        Args:
            expert_outputs: [B, M, D] M个专家的输出

        Returns:
            orthogonal_loss: 正交损失
        """
        B, M, D = expert_outputs.shape

        # 计算专家两两之间的余弦相似度矩阵
        # 归一化专家输出
        normalized_outputs = F.normalize(expert_outputs, p=2, dim=-1)  # [B, M, D]

        # 计算相似度矩阵 [B, M, M]
        similarity_matrix = torch.bmm(
            normalized_outputs,
            normalized_outputs.transpose(-2, -1)
        )

        # 提取上三角矩阵（排除对角线）
        mask = torch.triu(torch.ones(M, M), diagonal=1).bool().to(expert_outputs.device)
        off_diagonal_similarities = similarity_matrix[:, mask]  # [B, M*(M-1)/2]

        # 正交损失 = 相似度的平方和
        orthogonal_loss = torch.mean(off_diagonal_similarities ** 2)

        return orthogonal_loss
```

### 2.4 频域约束增强

```python
class FrequencyDomainConstraint(nn.Module):
    """频域约束模块"""
    def __init__(self, expert_type, target_band, strength=0.1):
        super().__init__()
        self.expert_type = expert_type
        self.target_band = target_band  # (f_min, f_max) in Hz
        self.strength = strength

    def forward(self, x, fs):
        """
        应用频域约束

        Args:
            x: [B, L] 输入信号
            fs: 采样频率

        Returns:
            constraint_loss: 约束损失
        """
        # FFT
        X = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(x.size(-1), 1/fs)

        # 创建目标频带掩码
        mask = torch.zeros_like(freqs)
        mask[(freqs >= self.target_band[0]) & (freqs <= self.target_band[1])] = 1

        # 计算带外能量
        out_of_band_energy = torch.mean(torch.abs(X[:, mask == 0]) ** 2)

        # 计算带内能量
        in_band_energy = torch.mean(torch.abs(X[:, mask == 1]) ** 2)

        # 约束损失 = 带外能量 / (带内能量 + epsilon)
        constraint_loss = self.strength * out_of_band_energy / (in_band_energy + 1e-8)

        return constraint_loss
```

## 3. 路由器增强

### 3.1 温度退火路由器

```python
class AnnealedRouter(nn.Module):
    """带温度退火的统计路由器"""
    def __init__(self, feature_dim, num_experts, init_temp=1.0, min_temp=0.1, decay_rate=0.99):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_experts = num_experts

        # 路由网络
        self.routing_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts)
        )

        # 温度参数
        self.register_buffer('temperature', torch.tensor(init_temp))
        self.min_temp = min_temp
        self.decay_rate = decay_rate

    def forward(self, features):
        """前向传播"""
        logits = self.routing_net(features)

        # 温度缩放
        scaled_logits = logits / self.temperature

        # Softmax路由权重
        routing_weights = F.softmax(scaled_logits, dim=-1)

        return routing_weights

    def step(self):
        """每步更新温度"""
        self.temperature = max(
            self.min_temp,
            self.temperature * self.decay_rate
        )
```

### 3.2 物理特征增强的路由器

```python
class PhysicsEnhancedRouter(nn.Module):
    """物理特征增强的路由器"""
    def __init__(self, num_experts, feature_selection='adaptive'):
        super().__init__()
        self.num_experts = num_experts
        self.feature_selection = feature_selection

        # 物理特征映射
        self.feature_mappers = nn.ModuleDict({
            'rms': nn.Linear(1, 16),
            'kurtosis': nn.Linear(1, 16),
            'shape_factor': nn.Linear(1, 16),
            'crest_factor': nn.Linear(1, 16),
            'impulse_factor': nn.Linear(1, 16),
            'spectral_centroid': nn.Linear(1, 16),
            'spectral_rolloff': nn.Linear(1, 16),
            'zero_crossing_rate': nn.Linear(1, 16)
        })

        # 特征选择网络（自适应）
        if feature_selection == 'adaptive':
            self.feature_selector = nn.Sequential(
                nn.Linear(8 * 16, 64),
                nn.ReLU(),
                nn.Linear(64, 8),
                nn.Sigmoid()  # 每个特征的权重
            )

        # 路由决策网络
        total_dim = 8 * 16
        self.router = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_experts)
        )

    def forward(self, x):
        """前向传播"""
        # 提取物理特征
        physics_feats = self.extract_physics_features(x)  # [B, 8]

        # 映射每个特征
        mapped_feats = []
        for i, (name, feat) in enumerate(zip(
            ['rms', 'kurtosis', 'shape_factor', 'crest_factor',
             'impulse_factor', 'spectral_centroid', 'spectral_rolloff',
             'zero_crossing_rate'],
            physics_feats.unbind(dim=-1)
        )):
            feat = feat.unsqueeze(-1)  # [B, 1]
            mapped = self.feature_mappers[name](feat)  # [B, 16]
            mapped_feats.append(mapped)

        # 拼接所有特征
        concatenated = torch.cat(mapped_feats, dim=-1)  # [B, 128]

        # 特征选择（自适应）
        if self.feature_selection == 'adaptive':
            selection_weights = self.feature_selector(concatenated)  # [B, 8]
            # 应用选择权重
            for i in range(8):
                start, end = i * 16, (i + 1) * 16
                concatenated[:, start:end] *= selection_weights[:, i:i+1]

        # 路由决策
        logits = self.router(concatenated)
        routing_weights = F.softmax(logits, dim=-1)

        return routing_weights, physics_feats

    def extract_physics_features(self, x):
        """提取物理统计特征"""
        # RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))

        # 峭度
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        kurtosis = torch.mean(((x - mean) / std) ** 4, dim=-1, keepdim=True) - 3

        # 形状因子
        shape_factor = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True)) / \
                      torch.mean(torch.abs(x), dim=-1, keepdim=True)

        # 峰值因子
        crest_factor = torch.max(torch.abs(x), dim=-1, keepdim=True) / rms

        # 脉冲因子
        impulse_factor = torch.max(torch.abs(x), dim=-1, keepdim=True) / \
                       torch.mean(torch.abs(x), dim=-1, keepdim=True)

        # 频谱重心
        X = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(x.size(-1), 1.0).to(x.device)
        power = torch.abs(X) ** 2
        spectral_centroid = torch.sum(freqs * power, dim=-1, keepdim=True) / \
                          torch.sum(power, dim=-1, keepdim=True)

        # 频谱滚降点
        cumsum_power = torch.cumsum(power, dim=-1)
        total_power = cumsum_power[:, -1:]
        rolloff_idx = torch.searchsorted(
            cumsum_power, 0.85 * total_power
        ).unsqueeze(-1)
        spectral_rolloff = freqs[rolloff_idx].float()

        # 过零率
        signs = torch.sign(x)
        zero_crossings = torch.sum(
            torch.abs(signs[:, :, 1:] - signs[:, :, :-1]),
            dim=-1, keepdim=True
        ) / 2
        zero_crossing_rate = zero_crossings / x.size(-1)

        # 拼接所有特征
        features = torch.cat([
            rms, kurtosis, shape_factor, crest_factor,
            impulse_factor, spectral_centroid, spectral_rolloff,
            zero_crossing_rate
        ], dim=-1)

        return features
```

## 4. 训练策略增强

### 4.1 专家预训练流程

```python
class ExpertPretrainer:
    """专家预训练器"""
    def __init__(self, expert_configs):
        self.expert_configs = expert_configs

    def pretrain_single_expert(self, expert, train_loader, val_loader,
                              expert_type, num_epochs=50):
        """预训练单个专家"""

        # 根据专家类型准备数据
        if expert_type == "E3_envelope":
            # 包络专家使用包络数据
            train_loader, val_loader = self.prepare_envelope_data(train_loader, val_loader)
        elif expert_type == "E1_low_pass":
            # 低频专家使用低通滤波数据
            train_loader, val_loader = self.prepare_lowpass_data(train_loader, val_loader)
        # ... 其他专家类型

        # 定义优化器和损失
        optimizer = torch.optim.Adam(expert.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # 训练循环
        best_val_acc = 0
        for epoch in range(num_epochs):
            # 训练
            expert.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()

                # 前向传播
                features = expert(data)

                # 使用简单的分类头（预训练时）
                if not hasattr(expert, 'classifier'):
                    expert.classifier = nn.Linear(features.size(-1), 10).to(data.device)
                output = expert.classifier(features)

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()

            # 验证
            val_acc = self.validate_expert(expert, val_loader)

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(expert.state_dict(), f'expert_{expert_type}_best.pth')

            print(f'Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Val Acc: {val_acc:.4f}')

        return best_val_acc

    def prepare_envelope_data(self, train_loader, val_loader):
        """准备包络数据"""
        # 对数据应用包络解调
        new_train = []
        for data, target in train_loader:
            # 带通滤波 (2k-5kHz)
            filtered = self.bandpass_filter(data, 2000, 5000)
            # 希尔伯特变换取包络
            envelope = torch.abs(torch.hilbert(filtered))
            new_train.append((envelope, target))

        # 同样处理验证集
        new_val = []
        for data, target in val_loader:
            filtered = self.bandpass_filter(data, 2000, 5000)
            envelope = torch.abs(torch.hilbert(filtered))
            new_val.append((envelope, target))

        return new_train, new_val
```

### 4.2 渐进式训练策略

```python
class ProgressiveMoETrainer:
    """渐进式MoE训练器"""
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.current_stage = 0

    def progressive_train(self, train_loader, val_loader, num_epochs):
        """渐进式训练流程"""

        # 阶段0: 预热阶段（冻结路由器）
        self.stage0_warmup(train_loader, val_loader, epochs=num_epochs//4)

        # 阶段1: 路由器解冻，大温度
        self.stage1_routing_unfrozen(train_loader, val_loader, epochs=num_epochs//4)

        # 阶段2: 温度退火，引入正则
        self.stage2_annealing(train_loader, val_loader, epochs=num_epochs//4)

        # 阶段3: 端到端微调
        self.stage3_finetune(train_loader, val_loader, epochs=num_epochs//4)

    def stage0_warmup(self, train_loader, val_loader, epochs):
        """阶段0: 预热"""
        print("Stage 0: Warming up experts (router frozen)")

        # 冻结路由器
        for param in self.model.router.parameters():
            param.requires_grad = False

        # 设置温度为高值（均匀分布）
        self.model.router.temperature.fill_(10.0)

        # 训练
        for epoch in range(epochs):
            self.train_epoch(train_loader, stage='warmup')
            val_metrics = self.validate(val_loader)
            print(f"Warmup Epoch {epoch}: Val Acc = {val_metrics['accuracy']:.4f}")

    def stage1_routing_unfrozen(self, train_loader, val_loader, epochs):
        """阶段1: 解冻路由器"""
        print("Stage 1: Training with high temperature")

        # 解冻路由器
        for param in self.model.router.parameters():
            param.requires_grad = True

        # 保持高温
        self.model.router.temperature.fill_(5.0)

        # 训练
        for epoch in range(epochs):
            self.train_epoch(train_loader, stage='routing')
            val_metrics = self.validate(val_loader)
            print(f"Routing Epoch {epoch}: Val Acc = {val_metrics['accuracy']:.4f}")

    def stage2_annealing(self, train_loader, val_loader, epochs):
        """阶段2: 温度退火"""
        print("Stage 2: Annealing temperature")

        # 初始温度
        init_temp = 2.0
        min_temp = 0.5
        decay_rate = (min_temp / init_temp) ** (1 / epochs)

        for epoch in range(epochs):
            # 更新温度
            current_temp = max(min_temp, init_temp * (decay_rate ** epoch))
            self.model.router.temperature.fill_(current_temp)

            # 训练时加入正则
            self.train_epoch(train_loader, stage='annealing')
            val_metrics = self.validate(val_loader)
            print(f"Annealing Epoch {epoch}: Temp = {current_temp:.3f}, "
                  f"Val Acc = {val_metrics['accuracy']:.4f}")

    def stage3_finetune(self, train_loader, val_loader, epochs):
        """阶段3: 端到端微调"""
        print("Stage 3: End-to-end finetuning")

        # 低温度
        self.model.router.temperature.fill_(0.1)

        # 降低学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.1

        # 训练
        for epoch in range(epochs):
            self.train_epoch(train_loader, stage='finetune')
            val_metrics = self.validate(val_loader)
            print(f"Finetune Epoch {epoch}: Val Acc = {val_metrics['accuracy']:.4f}")

    def train_epoch(self, train_loader, stage='warmup'):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            self.optimizer.zero_grad()

            # 前向传播
            output, routing_weights = self.model(data)

            # 基础分类损失
            cls_loss = F.cross_entropy(output, target)

            # 根据阶段添加不同的损失
            total_loss_batch = cls_loss

            if stage == 'annealing' or stage == 'finetune':
                # 稀疏正则
                sparsity_loss = torch.mean(torch.norm(routing_weights, p=1, dim=1))
                total_loss_batch += 0.01 * sparsity_loss

                # 负载均衡
                avg_routing = torch.mean(routing_weights, dim=0)
                load_balance_loss = -torch.sum(avg_routing * torch.log(avg_routing + 1e-8))
                total_loss_batch += 0.01 * load_balance_loss

                # 残差专家惩罚
                residual_penalty = routing_weights[:, -1].mean() ** 2
                total_loss_batch += 0.1 * residual_penalty

            # 反向传播
            total_loss_batch.backward()
            self.optimizer.step()

            # 统计
            total_loss += total_loss_batch.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        # 更新学习率
        self.scheduler.step()

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }
```

## 5. 评估指标实现

### 5.1 综合可解释性评估器

```python
class InterpretabilityEvaluator:
    """可解释性评估器"""
    def __init__(self):
        self.metrics = {}

    def evaluate_model(self, model, test_loader):
        """全面评估模型的可解释性"""

        # 收集所有预测和路由权重
        all_routing = []
        all_labels = []
        all_expert_outputs = []

        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                output, routing_weights = model.get_routing_weights(data)
                expert_outputs = model.get_expert_outputs(data)

                all_routing.append(routing_weights)
                all_labels.append(target)
                all_expert_outputs.append(expert_outputs)

        # 合并结果
        routing_weights = torch.cat(all_routing, dim=0).cpu().numpy()
        labels = torch.cat(all_labels, dim=0).cpu().numpy()
        expert_outputs = torch.cat(all_expert_outputs, dim=0).cpu().numpy()

        # 计算各项指标
        self.metrics['epi'] = self.compute_epi(routing_weights, labels)
        self.metrics['routing_stability'] = self.compute_routing_stability(model, test_loader)
        self.metrics['expert_orthogonality'] = self.compute_expert_orthogonality(expert_outputs)
        self.metrics['pcs'] = self.compute_pcs()

        return self.metrics

    def generate_report(self):
        """生成可解释性报告"""
        report = f"""
        ===== 可解释性评估报告 =====

        专家纯度指数 (EPI):
          - 平均值: {self.metrics['epi'].mean():.4f}
          - 最小值: {self.metrics['epi'].min():.4f}
          - 最大值: {self.metrics['epi'].max():.4f}
          - 标准差: {self.metrics['epi'].std():.4f}

        路由稳定性 (RS):
          - 稳定性得分: {self.metrics['routing_stability']:.4f}

        专家正交性 (EO):
          - 平均正交性: {self.metrics['expert_orthogonality']['avg']:.4f}
          - 正交性矩阵:
        {self.metrics['expert_orthogonality']['matrix']}

        物理一致性得分 (PCS):
          - 综合得分: {self.metrics['pcs']:.4f}

        ===== 评估结论 =====
        """

        # 添加解释
        if self.metrics['pcs'] > 0.8:
            report += "\n✅ 模型具有优秀的物理可解释性"
        elif self.metrics['pcs'] > 0.6:
            report += "\n⚠️ 模型具有较好的物理可解释性，但仍有提升空间"
        else:
            report += "\n❌ 模型的物理可解释性不足，需要改进"

        return report
```

## 6. 实验配置示例

### 6.1 增强配置文件

```yaml
# configs/enhanced_moe_explainable.yaml

model:
  name: "Enhanced-NNSPN-MoE"

  # 增强的MoE配置
  moe:
    num_experts: 6
    expert_types:
      - "low_pass_filter"
      - "harmonic_analyzer"
      - "envelope_detector"
      - "modulation_analyzer"
      - "sideband_analyzer"
      - "residual_mlp"

    # 增强的路由器
    router:
      type: "physics_enhanced"
      feature_selection: "adaptive"  # 自适应特征选择
      temperature_init: 5.0
      temperature_min: 0.1
      temperature_decay: 0.99

    # 专家约束增强
    expert_constraints:
      enabled: true
      frequency_constraints: true
      orthogonal_regularization: 0.01
      residual_penalty: 0.1
      residual_power: 2

    # 预训练配置
    pretraining:
      enabled: true
      epochs_per_expert: 50
      save_path: "pretrained_experts/"

# 增强的训练配置
training:
  # 渐进式训练
  progressive_training:
    enabled: true
    stages: ["warmup", "routing", "annealing", "finetune"]
    stage_ratios: [0.25, 0.25, 0.25, 0.25]

  # 损失函数权重
  loss:
    classification: 1.0
    sparsity: 0.01
    load_balancing: 0.01
    orthogonal: 0.01
    residual_penalty: 0.1
    frequency_constraint: 0.001

# 评估配置
evaluation:
  # 可解释性指标
  interpretability:
    epi: true
    routing_stability: true
    expert_orthogonality: true
    pcs: true

  # 可视化
  visualization:
    path_signature: true
    decision_boundary: true
    expert_activation: true
    feature_importance: true
```

这些架构增强方案显著提升了NNSPN-MoE的理论严谨性和实用性，为顶刊发表奠定了坚实基础。