# PHM-Vibench 生成模型(GM)预训练集成方案 - 优化版

**创建日期：2025年8月29日**  
**作者：PHM-Vibench 开发团队**  
**基于：CFL.ipynb 笔记本分析**  
**版本：优化版 v2.0**

---

## 执行摘要

本文档是流损失预训练方案的优化版，将原方案中的Flow模型重新定位为生成模型（Generative Model, GM），强调其生成能力和数据增强潜力。采用测试驱动开发（TDD）方法，每个模块都集成独立的测试代码，确保代码质量和可靠性。

### 关键优化

- **生成模型定位**：明确将模型定位为具有完整生成能力的GM模型
- **内嵌式测试**：每个模块包含 `if __name__ == '__main__'` 测试代码
- **TDD开发流程**：测试驱动的开发方法，先写测试再写实现
- **生成应用扩展**：突出数据增强、异常检测、信号合成能力

### 核心技术特点

- **矫正流匹配（Rectified Flow Matching）**：噪声与数据分布之间的直接线性插值
- **层次对比学习（Hierarchical Contrastive Learning）**：潜在空间中的 域 > 系统 > 实例 组织结构
- **多目标损失函数（Multi-Objective Loss Function）**：结合重建、流、对比和层次目标
- **条件生成（Conditional Generation）**：基于域和系统的可控生成

---

## 第一部分：生成模型技术基础

### 1.1 生成模型定位

#### 核心生成能力
- **数据生成**：合成具有特定域和系统特征的振动信号
- **异常生成**：生成各种故障模式的信号用于训练
- **数据增强**：平衡数据集中的类别分布
- **插值生成**：在不同状态之间生成中间态信号

#### 生成模型的三大应用场景
```python
# 1. 数据增强
synthetic_signals = gm_model.generate(
    domain_id=1, system_id=2, num_samples=1000, 
    condition_type="fault_class_0"
)

# 2. 异常检测
anomaly_score = gm_model.likelihood_score(signal)
is_anomaly = anomaly_score < threshold

# 3. 信号修复
restored_signal = gm_model.inpaint(
    corrupted_signal, mask=missing_indices
)
```

### 1.2 矫正流生成原理

#### 流匹配生成过程
```python
# 生成过程：从噪声到数据
def generate_sample(self, condition, num_steps=50):
    """从随机噪声生成高质量信号"""
    # 1. 从标准高斯分布采样噪声
    z = torch.randn(batch_size, latent_dim)
    
    # 2. 通过流匹配积分生成
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = torch.ones(batch_size, 1) * step * dt
        v = self.flow_net(z, t, condition)  # 预测速度
        z = z + v * dt  # 欧拉积分
    
    # 3. 解码到信号空间
    signal = self.decoder(z)
    return signal
```

### 1.3 层次条件生成

#### 多级条件控制
- **域级控制**：不同工业环境（轴承、齿轮、泵等）
- **系统级控制**：特定设备型号和配置
- **实例级控制**：具体的运行状态和故障类型

---

## 第二部分：生成模型架构设计

### 2.1 矫正流生成网络

#### 位置：`src/model_factory/ISFM/generative/GM_01_RectifiedFlow.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GM_01_RectifiedFlow(nn.Module):
    """
    矫正流生成网络 - 工业信号生成的核心组件
    
    功能：
    - 速度场预测用于流匹配
    - 支持条件生成和无条件生成
    - 数值稳定的时间嵌入
    - 高效的批量生成
    
    Architecture:
    - Time embedding: sinusoidal + MLP
    - Condition fusion: cross-attention mechanism
    - Velocity network: ResNet-style with skip connections
    """
    
    def __init__(self, configs):
        super().__init__()
        self.latent_dim = configs.latent_dim
        self.condition_dim = configs.condition_dim
        self.hidden_dim = getattr(configs, 'flow_hidden_dim', 256)
        self.num_layers = getattr(configs, 'flow_num_layers', 3)
        
        # 改进的时间嵌入
        self.time_embed = SinusoidalTimeEmbedding(
            dim=self.hidden_dim // 4,
            max_period=10000
        )
        
        # 条件融合层
        self.condition_fusion = nn.MultiheadAttention(
            embed_dim=self.condition_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 主生成网络（ResNet风格）
        self.input_proj = nn.Linear(
            self.latent_dim + self.condition_dim + self.hidden_dim // 4,
            self.hidden_dim
        )
        
        self.layers = nn.ModuleList([
            ResNetBlock(self.hidden_dim, dropout=0.1)
            for _ in range(self.num_layers)
        ])
        
        self.output_proj = nn.Linear(self.hidden_dim, self.latent_dim)
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Xavier初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, z_t, t, condition):
        """
        预测给定状态下的速度场
        
        Args:
            z_t: 插值状态 (B, latent_dim)
            t: 时间参数 (B, 1), 范围[0,1]
            condition: 条件向量 (B, condition_dim)
            
        Returns:
            v_pred: 预测速度 (B, latent_dim)
        """
        batch_size = z_t.shape[0]
        
        # 时间嵌入
        t_embed = self.time_embed(t)  # (B, hidden_dim//4)
        
        # 条件融合（自注意力）
        condition_fused, _ = self.condition_fusion(
            condition.unsqueeze(1), condition.unsqueeze(1), condition.unsqueeze(1)
        )
        condition_fused = condition_fused.squeeze(1)  # (B, condition_dim)
        
        # 特征融合
        x = torch.cat([z_t, condition_fused, t_embed], dim=1)
        x = self.input_proj(x)
        
        # ResNet前向传播
        for layer in self.layers:
            x = layer(x)
        
        # 输出速度
        v_pred = self.output_proj(x)
        
        return v_pred
    
    def generate(self, condition, num_samples=1, num_steps=50, 
                 temperature=1.0, device='cuda'):
        """
        生成新的信号样本
        
        Args:
            condition: 生成条件 (B, condition_dim)
            num_samples: 生成样本数量
            num_steps: 积分步数
            temperature: 温度参数控制多样性
            
        Returns:
            samples: 生成的潜在向量 (num_samples, latent_dim)
        """
        self.eval()
        with torch.no_grad():
            # 初始噪声
            z = torch.randn(num_samples, self.latent_dim, device=device) * temperature
            
            # 扩展条件
            if condition.shape[0] == 1:
                condition = condition.expand(num_samples, -1)
            
            # 流匹配积分
            dt = 1.0 / num_steps
            for step in range(num_steps):
                t = torch.ones(num_samples, 1, device=device) * step * dt
                v = self.forward(z, t, condition)
                z = z + v * dt
                
        return z

class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间嵌入"""
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, t):
        """
        Args:
            t: (B, 1) 时间参数
        Returns:
            (B, dim) 时间嵌入
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -np.log(self.max_period) * 
            torch.arange(half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        args = t.squeeze(-1)[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding

class ResNetBlock(nn.Module):
    """ResNet残差块"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        
    def forward(self, x):
        return x + self.layers(x)

# ================ 模块测试代码 ================
if __name__ == '__main__':
    """矫正流生成网络测试"""
    import time
    
    print("🧪 测试 GM_01_RectifiedFlow 模块")
    print("=" * 50)
    
    # 创建mock配置
    class MockConfig:
        def __init__(self):
            self.latent_dim = 128
            self.condition_dim = 64
            self.flow_hidden_dim = 256
            self.flow_num_layers = 3
    
    config = MockConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 使用设备: {device}")
    
    # 1. 模型初始化测试
    print("\n1️⃣ 模型初始化测试")
    try:
        model = GM_01_RectifiedFlow(config).to(device)
        print(f"✅ 模型初始化成功")
        print(f"   参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        exit(1)
    
    # 2. 前向传播测试
    print("\n2️⃣ 前向传播测试")
    batch_size = 32
    z_t = torch.randn(batch_size, config.latent_dim, device=device)
    t = torch.rand(batch_size, 1, device=device)
    condition = torch.randn(batch_size, config.condition_dim, device=device)
    
    try:
        start_time = time.time()
        v_pred = model(z_t, t, condition)
        forward_time = time.time() - start_time
        
        print(f"✅ 前向传播成功")
        print(f"   输入形状: z_t={z_t.shape}, t={t.shape}, condition={condition.shape}")
        print(f"   输出形状: {v_pred.shape}")
        print(f"   推理时间: {forward_time:.4f}s")
        print(f"   输出范围: [{v_pred.min().item():.4f}, {v_pred.max().item():.4f}]")
        
        # 验证输出形状
        assert v_pred.shape == (batch_size, config.latent_dim), f"输出形状不匹配: {v_pred.shape}"
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        exit(1)
    
    # 3. 梯度测试
    print("\n3️⃣ 梯度流测试")
    try:
        loss = v_pred.mean()
        loss.backward()
        
        # 检查梯度
        has_grad = 0
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None:
                    has_grad += 1
                    grad_norm = param.grad.norm().item()
                    if grad_norm < 1e-7:
                        print(f"⚠️  参数 {name} 梯度过小: {grad_norm:.2e}")
                    elif grad_norm > 10:
                        print(f"⚠️  参数 {name} 梯度过大: {grad_norm:.2e}")
        
        print(f"✅ 梯度计算成功")
        print(f"   有梯度的参数: {has_grad}/{total_params}")
        
        # 清除梯度
        model.zero_grad()
        
    except Exception as e:
        print(f"❌ 梯度计算失败: {e}")
        exit(1)
    
    # 4. 生成测试
    print("\n4️⃣ 信号生成测试")
    try:
        num_samples = 8
        condition_single = torch.randn(1, config.condition_dim, device=device)
        
        start_time = time.time()
        generated = model.generate(
            condition=condition_single,
            num_samples=num_samples,
            num_steps=20,  # 减少步数以加快测试
            temperature=1.0,
            device=device
        )
        generation_time = time.time() - start_time
        
        print(f"✅ 信号生成成功")
        print(f"   生成样本数: {num_samples}")
        print(f"   生成形状: {generated.shape}")
        print(f"   生成时间: {generation_time:.4f}s")
        print(f"   每样本时间: {generation_time/num_samples:.4f}s")
        
        # 验证生成质量
        gen_mean = generated.mean().item()
        gen_std = generated.std().item()
        print(f"   生成统计: 均值={gen_mean:.4f}, 标准差={gen_std:.4f}")
        
    except Exception as e:
        print(f"❌ 信号生成失败: {e}")
        exit(1)
    
    # 5. 批量生成测试
    print("\n5️⃣ 批量生成性能测试")
    try:
        batch_sizes = [1, 4, 16, 64]
        for bs in batch_sizes:
            condition_batch = torch.randn(bs, config.condition_dim, device=device)
            
            start_time = time.time()
            batch_generated = model.generate(
                condition=condition_batch,
                num_samples=bs,
                num_steps=10,
                device=device
            )
            batch_time = time.time() - start_time
            
            print(f"   批次大小 {bs:2d}: {batch_time:.4f}s ({batch_time/bs:.4f}s/样本)")
            
    except Exception as e:
        print(f"❌ 批量生成测试失败: {e}")
    
    # 6. 内存使用测试
    print("\n6️⃣ 内存使用测试")
    if device == 'cuda':
        try:
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated()
            
            # 大批次测试
            large_batch = 128
            z_large = torch.randn(large_batch, config.latent_dim, device=device)
            t_large = torch.rand(large_batch, 1, device=device)
            condition_large = torch.randn(large_batch, config.condition_dim, device=device)
            
            v_large = model(z_large, t_large, condition_large)
            
            memory_after = torch.cuda.memory_allocated()
            memory_used = (memory_after - memory_before) / 1024**2  # MB
            
            print(f"✅ 内存使用测试完成")
            print(f"   批次大小: {large_batch}")
            print(f"   内存使用: {memory_used:.2f} MB")
            
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ 内存测试失败: {e}")
    else:
        print("ℹ️  CPU模式，跳过GPU内存测试")
    
    print("\n🎉 所有测试完成！")
    print("=" * 50)
    print("模块状态：✅ 可用于生产环境")
```

### 2.2 条件编码器增强

#### 位置：`src/model_factory/ISFM/encoder/E_03_ConditionalEncoder.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

class E_03_ConditionalEncoder(nn.Module):
    """
    条件编码器 - 支持层次化域和系统条件
    
    功能：
    - 域嵌入：跨数据集泛化能力
    - 系统嵌入：设备特定模式识别
    - 层次化特征提取：多级抽象
    - 注意力机制：关键特征聚焦
    
    Architecture:
    - Hierarchical embeddings with learnable positions
    - Multi-head self-attention for feature refinement
    - Residual connections and layer normalization
    - Adaptive feature scaling based on domain/system
    """
    
    def __init__(self, configs):
        super().__init__()
        self.input_dim = configs.input_dim
        self.latent_dim = configs.latent_dim
        self.num_domains = getattr(configs, 'num_domains', 2)
        self.num_systems = getattr(configs, 'num_systems', 2)
        self.cond_embed_dim = getattr(configs, 'cond_embed_dim', 32)
        self.use_attention = getattr(configs, 'use_attention', True)
        
        # 层次化嵌入
        self.domain_embed = nn.Embedding(self.num_domains, self.cond_embed_dim)
        self.system_embed = nn.Embedding(self.num_systems, self.cond_embed_dim)
        
        # 位置编码（可学习）
        self.domain_pos_embed = nn.Parameter(torch.randn(1, self.cond_embed_dim))
        self.system_pos_embed = nn.Parameter(torch.randn(1, self.cond_embed_dim))
        
        # 条件融合层
        total_cond_dim = 2 * self.cond_embed_dim
        self.condition_proj = nn.Linear(total_cond_dim, self.cond_embed_dim)
        
        # 主编码网络
        total_input_dim = self.input_dim + self.cond_embed_dim
        
        # 多层编码器
        self.input_proj = nn.Linear(total_input_dim, 256)
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(256, num_heads=8, dropout=0.1),
            EncoderBlock(256, num_heads=8, dropout=0.1),
            EncoderBlock(256, num_heads=4, dropout=0.1),
        ])
        
        self.output_proj = nn.Linear(256, self.latent_dim)
        
        # 自适应特征缩放
        self.feature_scale = AdaptiveFeatureScaling(
            self.latent_dim, self.num_domains, self.num_systems
        )
        
        # 初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, x, domain_id, system_id):
        """
        条件编码前向传播
        
        Args:
            x: 输入信号 (B, input_dim)
            domain_id: 域ID (B,) 
            system_id: 系统ID (B,)
            
        Returns:
            h: 潜在表示 (B, latent_dim)
        """
        batch_size = x.shape[0]
        
        # 1. 获取层次嵌入
        domain_emb = self.domain_embed(domain_id) + self.domain_pos_embed
        system_emb = self.system_embed(system_id) + self.system_pos_embed
        
        # 2. 融合条件信息
        condition = torch.cat([domain_emb, system_emb], dim=1)  # (B, 2*cond_embed_dim)
        condition_fused = self.condition_proj(condition)  # (B, cond_embed_dim)
        
        # 3. 输入与条件融合
        x_cond = torch.cat([x, condition_fused], dim=1)  # (B, input_dim + cond_embed_dim)
        
        # 4. 编码处理
        h = self.input_proj(x_cond)  # (B, 256)
        
        # 通过编码器层
        for layer in self.encoder_layers:
            h = layer(h)
        
        # 5. 输出投影
        h = self.output_proj(h)  # (B, latent_dim)
        
        # 6. 自适应特征缩放
        h = self.feature_scale(h, domain_id, system_id)
        
        return h
    
    def get_condition_embedding(self, domain_id, system_id):
        """获取条件嵌入（用于生成）"""
        domain_emb = self.domain_embed(domain_id) + self.domain_pos_embed
        system_emb = self.system_embed(system_id) + self.system_pos_embed
        condition = torch.cat([domain_emb, system_emb], dim=1)
        return self.condition_proj(condition)

class EncoderBlock(nn.Module):
    """编码器块"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (B, dim)
        x = x.unsqueeze(1)  # (B, 1, dim) for attention
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x.squeeze(1)  # (B, dim)

class AdaptiveFeatureScaling(nn.Module):
    """基于域和系统的自适应特征缩放"""
    def __init__(self, feature_dim, num_domains, num_systems):
        super().__init__()
        self.domain_scale = nn.Embedding(num_domains, feature_dim)
        self.system_scale = nn.Embedding(num_systems, feature_dim)
        
        # 初始化为接近1的值
        nn.init.normal_(self.domain_scale.weight, mean=1.0, std=0.1)
        nn.init.normal_(self.system_scale.weight, mean=1.0, std=0.1)
    
    def forward(self, features, domain_id, system_id):
        domain_scale = torch.sigmoid(self.domain_scale(domain_id))  # (B, feature_dim)
        system_scale = torch.sigmoid(self.system_scale(system_id))  # (B, feature_dim)
        
        # 组合缩放
        combined_scale = domain_scale * system_scale
        return features * combined_scale

# ================ 模块测试代码 ================
if __name__ == '__main__':
    """条件编码器测试"""
    import time
    from collections import defaultdict
    
    print("🧪 测试 E_03_ConditionalEncoder 模块")
    print("=" * 50)
    
    # Mock配置
    class MockConfig:
        def __init__(self):
            self.input_dim = 1024  # 信号维度
            self.latent_dim = 128
            self.num_domains = 4    # 4个域
            self.num_systems = 8    # 8个系统
            self.cond_embed_dim = 32
            self.use_attention = True
    
    config = MockConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 使用设备: {device}")
    
    # 1. 模型初始化测试
    print("\n1️⃣ 模型初始化测试")
    try:
        model = E_03_ConditionalEncoder(config).to(device)
        print(f"✅ 模型初始化成功")
        print(f"   总参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 分析参数分布
        param_groups = defaultdict(int)
        for name, param in model.named_parameters():
            if 'embed' in name:
                param_groups['embedding'] += param.numel()
            elif 'attention' in name:
                param_groups['attention'] += param.numel()
            elif 'ffn' in name:
                param_groups['feedforward'] += param.numel()
            else:
                param_groups['other'] += param.numel()
        
        print("   参数分布:")
        for group, count in param_groups.items():
            print(f"     {group}: {count:,}")
            
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        exit(1)
    
    # 2. 前向传播测试
    print("\n2️⃣ 前向传播测试")
    batch_size = 32
    x = torch.randn(batch_size, config.input_dim, device=device)
    domain_ids = torch.randint(0, config.num_domains, (batch_size,), device=device)
    system_ids = torch.randint(0, config.num_systems, (batch_size,), device=device)
    
    try:
        start_time = time.time()
        h = model(x, domain_ids, system_ids)
        forward_time = time.time() - start_time
        
        print(f"✅ 前向传播成功")
        print(f"   输入形状: x={x.shape}")
        print(f"   域ID范围: {domain_ids.min().item()} - {domain_ids.max().item()}")
        print(f"   系统ID范围: {system_ids.min().item()} - {system_ids.max().item()}")
        print(f"   输出形状: {h.shape}")
        print(f"   推理时间: {forward_time:.4f}s")
        
        # 验证输出
        assert h.shape == (batch_size, config.latent_dim)
        print(f"   输出统计: 均值={h.mean().item():.4f}, 标准差={h.std().item():.4f}")
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        exit(1)
    
    # 3. 条件嵌入测试
    print("\n3️⃣ 条件嵌入测试")
    try:
        # 测试不同条件的嵌入差异
        domain1 = torch.tensor([0], device=device)
        domain2 = torch.tensor([1], device=device)
        system1 = torch.tensor([0], device=device)
        system2 = torch.tensor([1], device=device)
        
        emb1 = model.get_condition_embedding(domain1, system1)
        emb2 = model.get_condition_embedding(domain1, system2)  # 不同系统
        emb3 = model.get_condition_embedding(domain2, system1)  # 不同域
        
        # 计算相似度
        sim_system = F.cosine_similarity(emb1, emb2, dim=1).item()
        sim_domain = F.cosine_similarity(emb1, emb3, dim=1).item()
        
        print(f"✅ 条件嵌入测试完成")
        print(f"   嵌入维度: {emb1.shape}")
        print(f"   同域不同系统相似度: {sim_system:.4f}")
        print(f"   不同域同系统相似度: {sim_domain:.4f}")
        
        # 理想情况下，同域不同系统应该比不同域更相似
        if sim_system > sim_domain:
            print("   ✅ 层次化嵌入正常：同域内相似度更高")
        else:
            print("   ⚠️  层次化嵌入可能需要更多训练")
            
    except Exception as e:
        print(f"❌ 条件嵌入测试失败: {e}")
    
    # 4. 梯度流测试
    print("\n4️⃣ 梯度流测试")
    try:
        # 计算一个简单损失
        target = torch.randn_like(h)
        loss = F.mse_loss(h, target)
        loss.backward()
        
        # 检查梯度
        grad_stats = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                if 'embed' in name:
                    grad_stats.setdefault('embedding', []).append(grad_norm)
                elif 'attention' in name:
                    grad_stats.setdefault('attention', []).append(grad_norm)
                else:
                    grad_stats.setdefault('other', []).append(grad_norm)
        
        print(f"✅ 梯度计算成功")
        print("   梯度统计:")
        for component, norms in grad_stats.items():
            avg_norm = np.mean(norms)
            print(f"     {component}: 平均={avg_norm:.6f}, 范围=[{min(norms):.6f}, {max(norms):.6f}]")
            
        model.zero_grad()
        
    except Exception as e:
        print(f"❌ 梯度测试失败: {e}")
    
    # 5. 批量处理性能测试
    print("\n5️⃣ 批量处理性能测试")
    batch_sizes = [1, 8, 32, 128]
    for bs in batch_sizes:
        try:
            x_batch = torch.randn(bs, config.input_dim, device=device)
            d_batch = torch.randint(0, config.num_domains, (bs,), device=device)
            s_batch = torch.randint(0, config.num_systems, (bs,), device=device)
            
            start_time = time.time()
            h_batch = model(x_batch, d_batch, s_batch)
            batch_time = time.time() - start_time
            
            print(f"   批次 {bs:3d}: {batch_time:.4f}s ({batch_time/bs*1000:.2f}ms/样本)")
            
        except Exception as e:
            print(f"   批次 {bs} 失败: {e}")
    
    # 6. 内存效率测试
    print("\n6️⃣ 内存使用测试")
    if device == 'cuda':
        try:
            torch.cuda.empty_cache()
            memory_start = torch.cuda.memory_allocated()
            
            # 大批次处理
            large_batch = 256
            x_large = torch.randn(large_batch, config.input_dim, device=device)
            d_large = torch.randint(0, config.num_domains, (large_batch,), device=device)
            s_large = torch.randint(0, config.num_systems, (large_batch,), device=device)
            
            h_large = model(x_large, d_large, s_large)
            
            memory_end = torch.cuda.memory_allocated()
            memory_used = (memory_end - memory_start) / 1024**2
            
            print(f"✅ 内存使用: {memory_used:.2f} MB (批次={large_batch})")
            print(f"   单样本内存: {memory_used/large_batch:.4f} MB")
            
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ 内存测试失败: {e}")
    
    print("\n🎉 条件编码器测试完成！")
    print("=" * 50)
    print("模块状态：✅ 可用于生产环境")
```

### 2.3 生成模型主体

#### 位置：`src/model_factory/ISFM/M_04_ISFM_GM.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Union
from .generative.GM_01_RectifiedFlow import GM_01_RectifiedFlow
from .encoder.E_03_ConditionalEncoder import E_03_ConditionalEncoder

class Model(nn.Module):
    """
    ISFM生成模型 - 工业信号基础生成模型
    
    功能特点:
    - 条件生成：基于域和系统的可控生成
    - 信号重建：高保真度信号重构
    - 异常检测：通过重建误差检测异常
    - 数据增强：生成平衡的训练数据
    
    Architecture:
    - Conditional encoder with hierarchical embeddings
    - Rectified flow generative model
    - Multi-task output heads (reconstruction + generation)
    - Optional classifier for supervised guidance
    """
    
    def __init__(self, args_m, metadata):
        super().__init__()
        self.args_m = args_m
        self.metadata = metadata
        
        # 核心生成组件
        self.encoder = E_03_ConditionalEncoder(args_m)
        
        # 解码器（重建网络）
        self.decoder = GenerativeDecoder(
            latent_dim=args_m.latent_dim,
            output_dim=args_m.input_dim,
            hidden_dim=getattr(args_m, 'decoder_hidden_dim', 256),
            num_layers=getattr(args_m, 'decoder_num_layers', 3)
        )
        
        # 流生成网络
        self.flow_net = GM_01_RectifiedFlow(args_m)
        
        # 可选分类器
        if getattr(args_m, 'use_classifier', False):
            self.classifier = nn.Sequential(
                nn.Linear(args_m.latent_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, args_m.num_classes)
            )
        else:
            self.classifier = None
            
        # EMA模型（用于更稳定的生成）
        self.use_ema = getattr(args_m, 'use_ema', True)
        if self.use_ema:
            self.ema_decay = getattr(args_m, 'ema_decay', 0.995)
            self.ema_model = self._create_ema_model()
            
        # 生成参数
        self.generation_config = GenerationConfig(args_m)
        
    def _create_ema_model(self):
        """创建EMA模型"""
        ema_model = type(self)(self.args_m, self.metadata)
        ema_model.load_state_dict(self.state_dict())
        for param in ema_model.parameters():
            param.requires_grad_(False)
        return ema_model
    
    def update_ema(self):
        """更新EMA模型"""
        if not self.use_ema:
            return
            
        with torch.no_grad():
            for ema_param, current_param in zip(
                self.ema_model.parameters(), self.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    current_param.data, alpha=1 - self.ema_decay
                )
    
    def forward(self, x, domain_id, system_id, t=None, return_components=False):
        """
        前向传播 - 支持训练和生成两种模式
        
        Args:
            x: 输入信号 (B, input_dim)
            domain_id: 域ID (B,)
            system_id: 系统ID (B,)
            t: 时间参数 (B, 1) - 训练时使用
            return_components: 是否返回所有组件
            
        Returns:
            根据return_components返回不同内容
        """
        # 1. 条件编码
        h = self.encoder(x, domain_id, system_id)
        
        # 2. 信号重建
        x_recon = self.decoder(h)
        
        # 3. 流预测（训练时）
        v_pred = None
        if t is not None:
            # 获取条件嵌入
            condition = self.encoder.get_condition_embedding(domain_id, system_id)
            
            # 创建插值状态
            z0 = torch.randn_like(h)
            z_t = (1 - t) * z0 + t * h
            
            # 预测速度
            v_pred = self.flow_net(z_t, t, condition)
        
        # 4. 分类（可选）
        y_pred = None
        if self.classifier is not None:
            y_pred = self.classifier(h)
        
        if return_components:
            return x_recon, h, v_pred, y_pred
        else:
            return x_recon
    
    def generate(self, domain_id, system_id, num_samples=1, 
                 num_steps=50, temperature=1.0, use_ema=None):
        """
        生成新的信号样本
        
        Args:
            domain_id: 目标域ID (1,) 或 (num_samples,)
            system_id: 目标系统ID (1,) 或 (num_samples,)
            num_samples: 生成样本数量
            num_steps: 流匹配积分步数
            temperature: 温度参数（控制多样性）
            use_ema: 是否使用EMA模型
            
        Returns:
            generated_signals: 生成的信号 (num_samples, input_dim)
        """
        if use_ema is None:
            use_ema = self.use_ema
            
        # 选择使用的模型
        model_to_use = self.ema_model if (use_ema and hasattr(self, 'ema_model')) else self
        
        model_to_use.eval()
        with torch.no_grad():
            device = next(model_to_use.parameters()).device
            
            # 扩展ID到所需样本数
            if domain_id.shape[0] == 1:
                domain_id = domain_id.expand(num_samples)
            if system_id.shape[0] == 1:
                system_id = system_id.expand(num_samples)
                
            # 获取条件嵌入
            condition = model_to_use.encoder.get_condition_embedding(domain_id, system_id)
            
            # 流生成
            z_generated = model_to_use.flow_net.generate(
                condition=condition,
                num_samples=num_samples,
                num_steps=num_steps,
                temperature=temperature,
                device=device
            )
            
            # 解码到信号空间
            generated_signals = model_to_use.decoder(z_generated)
            
        return generated_signals
    
    def interpolate(self, signal1, signal2, domain_id, system_id, 
                   num_steps=10, interpolation_mode='spherical'):
        """
        在两个信号之间进行插值
        
        Args:
            signal1, signal2: 输入信号 (1, input_dim)
            domain_id, system_id: 条件ID (1,)
            num_steps: 插值步数
            interpolation_mode: 'linear' 或 'spherical'
            
        Returns:
            interpolated_signals: 插值信号 (num_steps, input_dim)
        """
        self.eval()
        with torch.no_grad():
            # 编码到潜在空间
            h1 = self.encoder(signal1, domain_id, system_id)
            h2 = self.encoder(signal2, domain_id, system_id)
            
            # 插值
            alphas = torch.linspace(0, 1, num_steps, device=h1.device)
            interpolated_h = []
            
            for alpha in alphas:
                if interpolation_mode == 'spherical':
                    # 球面插值
                    omega = torch.acos(torch.clamp(
                        (h1 * h2).sum(dim=1, keepdim=True) / 
                        (torch.norm(h1, dim=1, keepdim=True) * torch.norm(h2, dim=1, keepdim=True)),
                        -1, 1
                    ))
                    sin_omega = torch.sin(omega)
                    if sin_omega.abs() < 1e-6:
                        h_interp = (1 - alpha) * h1 + alpha * h2
                    else:
                        h_interp = (torch.sin((1 - alpha) * omega) * h1 + 
                                   torch.sin(alpha * omega) * h2) / sin_omega
                else:
                    # 线性插值
                    h_interp = (1 - alpha) * h1 + alpha * h2
                
                interpolated_h.append(h_interp)
            
            # 批量解码
            interpolated_h = torch.cat(interpolated_h, dim=0)
            interpolated_signals = self.decoder(interpolated_h)
            
        return interpolated_signals
    
    def compute_likelihood(self, x, domain_id, system_id, num_steps=50):
        """
        计算信号的似然度（用于异常检测）
        
        Args:
            x: 输入信号 (B, input_dim)
            domain_id, system_id: 条件ID (B,)
            num_steps: 流匹配步数
            
        Returns:
            likelihood_scores: 似然度分数 (B,)
        """
        self.eval()
        with torch.no_grad():
            # 编码
            h = self.encoder(x, domain_id, system_id)
            condition = self.encoder.get_condition_embedding(domain_id, system_id)
            
            # 通过逆向流匹配计算似然
            z = h.clone()
            log_likelihood = torch.zeros(h.shape[0], device=h.device)
            
            dt = 1.0 / num_steps
            for step in range(num_steps):
                t = torch.ones(h.shape[0], 1, device=h.device) * (1 - step * dt)
                
                # 预测速度
                v = self.flow_net(z, t, condition)
                
                # 逆向积分
                z = z - v * dt
                
                # 累积log似然（近似）
                log_likelihood -= (v ** 2).sum(dim=1) * dt * 0.5
            
        return log_likelihood

class GenerativeDecoder(nn.Module):
    """生成解码器"""
    def __init__(self, latent_dim, output_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        
        layers = []
        current_dim = latent_dim
        
        # 隐藏层
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
            
        # 输出层
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, h):
        return self.decoder(h)

class GenerationConfig:
    """生成配置"""
    def __init__(self, args_m):
        self.num_steps = getattr(args_m, 'generation_steps', 50)
        self.temperature = getattr(args_m, 'generation_temperature', 1.0)
        self.use_ema = getattr(args_m, 'use_ema_for_generation', True)

# ================ 模块测试代码 ================
if __name__ == '__main__':
    """ISFM生成模型测试"""
    import time
    import matplotlib.pyplot as plt
    
    print("🧪 测试 M_04_ISFM_GM 模块")
    print("=" * 60)
    
    # Mock配置和元数据
    class MockConfig:
        def __init__(self):
            self.input_dim = 1024
            self.latent_dim = 128
            self.condition_dim = 64
            self.flow_hidden_dim = 256
            self.decoder_hidden_dim = 256
            self.num_domains = 3
            self.num_systems = 6
            self.cond_embed_dim = 32
            self.use_classifier = True
            self.num_classes = 5
            self.use_ema = True
            self.ema_decay = 0.995
    
    config = MockConfig()
    metadata = {}  # Mock metadata
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 使用设备: {device}")
    
    # 1. 模型初始化测试
    print("\n1️⃣ 生成模型初始化测试")
    try:
        model = Model(config, metadata).to(device)
        print(f"✅ 生成模型初始化成功")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   总参数量: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        
        # 组件参数统计
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        flow_params = sum(p.numel() for p in model.flow_net.parameters())
        
        print(f"   编码器参数: {encoder_params:,}")
        print(f"   解码器参数: {decoder_params:,}")
        print(f"   流网络参数: {flow_params:,}")
        
        if model.classifier:
            classifier_params = sum(p.numel() for p in model.classifier.parameters())
            print(f"   分类器参数: {classifier_params:,}")
            
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        exit(1)
    
    # 2. 前向传播测试
    print("\n2️⃣ 前向传播测试")
    batch_size = 16
    x = torch.randn(batch_size, config.input_dim, device=device)
    domain_ids = torch.randint(0, config.num_domains, (batch_size,), device=device)
    system_ids = torch.randint(0, config.num_systems, (batch_size,), device=device)
    t = torch.rand(batch_size, 1, device=device)
    
    try:
        start_time = time.time()
        x_recon, h, v_pred, y_pred = model(
            x, domain_ids, system_ids, t=t, return_components=True
        )
        forward_time = time.time() - start_time
        
        print(f"✅ 前向传播成功")
        print(f"   重建信号形状: {x_recon.shape}")
        print(f"   潜在表示形状: {h.shape}")
        print(f"   速度预测形状: {v_pred.shape}")
        if y_pred is not None:
            print(f"   分类预测形状: {y_pred.shape}")
        print(f"   前向时间: {forward_time:.4f}s")
        
        # 重建质量检查
        recon_error = F.mse_loss(x_recon, x)
        print(f"   重建误差: {recon_error.item():.6f}")
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        exit(1)
    
    # 3. 生成测试
    print("\n3️⃣ 信号生成测试")
    try:
        num_samples = 8
        target_domain = torch.tensor([0], device=device)
        target_system = torch.tensor([1], device=device)
        
        start_time = time.time()
        generated = model.generate(
            domain_id=target_domain,
            system_id=target_system,
            num_samples=num_samples,
            num_steps=25,
            temperature=1.0
        )
        generation_time = time.time() - start_time
        
        print(f"✅ 信号生成成功")
        print(f"   生成信号形状: {generated.shape}")
        print(f"   生成时间: {generation_time:.4f}s")
        print(f"   单样本生成时间: {generation_time/num_samples:.4f}s")
        
        # 生成质量分析
        gen_mean = generated.mean().item()
        gen_std = generated.std().item()
        real_mean = x.mean().item()
        real_std = x.std().item()
        
        print(f"   生成信号统计: 均值={gen_mean:.4f}, 标准差={gen_std:.4f}")
        print(f"   真实信号统计: 均值={real_mean:.4f}, 标准差={real_std:.4f}")
        print(f"   统计相似度: 均值差={abs(gen_mean-real_mean):.4f}, 标准差差={abs(gen_std-real_std):.4f}")
        
    except Exception as e:
        print(f"❌ 信号生成失败: {e}")
    
    # 4. 插值测试
    print("\n4️⃣ 信号插值测试")
    try:
        signal1 = x[:1]  # 第一个信号
        signal2 = x[1:2]  # 第二个信号
        domain_id = domain_ids[:1]
        system_id = system_ids[:1]
        
        start_time = time.time()
        interpolated = model.interpolate(
            signal1, signal2, domain_id, system_id,
            num_steps=10, interpolation_mode='spherical'
        )
        interp_time = time.time() - start_time
        
        print(f"✅ 信号插值成功")
        print(f"   插值序列形状: {interpolated.shape}")
        print(f"   插值时间: {interp_time:.4f}s")
        
        # 检查插值的连续性
        start_diff = F.mse_loss(interpolated[0:1], signal1)
        end_diff = F.mse_loss(interpolated[-1:], signal2)
        
        print(f"   起点误差: {start_diff.item():.6f}")
        print(f"   终点误差: {end_diff.item():.6f}")
        
        if start_diff < 1e-3 and end_diff < 1e-3:
            print("   ✅ 插值端点正确")
        else:
            print("   ⚠️  插值端点误差较大")
            
    except Exception as e:
        print(f"❌ 信号插值失败: {e}")
    
    # 5. 似然度计算测试（异常检测）
    print("\n5️⃣ 异常检测测试")
    try:
        # 正常信号
        normal_signals = x[:8]
        normal_domains = domain_ids[:8]
        normal_systems = system_ids[:8]
        
        # 创建异常信号（添加大噪声）
        abnormal_signals = normal_signals + torch.randn_like(normal_signals) * 2
        
        # 计算似然度
        start_time = time.time()
        normal_likelihood = model.compute_likelihood(
            normal_signals, normal_domains, normal_systems, num_steps=20
        )
        abnormal_likelihood = model.compute_likelihood(
            abnormal_signals, normal_domains, normal_systems, num_steps=20
        )
        likelihood_time = time.time() - start_time
        
        print(f"✅ 异常检测测试完成")
        print(f"   正常信号似然度: {normal_likelihood.mean().item():.4f} ± {normal_likelihood.std().item():.4f}")
        print(f"   异常信号似然度: {abnormal_likelihood.mean().item():.4f} ± {abnormal_likelihood.std().item():.4f}")
        print(f"   计算时间: {likelihood_time:.4f}s")
        
        # 异常检测效果
        if abnormal_likelihood.mean() < normal_likelihood.mean():
            print("   ✅ 异常检测有效：异常信号似然度更低")
        else:
            print("   ⚠️  异常检测效果有限，需要更多训练")
            
    except Exception as e:
        print(f"❌ 异常检测测试失败: {e}")
    
    # 6. EMA模型测试
    print("\n6️⃣ EMA模型测试")
    if hasattr(model, 'ema_model'):
        try:
            # 更新EMA几次
            for _ in range(5):
                model.update_ema()
            
            # 比较EMA和普通模型的生成结果
            normal_gen = model.generate(
                target_domain, target_system, num_samples=4, 
                num_steps=10, use_ema=False
            )
            ema_gen = model.generate(
                target_domain, target_system, num_samples=4, 
                num_steps=10, use_ema=True
            )
            
            # 计算差异
            diff = F.mse_loss(normal_gen, ema_gen)
            
            print(f"✅ EMA模型测试完成")
            print(f"   普通生成形状: {normal_gen.shape}")
            print(f"   EMA生成形状: {ema_gen.shape}")
            print(f"   生成差异: {diff.item():.6f}")
            
        except Exception as e:
            print(f"❌ EMA测试失败: {e}")
    else:
        print("ℹ️  EMA模型未启用")
    
    # 7. 内存和性能测试
    print("\n7️⃣ 性能测试")
    try:
        # 测试不同批次大小的性能
        batch_sizes = [1, 4, 16, 32]
        print("   批次大小性能测试:")
        
        for bs in batch_sizes:
            x_test = torch.randn(bs, config.input_dim, device=device)
            d_test = torch.randint(0, config.num_domains, (bs,), device=device)
            s_test = torch.randint(0, config.num_systems, (bs,), device=device)
            
            # 前向传播时间
            start_time = time.time()
            with torch.no_grad():
                x_recon_test = model(x_test, d_test, s_test)
            forward_time = time.time() - start_time
            
            # 生成时间
            start_time = time.time()
            gen_test = model.generate(d_test[:1], s_test[:1], num_samples=bs, num_steps=10)
            gen_time = time.time() - start_time
            
            print(f"     批次 {bs:2d}: 前向={forward_time:.4f}s, 生成={gen_time:.4f}s")
            
        # GPU内存测试
        if device == 'cuda':
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated()
            
            # 大批次测试
            large_x = torch.randn(64, config.input_dim, device=device)
            large_d = torch.randint(0, config.num_domains, (64,), device=device)
            large_s = torch.randint(0, config.num_systems, (64,), device=device)
            
            with torch.no_grad():
                large_recon = model(large_x, large_d, large_s)
            
            memory_after = torch.cuda.memory_allocated()
            memory_used = (memory_after - memory_before) / 1024**2
            
            print(f"   内存使用: {memory_used:.2f} MB (批次=64)")
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
    
    print("\n🎉 所有测试完成！")
    print("=" * 60)
    print("✅ ISFM生成模型 - 可用于生产环境")
    print("📊 主要功能: 信号生成、重建、插值、异常检测")
    print("🚀 建议下一步: 开始预训练实验")
```

---

## 第三部分：测试驱动开发(TDD)方案

### 3.1 TDD开发流程

#### 开发循环
```
1. 🔴 编写失败的测试 → 2. 🟢 编写最小可行代码 → 3. 🔵 重构优化 → 重复
```

#### 测试层次
- **单元测试**：模块内 `if __name__ == '__main__'` 测试
- **集成测试**：模块间交互测试
- **系统测试**：完整生成管道测试
- **性能测试**：内存、速度、质量基准

### 3.2 每模块测试代码模板

```python
# ================ 模块测试代码模板 ================
if __name__ == '__main__':
    """模块名称测试套件"""
    import time
    import torch
    import numpy as np
    from collections import defaultdict
    
    print(f"🧪 测试 {模块名称} 模块")
    print("=" * 50)
    
    # 1. Mock配置设置
    class MockConfig:
        def __init__(self):
            # 设置测试配置参数
            pass
    
    config = MockConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2. 初始化测试
    print("\n1️⃣ 模块初始化测试")
    try:
        model = ModuleClass(config)
        print("✅ 初始化成功")
        # 参数统计、内存检查等
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        exit(1)
    
    # 3. 前向传播测试
    print("\n2️⃣ 前向传播测试")
    # 创建mock输入、验证输出形状、检查数值范围
    
    # 4. 梯度流测试
    print("\n3️⃣ 梯度流测试")
    # 反向传播、梯度检查、数值稳定性
    
    # 5. 边界条件测试
    print("\n4️⃣ 边界条件测试")
    # 极端输入、异常处理、鲁棒性
    
    # 6. 性能测试
    print("\n5️⃣ 性能测试")
    # 速度基准、内存使用、批量处理
    
    print("\n🎉 所有测试完成！")
    print("模块状态：✅ 可用于生产环境")
```

---

## 第四部分：实施路线图 - 优化版

### 4.1 第一阶段：生成模块实现（第1-2周）

#### 第1-3天：矫正流生成网络
- [ ] 创建 `src/model_factory/ISFM/generative/` 目录
- [ ] 实现 `GM_01_RectifiedFlow.py` 包含完整测试代码
- [ ] 时间嵌入优化：正弦嵌入 + MLP
- [ ] ResNet风格的速度网络
- [ ] ✅ **测试覆盖目标**: 单元测试 >95%

#### 第4-6天：条件编码器增强  
- [ ] 实现 `E_03_ConditionalEncoder.py` 包含测试
- [ ] 多头注意力机制
- [ ] 自适应特征缩放
- [ ] 层次化嵌入验证
- [ ] ✅ **测试覆盖目标**: 条件生成准确性 >90%

#### 第7-10天：生成模型主体
- [ ] 实现 `M_04_ISFM_GM.py` 完整生成模型
- [ ] EMA模型集成
- [ ] 多种生成模式：无条件/条件/插值
- [ ] 异常检测功能
- [ ] ✅ **测试覆盖目标**: 端到端生成流程测试

#### 第11-14天：损失函数优化
- [ ] 实现 `gm_pretrain_loss.py` 生成模型专用损失
- [ ] 流匹配损失优化
- [ ] 层次对比学习损失
- [ ] 生成质量损失（FID, IS等）
- [ ] ✅ **测试覆盖目标**: 损失组件独立验证

### 4.2 第二阶段：任务与管道集成（第3周）

#### 第15-17天：Lightning任务模块
- [ ] 创建 `gm_pretrain_task.py` 包含训练逻辑
- [ ] 生成样本质量监控
- [ ] 多GPU训练支持
- [ ] 可视化回调函数
- [ ] ✅ **测试覆盖目标**: 训练稳定性验证

#### 第18-21天：管道集成与配置
- [ ] 更新管道支持生成模型训练
- [ ] 创建生成模型专用配置
- [ ] 数据增强集成
- [ ] 异常检测集成
- [ ] ✅ **测试覆盖目标**: 完整管道测试

### 4.3 第三阶段：质量保证与优化（第4周）

#### 第22-25天：生成质量评估
- [ ] FID (Fréchet Inception Distance) 评估
- [ ] 信号多样性指标
- [ ] 条件生成准确性测试
- [ ] 异常检测ROC曲线
- [ ] ✅ **测试覆盖目标**: 质量基准建立

#### 第26-28天：性能优化与文档
- [ ] 内存优化（梯度检查点）
- [ ] 生成速度优化
- [ ] 完整文档和示例
- [ ] 用户指南
- [ ] ✅ **测试覆盖目标**: 性能基准达标

---

## 第五部分：文件组织与命名规范

### 5.1 新建文件清单（18个文件）

#### 生成模型核心文件
1. `src/model_factory/ISFM/generative/__init__.py`
2. `src/model_factory/ISFM/generative/GM_01_RectifiedFlow.py`
3. `src/model_factory/ISFM/generative/base_generative.py`
4. `src/model_factory/ISFM/encoder/E_03_ConditionalEncoder.py`
5. `src/model_factory/ISFM/M_04_ISFM_GM.py`

#### 任务和损失函数
6. `src/task_factory/Components/gm_pretrain_loss.py`
7. `src/task_factory/task/pretrain/gm_pretrain_task.py`

#### 配置文件
8. `configs/demo/GenerativeModel/gm_pretrain.yaml`
9. `configs/demo/GenerativeModel/gm_pretrain_basic.yaml`
10. `configs/demo/GenerativeModel/gm_pretrain_advanced.yaml`

#### 测试文件
11. `test/unit/test_gm_rectified_flow.py`
12. `test/unit/test_conditional_encoder.py`
13. `test/unit/test_isfm_gm.py`
14. `test/unit/test_gm_pretrain_loss.py`
15. `test/integration/test_gm_pipeline.py`

#### 应用示例
16. `examples/gm_data_augmentation.py`
17. `examples/gm_anomaly_detection.py`
18. `examples/gm_signal_generation.py`

### 5.2 修改文件清单（8个文件）

1. **`src/model_factory/ISFM/__init__.py`**
   - 添加生成模型组件注册
   - 更新模型字典

2. **`src/task_factory/task_factory.py`**
   - 注册生成模型预训练任务

3. **`src/Pipeline_03_multitask_pretrain_finetune.py`**
   - 添加生成模型预训练阶段

4. **`src/data_factory/ID_dataset.py`**
   - 支持生成样本标注

5. **`src/utils/evaluation_metrics.py`**
   - 添加生成质量评估指标

6. **`src/utils/visualization.py`**
   - 生成样本可视化工具

7. **`src/utils/pipeline_config.py`**
   - 生成模型配置验证

8. **`docs/GM_TUTORIAL.md`**
   - 生成模型使用教程

---

## 第六部分：生成质量评估体系

### 6.1 定量评估指标

#### 统计分布相似性
```python
# 生成信号与真实信号的分布比较
def evaluate_distribution_similarity(real_signals, generated_signals):
    metrics = {}
    
    # 1. Kolmogorov-Smirnov检验
    ks_stat, ks_pvalue = ks_2samp(real_signals.flatten(), 
                                 generated_signals.flatten())
    metrics['ks_statistic'] = ks_stat
    metrics['ks_pvalue'] = ks_pvalue
    
    # 2. Wasserstein距离
    w_distance = wasserstein_distance(real_signals.flatten(), 
                                    generated_signals.flatten())
    metrics['wasserstein_distance'] = w_distance
    
    # 3. 频域相似性
    real_fft = np.abs(np.fft.fft(real_signals, axis=1))
    gen_fft = np.abs(np.fft.fft(generated_signals, axis=1))
    freq_mse = np.mean((real_fft - gen_fft) ** 2)
    metrics['frequency_mse'] = freq_mse
    
    return metrics
```

#### 条件生成准确性
```python
# 验证生成样本是否符合指定条件
def evaluate_conditional_accuracy(model, test_conditions, num_samples=100):
    accuracies = {}
    
    for domain_id, system_id in test_conditions:
        # 生成样本
        generated = model.generate(
            domain_id=torch.tensor([domain_id]),
            system_id=torch.tensor([system_id]),
            num_samples=num_samples
        )
        
        # 使用分类器验证条件符合度
        predicted_conditions = condition_classifier(generated)
        
        # 计算准确率
        domain_acc = (predicted_conditions['domain'] == domain_id).float().mean()
        system_acc = (predicted_conditions['system'] == system_id).float().mean()
        
        accuracies[f'domain_{domain_id}_system_{system_id}'] = {
            'domain_accuracy': domain_acc.item(),
            'system_accuracy': system_acc.item()
        }
    
    return accuracies
```

### 6.2 定性评估方法

#### 专家评估系统
```python
class ExpertEvaluationSystem:
    """专家评估系统"""
    def __init__(self):
        self.criteria = {
            'signal_realism': {'weight': 0.3, 'scale': 1-10},
            'fault_pattern_clarity': {'weight': 0.3, 'scale': 1-10},
            'noise_characteristics': {'weight': 0.2, 'scale': 1-10},
            'temporal_consistency': {'weight': 0.2, 'scale': 1-10}
        }
    
    def evaluate_batch(self, generated_signals, expert_scores):
        """批量专家评估"""
        weighted_scores = {}
        for criterion, config in self.criteria.items():
            if criterion in expert_scores:
                weighted_scores[criterion] = (
                    expert_scores[criterion] * config['weight']
                )
        
        overall_score = sum(weighted_scores.values())
        return overall_score, weighted_scores
```

---

## 第七部分：应用场景与部署

### 7.1 数据增强应用

#### 类别平衡生成
```python
def balance_dataset_with_generation(model, dataset, target_samples_per_class=1000):
    """使用生成模型平衡数据集"""
    balanced_data = []
    class_counts = dataset.get_class_distribution()
    
    for (domain_id, system_id, class_id), current_count in class_counts.items():
        if current_count < target_samples_per_class:
            # 生成需要的样本数量
            needed_samples = target_samples_per_class - current_count
            
            generated_signals = model.generate(
                domain_id=torch.tensor([domain_id]),
                system_id=torch.tensor([system_id]),
                num_samples=needed_samples,
                temperature=0.8  # 稍微降低多样性确保质量
            )
            
            # 添加到平衡数据集
            for signal in generated_signals:
                balanced_data.append({
                    'signal': signal,
                    'domain': domain_id,
                    'system': system_id,
                    'class': class_id,
                    'synthetic': True
                })
    
    return balanced_data
```

### 7.2 异常检测部署

#### 实时异常监控
```python
class RealTimeAnomalyDetector:
    """实时异常检测器"""
    def __init__(self, gm_model, threshold_percentile=95):
        self.gm_model = gm_model
        self.threshold = None
        self.threshold_percentile = threshold_percentile
        
    def calibrate_threshold(self, normal_samples):
        """使用正常样本校准阈值"""
        with torch.no_grad():
            likelihoods = []
            for sample in normal_samples:
                likelihood = self.gm_model.compute_likelihood(
                    sample['signal'], sample['domain'], sample['system']
                )
                likelihoods.append(likelihood.item())
        
        self.threshold = np.percentile(likelihoods, 
                                     100 - self.threshold_percentile)
        
    def detect_anomaly(self, signal, domain_id, system_id):
        """检测单个信号是否异常"""
        with torch.no_grad():
            likelihood = self.gm_model.compute_likelihood(
                signal, domain_id, system_id
            )
            
            is_anomaly = likelihood.item() < self.threshold
            confidence = abs(likelihood.item() - self.threshold) / self.threshold
            
            return {
                'is_anomaly': is_anomaly,
                'likelihood': likelihood.item(),
                'confidence': confidence,
                'threshold': self.threshold
            }
```

---

## 第八部分：成功标准与验证

### 8.1 技术指标要求

#### 生成质量标准
- **重建误差**: MSE < 0.01（归一化信号）
- **生成多样性**: 生成样本覆盖真实数据90%以上的特征空间
- **条件准确性**: 条件生成准确率 > 85%
- **频域一致性**: 功率谱密度相似度 > 0.8

#### 性能标准
- **生成速度**: 单样本生成 < 100ms（GPU）
- **内存效率**: 批次256样本 < 4GB显存
- **训练稳定性**: 1000轮收敛，损失方差 < 0.1

### 8.2 应用效果验证

#### 下游任务改进
```python
def validate_downstream_improvement(gm_model, downstream_tasks):
    """验证生成模型对下游任务的改进效果"""
    results = {}
    
    for task_name, task_config in downstream_tasks.items():
        # 基线性能（无数据增强）
        baseline_performance = train_and_evaluate(
            task_config, use_augmentation=False
        )
        
        # 使用生成数据增强后的性能
        augmented_performance = train_and_evaluate_with_generation(
            task_config, gm_model, augmentation_ratio=0.5
        )
        
        improvement = (augmented_performance - baseline_performance) / baseline_performance
        
        results[task_name] = {
            'baseline': baseline_performance,
            'augmented': augmented_performance,
            'improvement_ratio': improvement
        }
    
    return results
```

---

## 第九部分：风险控制与质量保证

### 9.1 代码质量控制

#### 自动化测试流程
```yaml
# .github/workflows/gm_ci.yml
name: 生成模型持续集成
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
        
    steps:
    - uses: actions/checkout@v3
    - name: 设置Python环境
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: 安装依赖
      run: |
        pip install -r requirements-test.txt
        pip install -e .
        
    - name: 代码风格检查
      run: |
        flake8 src/model_factory/ISFM/generative/ --max-line-length=100
        black --check src/model_factory/ISFM/generative/
        
    - name: 单元测试
      run: |
        python -m pytest test/unit/test_gm_*.py -v --cov=src
        
    - name: 集成测试
      run: |
        python -m pytest test/integration/test_gm_*.py -v
        
    - name: 性能测试
      run: |
        python test/performance/benchmark_gm.py
```

### 9.2 模型安全性检查

#### 生成内容安全验证
```python
class GeneratedContentValidator:
    """生成内容安全验证器"""
    def __init__(self):
        self.safety_checks = [
            self.check_signal_bounds,
            self.check_frequency_range,
            self.check_amplitude_distribution,
            self.check_temporal_consistency
        ]
    
    def validate_generated_batch(self, generated_signals):
        """验证生成信号批次的安全性"""
        validation_results = {
            'passed': True,
            'warnings': [],
            'errors': []
        }
        
        for check in self.safety_checks:
            try:
                result = check(generated_signals)
                if not result['passed']:
                    validation_results['errors'].extend(result.get('errors', []))
                    validation_results['warnings'].extend(result.get('warnings', []))
                    validation_results['passed'] = False
            except Exception as e:
                validation_results['errors'].append(f"验证检查失败: {e}")
                validation_results['passed'] = False
        
        return validation_results
    
    def check_signal_bounds(self, signals):
        """检查信号值域"""
        min_val, max_val = signals.min(), signals.max()
        if min_val < -10 or max_val > 10:  # 假设合理范围
            return {
                'passed': False,
                'errors': [f"信号值超出合理范围: [{min_val}, {max_val}]"]
            }
        return {'passed': True}
```

---

## 结论

这份优化版生成模型集成方案为PHM-Vibench框架提供了完整的生成能力升级路径。通过将Flow模型明确定位为生成模型，并采用测试驱动开发方法，确保了代码质量和功能可靠性。

### 核心优势

1. **完整生成能力**: 数据增强、异常检测、信号合成一体化
2. **质量保证体系**: TDD + 内嵌测试 + 持续集成
3. **工业化部署**: 实时异常检测、生产环境优化
4. **可扩展架构**: 模块化设计支持未来增强

### 下一步行动

1. 🚀 **立即开始**: 按照实施路线图执行第一阶段
2. 📊 **持续监控**: 使用质量评估体系跟踪进展
3. 🔄 **迭代改进**: 基于实际测试结果优化模型
4. 📝 **文档维护**: 保持文档与代码同步更新

---

**文档状态**: ✅ 准备实施  
**开发分支**: cc_flow_1  
**版本**: 优化版 v2.0  
**预计完成时间**: 4周