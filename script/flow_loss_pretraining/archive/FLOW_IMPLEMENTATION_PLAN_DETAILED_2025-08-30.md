# Flow模型预训练详细实施计划

**创建日期：2025年8月30日**  
**版本：V4.0 - 详细实施版**  
**基于：GM_FLOW_PLAN_OPTIMIZED_2025-08-30.md 分析结果**

---

## 🎯 执行摘要

基于对现有Flow计划的深度分析，本计划解决了关键的技术集成问题，提供了可执行的分阶段实施方案，确保Rectified Flow生成模型成功集成到PHM-Vibench框架中。

### 核心改进点
- ✅ **维度兼容性修复**: 适配(B,L,C)张量格式
- ✅ **元数据集成**: 使用file_id提取层次化条件信息  
- ✅ **工厂模式合规**: 遵循PHM-Vibench架构模式
- ✅ **数值稳定性**: 完整的错误处理和验证
- ✅ **测试驱动**: 全面的单元和集成测试

---

## 📊 技术架构详细设计

### 1. 核心组件架构

```
Flow-based Pretraining System
├── Sequence Adapter          # 维度适配层
│   ├── flatten_sequence()   # (B,L,C) → (B,L*C)
│   └── unflatten_sequence() # (B,L*C) → (B,L,C)
├── Rectified Flow Model      # 核心生成模型
│   ├── velocity_network()   # 速度场预测
│   ├── flow_matching()      # 流匹配训练
│   └── ode_sampling()       # ODE积分采样
├── Conditional Encoder       # 层次化条件编码
│   ├── domain_encoder()     # 域级编码
│   ├── system_encoder()     # 系统级编码
│   └── instance_encoder()   # 实例级编码
└── Flow Utilities           # 辅助工具
    ├── solvers/            # ODE求解器
    ├── schedulers/         # 噪声调度
    └── metrics/           # 评估指标
```

### 2. 维度处理策略

#### 问题分析
- **现有代码**: 假设输入为`(batch_size, latent_dim)`
- **PHM-Vibench实际**: 使用`(batch_size, sequence_length, channels)`格式
- **典型参数**: sequence_length=1024, channels=1-3

#### 解决方案设计

```python
class SequenceAdapter(nn.Module):
    """序列维度适配器 - 处理3D张量格式转换"""
    
    def __init__(self, seq_len: int, channels: int, latent_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.latent_dim = latent_dim
        
        # 方案1: 直接展开(推荐)
        self.use_flatten = True
        
        # 方案2: 卷积降维(备选)
        if not self.use_flatten:
            self.conv_encoder = nn.Conv1d(channels, latent_dim//seq_len, 1)
            self.conv_decoder = nn.Conv1d(latent_dim//seq_len, channels, 1)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码: (B, L, C) -> (B, D)
        """
        B, L, C = x.shape
        if self.use_flatten:
            return x.view(B, L * C)
        else:
            # 使用卷积降维
            x = x.transpose(1, 2)  # (B, C, L)
            x = self.conv_encoder(x)  # (B, D/L, L)
            return x.view(B, -1)  # (B, D)
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        解码: (B, D) -> (B, L, C)
        """
        B = x.shape[0]
        if self.use_flatten:
            return x.view(B, self.seq_len, self.channels)
        else:
            # 使用卷积升维
            x = x.view(B, self.latent_dim//self.seq_len, self.seq_len)
            x = self.conv_decoder(x)  # (B, C, L)
            return x.transpose(1, 2)  # (B, L, C)
```

### 3. 元数据集成系统

#### 层次化信息提取

```python
class MetadataConditionExtractor:
    """
    从PHM-Vibench现有metadata直接提取条件信息
    避免冗余映射表，支持未知域和系统处理
    """
    
    @staticmethod
    def extract_conditions(metadata_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        从metadata字典提取条件信息
        
        Args:
            metadata_dict: 单个样本的metadata信息（metadata[file_id]）
            
        Returns:
            包含domain_id, system_id等的条件字典
        """
        # 直接使用PHM-Vibench metadata中的值
        domain_id = metadata_dict.get('Domain_id', -1)  # -1表示未知
        system_id = metadata_dict.get('Dataset_id', -1)  # -1表示未知
        
        # 处理pandas NaN和None值
        if pd.isna(domain_id) or domain_id is None:
            domain_id = -1  # 未知域
        if pd.isna(system_id) or system_id is None:
            system_id = -1  # 未知系统
            
        return {
            'domain_id': int(domain_id),
            'system_id': int(system_id),
            'dataset_name': metadata_dict.get('Name', 'unknown'),
            'label': metadata_dict.get('Label', -1),
            'sample_rate': metadata_dict.get('Sample_rate', 0)
        }
    
    @staticmethod
    def get_metadata_statistics(metadata_df) -> Dict[str, int]:
        """
        从metadata DataFrame统计域和系统的数量
        
        Args:
            metadata_df: PHM-Vibench的metadata DataFrame
            
        Returns:
            统计信息字典
        """
        # 统计有效的域和系统ID
        valid_domains = metadata_df['Domain_id'].dropna()
        valid_systems = metadata_df['Dataset_id'].dropna()
        
        # 转换为整数类型
        try:
            valid_domains = valid_domains.astype(int)
            valid_systems = valid_systems.astype(int)
        except:
            pass  # 如果无法转换，保持原类型
        
        unique_domains = valid_domains.unique()
        unique_systems = valid_systems.unique()
        
        return {
            'num_domains': len(unique_domains),
            'num_systems': len(unique_systems),
            'max_domain_id': int(max(unique_domains)) if len(unique_domains) > 0 else -1,
            'max_system_id': int(max(unique_systems)) if len(unique_systems) > 0 else -1,
            'domain_ids': sorted(unique_domains.tolist()),
            'system_ids': sorted(unique_systems.tolist())
        }
```

### 4. 增强的ODE求解器

```python
class FlowODESolver:
    """高精度ODE求解器集合"""
    
    def __init__(self, solver_type: str = 'euler'):
        self.solver_type = solver_type
        self.solver_registry = {
            'euler': self.euler_step,
            'heun': self.heun_step,
            'rk4': self.rk4_step,
            'adaptive': self.adaptive_step
        }
    
    def euler_step(self, model, x, t, dt, condition=None):
        """一阶欧拉方法"""
        with torch.no_grad():
            t_tensor = torch.full((x.size(0),), t, device=x.device)
            t_emb = model.time_embedding(t_tensor)
            v = model.velocity_net(x, t_emb, condition)
            return x + dt * v
    
    def heun_step(self, model, x, t, dt, condition=None):
        """二阶Heun方法 (改进的欧拉法)"""
        with torch.no_grad():
            # 第一步预测
            t_tensor = torch.full((x.size(0),), t, device=x.device)
            t_emb = model.time_embedding(t_tensor)
            k1 = model.velocity_net(x, t_emb, condition)
            x_temp = x + dt * k1
            
            # 第二步校正
            t_next_tensor = torch.full((x.size(0),), t + dt, device=x.device)
            t_next_emb = model.time_embedding(t_next_tensor)
            k2 = model.velocity_net(x_temp, t_next_emb, condition)
            
            # 最终结果
            return x + dt * (k1 + k2) / 2
    
    def rk4_step(self, model, x, t, dt, condition=None):
        """四阶Runge-Kutta方法"""
        with torch.no_grad():
            # k1
            t_tensor = torch.full((x.size(0),), t, device=x.device)
            t_emb = model.time_embedding(t_tensor)
            k1 = model.velocity_net(x, t_emb, condition)
            
            # k2
            x2 = x + dt * k1 / 2
            t2_tensor = torch.full((x.size(0),), t + dt/2, device=x.device)
            t2_emb = model.time_embedding(t2_tensor)
            k2 = model.velocity_net(x2, t2_emb, condition)
            
            # k3
            x3 = x + dt * k2 / 2
            k3 = model.velocity_net(x3, t2_emb, condition)
            
            # k4
            x4 = x + dt * k3
            t4_tensor = torch.full((x.size(0),), t + dt, device=x.device)
            t4_emb = model.time_embedding(t4_tensor)
            k4 = model.velocity_net(x4, t4_emb, condition)
            
            # 最终结果
            return x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def adaptive_step(self, model, x, t, dt, condition=None, tol=1e-5):
        """自适应步长控制"""
        # 使用全步长
        x1 = self.rk4_step(model, x, t, dt, condition)
        
        # 使用两个半步长
        x_half = self.rk4_step(model, x, t, dt/2, condition)
        x2 = self.rk4_step(model, x_half, t + dt/2, dt/2, condition)
        
        # 估计误差
        error = torch.norm(x1 - x2, dim=-1).max()
        
        if error < tol:
            return x2, dt  # 接受步长
        else:
            # 减小步长重新计算
            new_dt = dt * 0.8 * (tol / error) ** 0.2
            return self.adaptive_step(model, x, t, new_dt, condition, tol)
```

---

## 📝 重要更新说明

### 基于反馈的关键改进

#### 1. 避免冗余映射表
**原方案问题**：创建了人工的`DATASET_DOMAIN_MAPPING`和`SYSTEM_TYPE_MAPPING`
**改进后方案**：
- ✅ 直接使用PHM-Vibench现有的`metadata[file_id]['Dataset_id']`
- ✅ 直接使用PHM-Vibench现有的`metadata[file_id]['Domain_id']`  
- ✅ 无需维护额外的映射关系

#### 2. 智能处理未知值
**支持场景**：
- ✅ `Domain_id`或`Dataset_id`为NaN/None
- ✅ 新增数据集的动态适应
- ✅ metadata字段缺失的容错处理

**处理策略**：
```python
# 缺失值统一处理为-1，然后映射到padding_idx=0
domain_id = metadata_dict.get('Domain_id', -1)
if pd.isna(domain_id) or domain_id is None:
    domain_id = -1  # 标记为未知
# 在embedding时：-1 -> 0 (padding_idx)
```

#### 3. 动态容量分配
**智能统计**：
```python
stats = MetadataConditionExtractor.get_metadata_statistics(metadata.df)
args_m.num_domains = max(stats['num_domains'], 10) + 10  # 预留扩展空间
args_m.num_systems = max(stats['num_systems'], 10) + 10
```

#### 4. 文件结构对齐
- `components/` → `layers/` (与用户编辑一致)
- 删除`metadata_extractor.py`(不再需要)
- 保持与PHM-Vibench现有架构的一致性

---

## 🔧 简化后的分阶段实施方案

**设计原则**: 避免炫技复杂度，一个方案优于多个选择

### Phase 1: 最小可行版本 (第1-4天)

#### 1.1 简化的项目结构

```bash
# 最小化文件结构
src/model_factory/ISFM/
├── M_04_ISFM_Flow.py           # 主集成模型
├── layers/
│   ├── __init__.py
│   ├── flow_model.py           # RectifiedFlow核心(合并原GM_01)
│   └── condition_encoder.py    # 条件编码(合并原E_03)
├── utils/
│   ├── __init__.py
│   └── flow_utils.py           # 必要工具函数(维度适配等)
└── tests/
    ├── test_flow_basics.py     # 基础功能测试
    └── test_integration.py     # 集成测试
```

#### 1.2 实施步骤 (渐进式实现)

**Day 1-2: 核心Flow模型**
- [ ] 创建 `flow_model.py` - 基础RectifiedFlow
- [ ] 仅实现Euler ODE求解器 (最简单)
- [ ] 直接展开维度适配 (B,L,C) → (B,L*C)
- [ ] 基础前向传播和采样

**Day 3-4: 条件编码与集成**
- [ ] 创建 `condition_encoder.py` - 直接使用metadata
- [ ] 支持Dataset_id和Domain_id (无映射表)
- [ ] 集成到主模型 `M_04_ISFM_Flow.py`
- [ ] 基础测试验证

#### 1.3 验收标准
- ✅ 所有形状测试通过 (>95%覆盖率)
- ✅ 支持可变序列长度 (512-4096)
- ✅ 内存使用合理 (<8GB for batch_size=32)
- ✅ 与现有数据加载器兼容

### Phase 2: 功能完善 (第5-8天)

#### 2.1 训练系统集成

**Day 5-6: 损失函数与训练任务**
- [ ] 实现基础RectifiedFlow损失函数
- [ ] 创建预训练任务类集成TaskFactory
- [ ] 基础配置文件模板
- [ ] 端到端训练测试

**Day 7-8: 稳定性和性能**
- [ ] 添加梯度裁剪和NaN检测
- [ ] 基础性能优化(内存、速度)
- [ ] 支持单数据集训练验证
- [ ] 如需要可添加Heun求解器

#### 2.2 验收标准(务实目标)
- ✅ 端到端训练成功(10+ epochs)
- ✅ 损失收敛稳定无NaN
- ✅ 生成样本基本合理
- ✅ 与现有框架无冲突

### Phase 3: 优化提升 (第9-12天)

#### 3.1 测试与验证

**Day 9-10: 全面测试覆盖**
- [ ] 完善单元测试覆盖所有核心功能
- [ ] 多数据集兼容性测试
- [ ] 性能基准测试和内存分析
- [ ] 边界情况和错误处理测试

**Day 11-12: 性能优化与文档**
- [ ] 根据测试结果优化性能瓶颈
- [ ] 完善配置文件和使用示例
- [ ] 创建简洁的使用文档
- [ ] 准备集成到主分支

#### 3.2 验收标准(最终目标)
- ✅ 测试覆盖率 >90%
- ✅ 支持主要PHM-Vibench数据集
- ✅ 性能指标达到预期
- ✅ 代码通过review

---

## 📊 实施重点简化

### 1. 文件最小化
- `flow_model.py`: 仅基础RectifiedFlow + Euler求解器
- `condition_encoder.py`: 直接使用Dataset_id/Domain_id
- `flow_utils.py`: 维度适配等必要工具

### 2. 功能渐进式
- Phase 1: 能跑的最小版本
- Phase 2: 加入训练和损失
- Phase 3: 测试和优化

### 3. 避免过度设计
- 删除多求解器选择（先用Euler）
- 删除复杂维度适配（直接展开）
- 删除过多工具文件

---

## 💻 核心代码框架

### 基础RectifiedFlow实现

```python
# src/model_factory/ISFM/layers/flow_model.py - 最简实现
class RectifiedFlow(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.velocity_net = nn.Sequential(
            nn.Linear(latent_dim + 64, hidden_dim),  # +64 for time embedding
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, x, t):
        # 简化的前向传播
        noise = torch.randn_like(x) 
        x_t = (1 - t) * noise + t * x
        v_pred = self.velocity_net(torch.cat([x_t, self.time_emb(t)], dim=-1))
        return {'v_pred': v_pred, 'v_true': x - noise}
```

---

## ⏱️ 时间线总结

**12天总计**:
- Phase 1 (Day 1-4): 最小可行版本
- Phase 2 (Day 5-8): 训练集成
- Phase 3 (Day 9-12): 测试优化

**简化原则**: 每个阶段都有可工作的版本，避免大爆炸式开发

---

*简化版实施计划 - 专注核心功能，避免炫技复杂度*  
*更新时间：2025年8月30日*
