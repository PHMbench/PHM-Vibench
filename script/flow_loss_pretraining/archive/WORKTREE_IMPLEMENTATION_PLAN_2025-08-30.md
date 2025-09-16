# PHM-Vibench Flow Loss 预训练实施计划 - Worktree版

**创建日期：2025年8月30日**  
**版本：V2.0 - Worktree多进程开发版**  
**基于：CFL.ipynb + 优化的生成模型架构**

---

## 🌳 Worktree 开发环境

### Git Worktree 配置

已成功建立多进程开发环境：

```bash
# 主工作树 (stable development)
/home/lq/LQcode/2_project/PHMBench/PHM-Vibench       [main]

# Flow开发树 (experimental features) 
/home/lq/LQcode/2_project/PHMBench/PHM-Vibench-flow  [cc_flow_1]
```

### 开发流程

1. **并行开发**：主分支继续稳定开发，flow分支实验新功能
2. **独立测试**：每个分支可独立运行测试，互不干扰
3. **快速切换**：`cd ../PHM-Vibench-flow` 即可切换到flow开发环境
4. **安全合并**：flow分支成熟后再合并到主分支

---

## 📋 四周实施计划

### 第一周：生成模块实现 (2025-08-30 ~ 2025-09-06)

#### 目标：建立核心生成模块

**工作目录**：`PHM-Vibench-flow/src/model_factory/`

#### 1.1 矫正流生成网络 - GM_01_RectifiedFlow.py

**位置**：`src/model_factory/GM/GM_01_RectifiedFlow.py`

**核心功能**：
- 矫正流匹配 (Rectified Flow Matching)
- 条件生成 (Conditional Generation)  
- 噪声到数据的直线插值

**关键组件**：

```python
class GM_01_RectifiedFlow(nn.Module):
    """矫正流生成模型"""
    
    def __init__(self, args_m):
        # 配置参数
        self.latent_dim = getattr(args_m, 'latent_dim', 128)
        self.condition_dim = getattr(args_m, 'condition_dim', 64)
        self.hidden_dim = getattr(args_m, 'hidden_dim', 256)
        
        # 速度网络 - 核心组件
        self.velocity_net = FlowMLP(...)
        
    def forward(self, x, condition=None):
        """训练时前向传播"""
        # 1. 采样时间步 t ~ Uniform[0, 1]
        # 2. 采样噪声 z ~ N(0, I)
        # 3. 线性插值 x_t = (1-t)*z + t*x
        # 4. 预测速度 v = velocity_net(x_t, t, condition)
        # 5. 返回损失计算所需项
        
    def sample(self, batch_size, condition=None, num_steps=50):
        """生成新样本"""
        # 从高斯噪声开始，逐步演化到数据分布
```

**自测试代码**：
```python
if __name__ == '__main__':
    """矫正流生成网络测试"""
    # Mock配置、模型初始化、前向传播
    # 损失计算、梯度反传、采样生成
    # 插值功能、性能测试
```

#### 1.2 条件编码器 - E_03_ConditionalEncoder.py

**位置**：`src/model_factory/ISFM/embedding/E_03_ConditionalEncoder.py`

**核心功能**：
- 域/系统条件编码
- 层次化表示学习
- 与现有HSE模块兼容

```python
class E_03_ConditionalEncoder(nn.Module):
    """条件编码器：域、系统、实例层次编码"""
    
    def __init__(self, args_m):
        # 域编码器、系统编码器、实例编码器
        self.domain_encoder = nn.Embedding(...)
        self.system_encoder = nn.Embedding(...)
        self.instance_encoder = nn.Sequential(...)
        
    def forward(self, x, domain_id, system_id):
        """生成层次化条件表示"""
        # 域级表示、系统级表示、实例级表示
        # 层次化融合策略
```

#### 1.3 主生成模型 - M_04_ISFM_GM.py

**位置**：`src/model_factory/ISFM/M_04_ISFM_GM.py`

**核心功能**：
- 整合矫正流与ISFM架构
- 支持多任务预训练
- 保持工厂模式兼容性

```python
class Model(nn.Module):  # 必须命名为Model以符合工厂模式
    """ISFM生成模型 - 整合矫正流"""
    
    def __init__(self, args_m, metadata):
        # 标准ISFM组件
        self.embedding = Embedding_dict[args_m.embedding](args_m)
        self.backbone = Backbone_dict[args_m.backbone](args_m)
        
        # 新增生成组件
        self.flow_net = GM_01_RectifiedFlow(args_m)
        self.condition_encoder = E_03_ConditionalEncoder(args_m)
        
    def forward(self, x, file_id=None, task_id='generation'):
        if task_id == 'generation':
            # 生成任务流程
        elif task_id in ['classification', 'prediction']:
            # 传统任务流程
```

---

### 第二周：损失函数与训练任务 (2025-09-06 ~ 2025-09-13)

#### 目标：构建训练框架

**工作目录**：`PHM-Vibench-flow/src/task_factory/`

#### 2.1 Flow匹配损失 - flow_loss.py

**位置**：`src/task_factory/Components/flow_loss.py`

```python
@dataclass
class FlowLossCfg:
    """Flow损失配置"""
    lambda_flow: float = 1.0      # 流匹配损失权重
    lambda_recon: float = 1.0     # 重建损失权重
    lambda_hier: float = 0.5      # 层次化损失权重
    lambda_reg: float = 1e-3      # 正则化损失权重
    target_radius: float = 1.0    # 目标半径
    margin: float = 0.1           # 层次边际

class FlowMatchingLoss(nn.Module):
    """矫正流匹配损失"""
    
    def forward(self, model_outputs):
        # 计算流匹配损失：MSE(v_pred, v_true)
        # 计算层次化对比损失
        # 计算正则化损失
        
class HierarchicalFlowLoss(nn.Module):
    """层次化流损失 - 域>系统>实例"""
    
    def forward(self, features, domain_labels, system_labels):
        # 域间分离损失
        # 系统间分离损失  
        # 层次边际损失
```

#### 2.2 预训练任务 - pretrain_flow_task.py

**位置**：`src/task_factory/task/pretrain/pretrain_flow_task.py`

```python
class FlowPretrainTask(pl.LightningModule):
    """基于流匹配的预训练任务"""
    
    def __init__(self, network, args_data, args_model, args_task, 
                 args_trainer, args_environment, metadata):
        super().__init__()
        self.network = network
        self.flow_loss = FlowMatchingLoss(args_task.flow_cfg)
        self.hier_loss = HierarchicalFlowLoss(args_task.hier_cfg)
        
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        (x, y), data_name = batch
        
        # 提取元数据
        file_ids = [item['file_id'] for item in y]
        domain_ids = [self.metadata[fid]['domain'] for fid in file_ids]
        system_ids = [self.metadata[fid]['system'] for fid in file_ids]
        
        # 前向传播
        outputs = self.network(x, file_ids, task_id='generation')
        
        # 损失计算
        flow_loss = self.flow_loss(outputs)
        hier_loss = self.hier_loss(outputs['features'], domain_ids, system_ids)
        
        total_loss = flow_loss + self.args_task.hier_weight * hier_loss
        
        # 日志记录
        self.log('train_flow_loss', flow_loss)
        self.log('train_hier_loss', hier_loss)
        self.log('train_total_loss', total_loss)
        
        return total_loss
        
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        # 类似训练步骤，但不更新参数
        
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.args_task.lr,
            weight_decay=self.args_task.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.args_trainer.max_epochs
        )
        
        return [optimizer], [scheduler]
```

---

### 第三周：配置与测试 (2025-09-13 ~ 2025-09-20)

#### 目标：完善配置系统和测试框架

**工作目录**：`PHM-Vibench-flow/`

#### 3.1 配置文件

**基础流预训练配置** - `configs/flow_pretrain/base_flow.yaml`:

```yaml
# 环境配置
environment:
  cuda: true
  device: "cuda:0"
  seed: 42

# 数据配置
data:
  data_dir: "./data"
  datasets: ["CWRU", "XJTU", "FEMTO"]
  sample_rate: 16000
  sequence_length: 1024
  batch_size: 32
  
# 模型配置
model:
  name: "M_04_ISFM_GM"
  type: "ISFM"
  embedding: "E_03_ConditionalEncoder"
  backbone: "B_08_PatchTST"
  
  # GM特有配置
  latent_dim: 128
  condition_dim: 64
  hidden_dim: 256
  num_layers: 4
  
# 任务配置
task:
  name: "flow_pretrain"
  type: "pretrain"
  
  # Flow损失配置
  flow_cfg:
    lambda_flow: 1.0
    lambda_recon: 1.0
    lambda_hier: 0.5
    lambda_reg: 1e-3
    target_radius: 1.0
    margin: 0.1
    
  # 训练参数
  lr: 5e-4
  weight_decay: 1e-4
  hier_weight: 0.1
  
# 训练配置
trainer:
  max_epochs: 100
  check_val_every_n_epoch: 5
  precision: 16
  gradient_clip_val: 1.0
```

**层次化流配置** - `configs/flow_pretrain/hierarchical_flow.yaml`:

```yaml
# 继承基础配置
base_config: "./base_flow.yaml"

# 强化层次化学习
task:
  hier_cfg:
    domain_weight: 1.0      # 域分离权重
    system_weight: 0.8      # 系统分离权重
    instance_weight: 0.5    # 实例对比权重
    margin: 0.2            # 增大边际
    
  hier_weight: 0.3         # 提高层次损失权重

# 扩展训练
trainer:
  max_epochs: 200
```

#### 3.2 测试脚本

**矫正流测试** - `test/flow/test_rectified_flow.py`:

```python
import pytest
import torch
from src.model_factory.GM.GM_01_RectifiedFlow import GM_01_RectifiedFlow

class TestRectifiedFlow:
    """矫正流网络测试套件"""
    
    def setup_method(self):
        """测试初始化"""
        self.config = MockConfig()
        self.model = GM_01_RectifiedFlow(self.config)
        
    def test_initialization(self):
        """测试模型初始化"""
        assert self.model.latent_dim == 128
        assert self.model.condition_dim == 64
        
    def test_forward_pass(self):
        """测试前向传播"""
        batch_size = 16
        x = torch.randn(batch_size, self.config.latent_dim)
        condition = torch.randn(batch_size, self.config.condition_dim)
        
        outputs = self.model(x, condition)
        
        assert 'v_pred' in outputs
        assert 'v_true' in outputs
        assert outputs['v_pred'].shape == outputs['v_true'].shape
        
    def test_loss_computation(self):
        """测试损失计算"""
        # 前向传播
        outputs = self.get_sample_outputs()
        
        # 计算损失
        loss = self.model.compute_loss(outputs)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        
    def test_sampling(self):
        """测试采样生成"""
        samples = self.model.sample(batch_size=8, num_steps=10)
        
        assert samples.shape == (8, self.config.latent_dim)
        assert not torch.isnan(samples).any()
        
    def test_interpolation(self):
        """测试插值功能"""
        x0 = torch.randn(4, self.config.latent_dim)
        x1 = torch.randn(4, self.config.latent_dim)
        t = torch.tensor([0.0, 0.33, 0.67, 1.0])
        
        interpolated = self.model.interpolate(x0, x1, t)
        
        # 验证边界条件
        assert torch.allclose(interpolated[0], x0[0], atol=1e-6)
        assert torch.allclose(interpolated[-1], x1[-1], atol=1e-6)
```

**集成测试** - `test/flow/test_integration.py`:

```python
class TestFlowIntegration:
    """流预训练集成测试"""
    
    def test_end_to_end_training(self):
        """端到端训练测试"""
        # 加载配置
        config = load_config('configs/flow_pretrain/base_flow.yaml')
        
        # 创建数据加载器
        dataloader = create_dataloader(config)
        
        # 创建模型
        model = model_factory(config.model, metadata)
        
        # 创建任务
        task = FlowPretrainTask(model, **config)
        
        # 短期训练测试
        trainer = pl.Trainer(max_epochs=2, fast_dev_run=True)
        trainer.fit(task, dataloader)
        
        assert task.current_epoch > 0
        
    def test_multi_dataset_training(self):
        """多数据集训练测试"""
        # 测试跨数据集的层次化学习
        
    def test_downstream_performance(self):
        """下游任务性能测试"""
        # 预训练后的分类性能评估
```

---

### 第四周：应用与优化 (2025-09-20 ~ 2025-09-27)

#### 目标：下游应用集成与性能优化

#### 4.1 数据增强应用

**增强管道** - `src/utils/flow_augmentation.py`:

```python
class FlowAugmentation:
    """基于流模型的数据增强"""
    
    def __init__(self, flow_model_path):
        self.flow_model = load_flow_model(flow_model_path)
        
    def augment_dataset(self, dataset, num_samples_per_class=100):
        """数据集增强"""
        augmented_data = []
        
        for class_id in dataset.classes:
            # 提取类条件
            condition = self.extract_class_condition(class_id)
            
            # 生成新样本
            new_samples = self.flow_model.sample(
                batch_size=num_samples_per_class,
                condition=condition
            )
            
            augmented_data.extend(new_samples)
            
        return augmented_data
```

#### 4.2 异常检测应用

**异常检测器** - `src/utils/flow_anomaly.py`:

```python
class FlowAnomalyDetector:
    """基于流模型的异常检测"""
    
    def detect_anomaly(self, x, threshold=None):
        """异常检测"""
        # 计算重建误差
        recon_error = self.compute_reconstruction_error(x)
        
        # 计算流密度
        flow_density = self.compute_flow_density(x)
        
        # 异常分数
        anomaly_score = recon_error / (flow_density + 1e-8)
        
        return anomaly_score > threshold
```

#### 4.3 性能优化

**优化策略**：
1. **混合精度训练**：使用AMP加速训练
2. **梯度累积**：处理大批量训练
3. **模型蒸馏**：压缩部署模型
4. **缓存机制**：加速重复计算

---

## 🚀 开发工作流

### 日常开发流程

```bash
# 1. 切换到flow开发环境
cd /home/lq/LQcode/2_project/PHMBench/PHM-Vibench-flow

# 2. 确认分支
git branch --show-current  # 应显示 cc_flow_1

# 3. 开发和测试
python -m pytest test/flow/ -v

# 4. 提交更改
git add .
git commit -m "feat: implement rectified flow network"

# 5. 推送到远程
git push origin cc_flow_1
```

### 合并到主分支

```bash
# 1. 切换到主工作树
cd /home/lq/LQcode/2_project/PHMBench/PHM-Vibench

# 2. 更新主分支
git checkout main
git pull origin main

# 3. 合并flow分支
git merge cc_flow_1

# 4. 解决冲突（如有）
# 5. 运行完整测试
python run_tests.py

# 6. 推送合并结果
git push origin main
```

---

## 📊 里程碑检查点

### Week 1 检查点 ✓
- [ ] GM_01_RectifiedFlow 实现完成
- [ ] E_03_ConditionalEncoder 实现完成  
- [ ] M_04_ISFM_GM 实现完成
- [ ] 所有模块通过自测试

### Week 2 检查点 ✓
- [ ] FlowMatchingLoss 实现完成
- [ ] HierarchicalFlowLoss 实现完成
- [ ] FlowPretrainTask 实现完成
- [ ] 训练循环正常运行

### Week 3 检查点 ✓
- [ ] 配置文件系统完成
- [ ] 单元测试套件完成
- [ ] 集成测试通过
- [ ] 文档完整

### Week 4 检查点 ✓
- [ ] 下游应用集成完成
- [ ] 性能优化完成
- [ ] 最终测试通过
- [ ] 准备合并到主分支

---

## 🎯 预期成果

### 技术成果
1. **完整的流匹配预训练系统**
2. **层次化表示学习能力**
3. **条件生成功能**
4. **改进的少样本学习性能**

### 代码成果
- **18个新文件**：核心模块、损失函数、任务、配置、测试
- **8个修改文件**：现有模块的扩展和集成
- **完整的测试覆盖**：单元测试、集成测试、性能测试

### 应用成果
- **数据增强**：提高数据稀缺场景的性能
- **异常检测**：更好的分布外检测能力
- **域适应**：跨数据集的泛化能力

---

## 🔧 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少批量大小
   - 使用梯度累积
   - 启用混合精度

2. **训练不收敛**
   - 检查学习率设置
   - 验证数据预处理
   - 调整损失权重

3. **生成质量差**
   - 增加采样步数
   - 调整流网络架构
   - 检查条件编码

### 调试工具

```python
# 1. 可视化生成样本
plot_generated_samples(model, condition)

# 2. 监控损失曲线
wandb.log({'flow_loss': loss_value})

# 3. 检查梯度
check_gradients(model)

# 4. 分析条件空间
visualize_condition_space(encoder)
```

---

这个计划为PHM-Vibench的流损失预训练提供了完整的roadmap，利用worktree实现安全的并行开发，确保实验性功能不影响主分支的稳定性。