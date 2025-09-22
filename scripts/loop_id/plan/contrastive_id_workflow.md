# ContrastiveIDTask 分步实施工作流指南

> 🎯 **基于PHM-Vibench框架的对比学习预训练完整实施指南**  
> 遵循PHM-Vibench设计理念，提供从数据准备到生产部署的全流程操作指南

## 📋 目录

- [🚀 快速开始（5分钟上手）](#-快速开始5分钟上手)
- [📊 三大核心概念](#-三大核心概念)
- [⚙️ 分步实施工作流](#️-分步实施工作流)
- [🔧 高级用法](#-高级用法)
- [🐛 问题诊断流程](#-问题诊断流程)
- [📊 性能优化检查清单](#-性能优化检查清单)
- [🎯 快速命令参考](#-快速命令参考)

---

## 🚀 快速开始（5分钟上手）

### 适用对象
- **PHM基础模型开发者**: 想要使用对比学习预训练
- **振动信号研究者**: 需要无监督特征学习
- **工程师**: 希望提升模型泛化能力

### 第一步：环境验证

```bash
# 1. 确认PHM-Vibench环境
python -c "import torch; print(f'✅ PyTorch {torch.__version__}')"
python -c "from src.configs import load_config; print('✅ 配置系统就绪')"
python -c "from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask; print('✅ ContrastiveIDTask就绪')"
```

### 第二步：运行最小示例

```bash
# 2. 使用预设配置快速验证（1 epoch，CPU模式）
python main.py --pipeline Pipeline_ID --config configs/id_contrastive/debug.yaml
```

期待输出：
```
✅ ContrastiveIDTask初始化成功
🔄 开始训练 (1 epoch)...
📈 Epoch 1: loss=2.45, contrastive_acc=0.25
✅ 训练完成! 结果保存在: save/metadata_6_1/ContrastiveIDTask/*/
```

### 第三步：验证结果

```bash
# 3. 查看训练结果
ls -la save/*/ContrastiveIDTask/*/
# 应该看到: checkpoints/, metrics.json, log.txt, figures/
```

🎉 **恭喜！你已成功运行ContrastiveIDTask**

---

## 📊 三大核心概念

### 1. Pipeline_ID 工作流体系

ContrastiveIDTask完全集成在PHM-Vibench的Pipeline_ID中：

```python
# Pipeline_ID 调用链
main.py --pipeline Pipeline_ID 
    ↓
src/Pipeline_ID.py (委托给默认pipeline)
    ↓  
src/Pipeline_01_default.py
    ↓
Factory模式组件自动加载:
├── data_factory: id_data_factory + ID_dataset
├── model_factory: ISFM + PatchTST backbone  
├── task_factory: ContrastiveIDTask
└── trainer_factory: PyTorch Lightning
```

### 2. 配置预设系统（PHM-Vibench v5.0）

PHM-Vibench提供4个ContrastiveIDTask预设配置：

```python
# 配置预设映射
PRESET_TEMPLATES = {
    'contrastive': 'configs/id_contrastive/debug.yaml',          # 🐛 调试模式
    'contrastive_prod': 'configs/id_contrastive/production.yaml', # 🚀 生产模式
    'contrastive_ablation': 'configs/id_contrastive/ablation.yaml', # 🧪 消融研究
    'contrastive_cross': 'configs/id_contrastive/cross_dataset.yaml' # 🌍 跨域泛化
}

# 统一加载方式
from src.configs import load_config
config = load_config('contrastive')  # 自动加载debug.yaml
```

**配置场景矩阵**：

| 预设名称 | 用途 | 资源需求 | 执行时间 | 最佳场景 |
|---------|------|----------|----------|----------|
| `contrastive` | 🐛 快速验证 | CPU, <4GB | 2-5分钟 | 开发调试 |
| `contrastive_prod` | 🚀 完整训练 | GPU, 16GB+ | 2-24小时 | 正式实验 |
| `contrastive_ablation` | 🧪 参数研究 | GPU, 8GB+ | 1-12小时 | 论文实验 |
| `contrastive_cross` | 🌍 跨域测试 | Multi-GPU | 4-48小时 | 泛化验证 |

### 3. Factory模式集成架构

ContrastiveIDTask无缝集成PHM-Vibench的四大工厂：

```yaml
# 完整的Factory配置示例
data:
  factory_name: "id"              # → id_data_factory  
  dataset_name: "ID_dataset"      # → ID数据处理器
  window_size: 1024               # → 长信号窗口采样
  
model:
  type: "ISFM"                    # → model_factory/ISFM
  backbone: "B_08_PatchTST"       # → Transformer backbone
  task_head: "H_01_Linear_cla"    # → 分类头（对比学习中不使用）
  
task:
  name: "contrastive_id"          # → task_factory注册名
  temperature: 0.07               # → InfoNCE温度参数
  projection_dim: 128             # → 对比学习投影维度
  
trainer:
  accelerator: "auto"             # → trainer_factory/PyTorch Lightning
  devices: 1                      # → GPU设备配置
  precision: "16-mixed"           # → 混合精度训练
```

---

## ⚙️ 分步实施工作流

### 阶段一：数据准备（Steps 1-3）

#### Step 1: 准备Metadata文件

ContrastiveIDTask使用PHM-Vibench的三文件数据架构：

```bash
# 数据文件结构
data/
├── metadata_contrastive.xlsx    # 📋 数据索引（必需）
├── contrastive_data.h5         # 📊 信号数据（必需）  
└── corpus_contrastive.xlsx     # 📝 文本描述（可选）
```

**metadata文件格式**：
```excel
Id        | label | dataset | signal_length | sampling_rate | file_path
id_cwru_001 | 0   | CWRU   | 10240        | 12000        | data/cwru/001.mat
id_xjtu_001 | 1   | XJTU   | 20480        | 25600        | data/xjtu/001.mat
id_pu_001   | 2   | PU     | 8192         | 64000        | data/pu/001.mat
```

#### Step 2: 生成H5数据文件

使用PHM-Vibench的数据工厂工具：

```python
# 方式1: 使用data_factory工具生成
from src.data_factory.id_data_factory import generate_h5_from_metadata

# 从metadata生成H5文件
generate_h5_from_metadata(
    metadata_path="data/metadata_contrastive.xlsx",
    output_h5_path="data/contrastive_data.h5",
    signal_column="signal",  # H5中的信号数据列名
    progress_bar=True
)
```

```bash
# 方式2: 使用命令行工具
python scripts/prepare_data.py \
    --metadata data/metadata_contrastive.xlsx \
    --output data/contrastive_data.h5 \
    --format h5
```

#### Step 3: 验证数据完整性

```python
# 数据验证脚本
from src.data_factory import id_data_factory

# 加载数据验证
try:
    data_dict = id_data_factory.get_data("metadata_contrastive.xlsx")
    print(f"✅ 数据加载成功: {len(data_dict)} 个样本")
    
    # 检查数据质量
    sample_id = list(data_dict.keys())[0]
    sample_data = data_dict[sample_id]
    print(f"✅ 样本形状: {sample_data.shape}")
    print(f"✅ 数据类型: {sample_data.dtype}")
    
except Exception as e:
    print(f"❌ 数据验证失败: {e}")
```

### 阶段二：配置选择与定制（Steps 4-6）

#### Step 4: 选择合适的配置场景

根据你的需求选择配置：

```bash
# 🐛 快速验证 - 5分钟内完成
python main.py --pipeline Pipeline_ID --config contrastive

# 🚀 生产训练 - 完整实验
python main.py --pipeline Pipeline_ID --config contrastive_prod  

# 🧪 消融研究 - 参数对比
python main.py --pipeline Pipeline_ID --config contrastive_ablation

# 🌍 跨域泛化 - 多数据集
python main.py --pipeline Pipeline_ID --config contrastive_cross
```

#### Step 5: 配置参数定制

使用PHM-Vibench v5.0配置系统进行参数覆盖：

```python
# 方式1: 配置文件 + 命令行覆盖
python main.py \
    --pipeline Pipeline_ID \
    --config contrastive \
    --data.batch_size 32 \
    --task.temperature 0.1 \
    --trainer.epochs 10
```

```python
# 方式2: Python API配置覆盖
from src.configs import load_config

# 加载基础配置并覆盖参数
config = load_config('contrastive', {
    'data.batch_size': 32,
    'data.window_size': 2048,
    'task.temperature': 0.1,
    'task.projection_dim': 256,
    'trainer.epochs': 50
})

# 链式配置方式
config = load_config('contrastive').copy().update({
    'model.d_model': 512,
    'trainer.devices': 4,
    'trainer.strategy': 'ddp'
})
```

#### Step 6: 配置验证检查

```python
# 配置验证脚本
from src.configs import load_config, validate_config_completeness

config = load_config('contrastive')

# 检查必需字段
required_fields = [
    'data.factory_name', 'data.dataset_name', 'data.window_size',
    'model.type', 'task.name', 'task.temperature',
    'trainer.epochs', 'trainer.devices'
]

for field in required_fields:
    if not hasattr(config, field.replace('.', '.')):
        print(f"❌ 缺少必需字段: {field}")
    else:
        print(f"✅ {field}: {getattr(config, field.replace('.', '.'))}")
```

### 阶段三：训练执行（Steps 7-12）

#### Step 7: 基础训练启动

```bash
# 标准训练命令
python main.py \
    --pipeline Pipeline_ID \
    --config configs/id_contrastive/debug.yaml \
    --notes "ContrastiveID首次训练实验"
```

训练过程监控要点：
- **InfoNCE损失**：应该从约3.0逐渐降到1.5-2.0
- **对比准确率**：从随机水平（~0.25）提升到0.6+
- **内存使用**：确保GPU内存使用<80%

#### Step 8: 实时监控设置

```python
# 启动TensorBoard监控
import subprocess
subprocess.Popen(["tensorboard", "--logdir", "save/"])
print("🔍 TensorBoard已启动: http://localhost:6006")

# 实时日志监控
tail -f save/*/ContrastiveIDTask/*/log.txt
```

```python
# 自定义监控脚本
import time
import json
from pathlib import Path

def monitor_training(save_dir="save/", interval=30):
    """实时监控训练进程"""
    while True:
        # 查找最新实验
        latest_exp = sorted(Path(save_dir).glob("*/ContrastiveIDTask/*"))[-1]
        metrics_file = latest_exp / "metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
                print(f"📊 Epoch {metrics.get('epoch', 0)}: "
                      f"Loss={metrics.get('train_loss', 0):.4f}, "
                      f"Acc={metrics.get('contrastive_acc', 0):.4f}")
        
        time.sleep(interval)
```

#### Step 9: 训练中断与恢复

```bash
# 自动恢复最新checkpoint
python main.py \
    --pipeline Pipeline_ID \
    --config contrastive \
    --resume_from_checkpoint save/latest_run/checkpoints/last.ckpt

# 从特定checkpoint恢复
python main.py \
    --pipeline Pipeline_ID \
    --config contrastive \
    --resume_from_checkpoint save/metadata_6_1/ContrastiveIDTask/20241201_143052/checkpoints/epoch_10.ckpt
```

#### Step 10: 分布式训练扩展

```bash
# 单机多GPU训练
python main.py \
    --pipeline Pipeline_ID \
    --config contrastive_prod \
    --trainer.devices 4 \
    --trainer.strategy ddp

# 多机分布式训练
python main.py \
    --pipeline Pipeline_ID \
    --config contrastive_prod \
    --trainer.devices 8 \
    --trainer.num_nodes 2 \
    --trainer.strategy ddp
```

#### Step 11: 混合精度优化

```bash
# 启用FP16混合精度（节省50%内存）
python main.py \
    --pipeline Pipeline_ID \
    --config contrastive \
    --trainer.precision 16-mixed

# 启用BF16精度（更稳定）
python main.py \
    --pipeline Pipeline_ID \
    --config contrastive \
    --trainer.precision bf16-mixed
```

#### Step 12: 结果分析与保存

```python
# 加载训练结果
import torch
from pathlib import Path

# 查找最新实验结果
latest_run = sorted(Path("save").glob("*/ContrastiveIDTask/*"))[-1]
print(f"📁 最新实验: {latest_run}")

# 加载最佳模型
best_model = torch.load(latest_run / "checkpoints" / "best.ckpt")
print(f"🏆 最佳模型 - Epoch: {best_model['epoch']}, "
      f"Loss: {best_model['state_dict']['train_loss']:.4f}")

# 读取完整metrics
import json
with open(latest_run / "metrics.json") as f:
    metrics = json.load(f)
    print(f"📊 最终性能:")
    print(f"   训练损失: {metrics['train_loss']:.4f}")
    print(f"   对比准确率: {metrics['contrastive_acc']:.4f}")
    print(f"   训练时长: {metrics['training_time']:.2f}s")
```

### 阶段四：批量实验管理（Steps 13-16）

#### Step 13: 多数据集实验

```bash
# 使用实验管理脚本
python scripts/multi_dataset_experiments.py \
    --datasets CWRU,XJTU,PU,MFPT \
    --config contrastive \
    --parallel 2 \
    --output_dir experiments/multi_dataset/

# 跨数据集泛化实验
python scripts/multi_dataset_experiments.py \
    --source_datasets CWRU,XJTU \
    --target_datasets PU,MFPT \
    --config contrastive_cross \
    --mode cross_domain
```

#### Step 14: 参数消融研究

```bash
# 温度参数消融
python scripts/ablation_studies.py \
    --param temperature \
    --values 0.01,0.05,0.07,0.1,0.2,0.5 \
    --config contrastive_ablation \
    --output_dir ablation/temperature/

# 窗口大小消融  
python scripts/ablation_studies.py \
    --param window_size \
    --values 256,512,1024,2048,4096 \
    --config contrastive_ablation \
    --output_dir ablation/window_size/

# 多维度组合消融
python scripts/ablation_studies.py \
    --param_grid '{"temperature": [0.05, 0.07, 0.1], "window_size": [512, 1024, 2048]}' \
    --config contrastive_ablation \
    --output_dir ablation/grid_search/
```

#### Step 15: 性能基准测试

```bash
# 完整性能基准
python scripts/run_performance_benchmark.py \
    --config contrastive \
    --tests training,data_processing,model,scalability,hardware \
    --output_format html

# 快速基准（适合CI/CD）
python scripts/run_performance_benchmark.py \
    --config contrastive \
    --quick \
    --output_format json

# 特定类型测试
python scripts/run_performance_benchmark.py \
    --config contrastive_prod \
    --test scalability \
    --batch_sizes 16,32,64,128,256
```

#### Step 16: 自动化报告生成

```python
# 生成实验报告
from benchmarks.contrastive_performance_benchmark import ContrastivePerformanceBenchmark

# 创建benchmark实例
benchmark = ContrastivePerformanceBenchmark()

# 运行基准测试
results = benchmark.run_full_benchmark(
    config_path="configs/id_contrastive/production.yaml",
    output_dir="benchmark_results/"
)

# 生成HTML报告
benchmark.generate_report(
    results=results,
    output_path="reports/contrastive_benchmark.html",
    format="html"
)

print("📊 基准报告已生成: reports/contrastive_benchmark.html")
```

---

## 🔧 高级用法

### 自定义配置工作流（Steps 17-20）

#### Step 17: 创建项目专用配置

```yaml
# configs/project_contrastive.yaml
# 基于生产配置的项目定制
data:
  factory_name: "id"
  dataset_name: "ID_dataset"
  metadata_file: "metadata_project.xlsx"  # 项目专用数据
  window_size: 1024
  stride: 512
  num_windows: 4                          # 增加窗口数量
  
model:
  type: "ISFM"
  backbone: "B_04_Dlinear"               # 使用轻量级backbone
  d_model: 256
  
task:
  name: "contrastive_id"
  temperature: 0.08                       # 项目优化的温度
  projection_dim: 256                     # 更大的投影维度
  loss_weight: 1.0
  
trainer:
  epochs: 100                             # 更长训练
  devices: 2
  strategy: "ddp"
  gradient_clip_val: 1.0                  # 梯度裁剪
  
# 项目专用日志配置
logging:
  save_top_k: 5                          # 保存更多checkpoint
  monitor: "contrastive_acc"             # 监控对比准确率
  mode: "max"
```

#### Step 18: 参数网格搜索自动化

```python
# 高级网格搜索脚本
from src.configs import load_config
import itertools
import subprocess

def grid_search_contrastive():
    """对比学习参数网格搜索"""
    
    # 定义参数网格
    param_grid = {
        'temperature': [0.05, 0.07, 0.1, 0.15],
        'window_size': [512, 1024, 2048],
        'projection_dim': [128, 256, 512],
        'batch_size': [16, 32, 64]
    }
    
    # 生成所有参数组合
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    results = []
    for i, combo in enumerate(combinations):
        # 创建参数字典
        params = dict(zip(keys, combo))
        
        # 构建配置覆盖
        overrides = {
            f'task.temperature': params['temperature'],
            f'data.window_size': params['window_size'],
            f'task.projection_dim': params['projection_dim'],
            f'data.batch_size': params['batch_size']
        }
        
        # 运行实验
        cmd = [
            'python', 'main.py',
            '--pipeline', 'Pipeline_ID',
            '--config', 'contrastive',
            '--notes', f'grid_search_exp_{i}'
        ]
        
        # 添加参数覆盖
        for key, value in overrides.items():
            cmd.extend([f'--{key}', str(value)])
        
        print(f"🔄 运行实验 {i+1}/{len(combinations)}: {params}")
        subprocess.run(cmd)
        
        results.append(params)
    
    return results

# 执行网格搜索
if __name__ == "__main__":
    results = grid_search_contrastive()
    print(f"✅ 完成 {len(results)} 个实验组合")
```

#### Step 19: Pipeline链式组合

```bash
# 预训练 → 少样本学习工作流
python main.py \
    --pipeline Pipeline_02_pretrain_fewshot \
    --config_path configs/id_contrastive/production.yaml \
    --fs_config_path configs/demo/GFS/GFS_demo.yaml \
    --notes "对比预训练+少样本微调"

# 多任务预训练 → 微调工作流  
python main.py \
    --pipeline Pipeline_03_multitask_pretrain_finetune \
    --config_path configs/id_contrastive/production.yaml \
    --finetune_tasks classification,regression \
    --notes "多任务对比预训练"
```

#### Step 20: 模型导出与部署

```python
# 模型导出脚本
import torch
import torch.onnx
from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask
from src.configs import load_config

def export_contrastive_model(checkpoint_path, export_format="onnx"):
    """导出训练好的对比学习模型"""
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = load_config('contrastive')
    
    # 重建模型
    task = ContrastiveIDTask(config)
    task.load_state_dict(checkpoint['state_dict'])
    task.eval()
    
    # 创建示例输入
    batch_size = 1
    window_size = config.data.window_size
    num_channels = 2
    example_input = torch.randn(batch_size, window_size, num_channels)
    
    if export_format == "onnx":
        # 导出ONNX格式
        torch.onnx.export(
            task.model,
            example_input,
            "contrastive_model.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['signal'],
            output_names=['features'],
            dynamic_axes={
                'signal': {0: 'batch_size'},
                'features': {0: 'batch_size'}
            }
        )
        print("✅ ONNX模型已导出: contrastive_model.onnx")
        
    elif export_format == "torchscript":
        # 导出TorchScript格式
        traced_model = torch.jit.trace(task.model, example_input)
        traced_model.save("contrastive_model.pt")
        print("✅ TorchScript模型已导出: contrastive_model.pt")
        
    elif export_format == "state_dict":
        # 导出纯权重
        torch.save(task.model.state_dict(), "contrastive_weights.pth")
        print("✅ 模型权重已导出: contrastive_weights.pth")

# 使用示例
if __name__ == "__main__":
    checkpoint_path = "save/best_experiment/checkpoints/best.ckpt"
    export_contrastive_model(checkpoint_path, "onnx")
```

---

## 🐛 问题诊断流程

### 内存问题解决路径

```
💾 GPU内存不足?
├─ 🔄 减小batch_size (建议: 32→16→8→4)
│   └─ 修改: data.batch_size
├─ 🔄 减小window_size (建议: 2048→1024→512)  
│   └─ 修改: data.window_size
├─ 🔄 启用gradient_checkpointing
│   └─ 添加: trainer.gradient_checkpointing=True
├─ 🔄 使用混合精度
│   └─ 修改: trainer.precision="16-mixed"
├─ 🔄 减少num_windows
│   └─ 修改: data.num_windows (默认2)
└─ 🔄 关闭不必要的logging
    └─ 设置: logging.save_top_k=1
```

**诊断命令**：
```bash
# 检查GPU内存使用
nvidia-smi

# 监控内存使用趋势
watch -n 1 nvidia-smi

# 检查具体内存占用
python -c "
import torch
print(f'GPU内存: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
print(f'已分配: {torch.cuda.memory_allocated()/1e9:.1f}GB')
print(f'已缓存: {torch.cuda.memory_reserved()/1e9:.1f}GB')
"
```

### 收敛问题解决路径

```
📉 损失不下降?
├─ 🌡️ 检查温度参数 (建议: 0.05-0.1)
│   └─ task.temperature太高→难收敛, 太低→梯度消失
├─ 📈 调整学习率
│   ├─ 太高: 1e-3 → 1e-4 → 1e-5
│   └─ 太低: 1e-5 → 1e-4 → 1e-3
├─ 🎯 增加projection_dim (建议: 128→256→512)
│   └─ 更大的投影空间有助于特征分离
├─ 🔄 检查数据质量
│   ├─ 验证metadata完整性
│   ├─ 检查H5数据格式
│   └─ 确认窗口采样策略
└─ ⏰ 延长训练时间
    └─ 对比学习通常需要更多epoch才能收敛
```

**诊断脚本**：
```python
# 收敛诊断工具
def diagnose_convergence(log_file):
    """分析训练日志诊断收敛问题"""
    
    import re
    losses = []
    accuracies = []
    
    with open(log_file) as f:
        for line in f:
            # 提取损失值
            loss_match = re.search(r'train_loss=([0-9.]+)', line)
            if loss_match:
                losses.append(float(loss_match.group(1)))
            
            # 提取准确率
            acc_match = re.search(r'contrastive_acc=([0-9.]+)', line)
            if acc_match:
                accuracies.append(float(acc_match.group(1)))
    
    # 诊断分析
    if len(losses) > 10:
        recent_loss_trend = losses[-5:] 
        early_loss_trend = losses[:5]
        
        print("🔍 收敛诊断报告:")
        print(f"   初期损失: {early_loss_trend[0]:.4f}")
        print(f"   最新损失: {recent_loss_trend[-1]:.4f}")
        print(f"   损失下降: {early_loss_trend[0] - recent_loss_trend[-1]:.4f}")
        
        if recent_loss_trend[-1] > 2.5:
            print("⚠️  损失偏高，建议:")
            print("   - 降低温度参数至0.05-0.07")
            print("   - 增加projection_dim至256+")
            print("   - 检查学习率设置")
        
        if max(accuracies) < 0.4:
            print("⚠️  准确率偏低，建议:")
            print("   - 增加窗口数量(num_windows)")
            print("   - 调整窗口大小和stride")
            print("   - 验证数据预处理")

# 使用示例
diagnose_convergence("save/latest_run/log.txt")
```

### 数据加载问题解决路径

```
📊 数据加载错误?
├─ 📋 检查metadata格式
│   ├─ 必需列: Id, label, dataset
│   ├─ 数据类型: Id(str), label(int), dataset(str)
│   └─ 路径检查: file_path列是否存在
├─ 📦 验证H5文件
│   ├─ 检查文件完整性: h5py.is_hdf5()
│   ├─ 验证数据结构: 每个Id对应的数据形状
│   └─ 内存映射: 确保H5文件未损坏
├─ 🔧 配置检查
│   ├─ factory_name: 必须是"id"
│   ├─ dataset_name: 必须是"ID_dataset"  
│   └─ data_dir: 指向正确的数据目录
└─ 🔄 权限检查
    ├─ 文件读取权限
    └─ 目录访问权限
```

**数据验证脚本**：
```python
# 数据完整性检查
def validate_data_setup(metadata_path, h5_path=None):
    """验证数据设置的完整性"""
    
    import pandas as pd
    import h5py
    from pathlib import Path
    
    print("🔍 数据验证开始...")
    
    # 1. 检查metadata文件
    if not Path(metadata_path).exists():
        print(f"❌ Metadata文件不存在: {metadata_path}")
        return False
        
    try:
        df = pd.read_excel(metadata_path)
        print(f"✅ Metadata加载成功: {len(df)} 行")
        
        # 检查必需列
        required_cols = ['Id', 'label', 'dataset']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ 缺少必需列: {missing_cols}")
            return False
        print(f"✅ 必需列检查通过: {required_cols}")
        
    except Exception as e:
        print(f"❌ Metadata读取失败: {e}")
        return False
    
    # 2. 检查H5文件（如果提供）
    if h5_path and Path(h5_path).exists():
        try:
            with h5py.File(h5_path, 'r') as f:
                h5_ids = set(f.keys())
                metadata_ids = set(df['Id'].astype(str))
                
                missing_in_h5 = metadata_ids - h5_ids
                missing_in_metadata = h5_ids - metadata_ids
                
                if missing_in_h5:
                    print(f"⚠️  H5文件中缺少 {len(missing_in_h5)} 个ID")
                if missing_in_metadata:
                    print(f"⚠️  Metadata中缺少 {len(missing_in_metadata)} 个ID")
                
                print(f"✅ H5数据检查完成: {len(h5_ids)} 个样本")
                
        except Exception as e:
            print(f"❌ H5文件验证失败: {e}")
            return False
    
    print("✅ 数据验证通过!")
    return True

# 使用示例  
validate_data_setup("data/metadata_contrastive.xlsx", "data/contrastive_data.h5")
```

---

## 📊 性能优化检查清单

### 🚀 训练性能优化

- [ ] **批处理大小优化**
  ```python
  # 找到最优batch_size
  def find_optimal_batch_size():
      for batch_size in [16, 32, 64, 128, 256]:
          try:
              # 测试GPU内存使用
              config = load_config('contrastive', {'data.batch_size': batch_size})
              print(f"批处理 {batch_size}: 内存使用 {get_gpu_memory():.1f}GB")
          except RuntimeError as e:
              print(f"批处理 {batch_size}: 内存溢出")
              break
  ```

- [ ] **数据加载并行化**
  ```yaml
  # 配置文件优化
  data:
    num_workers: 8        # CPU核心数
    pin_memory: true      # 固定内存
    persistent_workers: true  # 持久化worker
  ```

- [ ] **混合精度训练**
  ```yaml
  trainer:
    precision: "16-mixed"  # 节省50%内存，加速2x
    # 或者 precision: "bf16-mixed"  # 更稳定
  ```

- [ ] **编译优化（PyTorch 2.0+）**
  ```python
  # 在task初始化后添加
  self.model = torch.compile(self.model, mode="reduce-overhead")
  ```

- [ ] **高效的数据采样**
  ```yaml
  data:
    window_sampling_strategy: "evenly_spaced"  # 比random更高效
    stride: 512  # 合理的stride设置
  ```

### 💾 内存优化策略

- [ ] **梯度检查点**
  ```yaml
  trainer:
    gradient_checkpointing: true  # 用时间换内存
  ```

- [ ] **小批量累积**
  ```yaml
  trainer:
    accumulate_grad_batches: 4  # 模拟大batch_size
  ```

- [ ] **定期清理缓存**
  ```python
  # 在训练循环中定期调用
  if step % 100 == 0:
      torch.cuda.empty_cache()
  ```

### 📈 I/O性能优化

- [ ] **SSD存储**：将数据集放在SSD上
- [ ] **内存映射**：使用H5文件的内存映射
- [ ] **预加载数据**：对于小数据集，预加载到内存
- [ ] **异步I/O**：使用异步数据加载

### 🔄 分布式训练优化

- [ ] **选择合适的策略**
  ```yaml
  trainer:
    strategy: "ddp"          # 单机多GPU
    # strategy: "deepspeed"   # 大模型优化
    # strategy: "fsdp"        # 全分片数据并行
  ```

- [ ] **网络优化**
  ```yaml
  trainer:
    sync_batchnorm: true     # 同步BatchNorm
    find_unused_parameters: false  # 提升性能
  ```

---

## 🎯 快速命令参考

### 📋 必备命令

```bash
# 🚀 核心训练命令
python main.py --pipeline Pipeline_ID --config contrastive              # 快速验证
python main.py --pipeline Pipeline_ID --config contrastive_prod         # 生产训练
python main.py --pipeline Pipeline_ID --config contrastive_ablation     # 消融研究
python main.py --pipeline Pipeline_ID --config contrastive_cross        # 跨域泛化

# 🧪 实验管理命令
python scripts/multi_dataset_experiments.py --quick                     # 批量实验
python scripts/ablation_studies.py --config contrastive_ablation       # 参数消融
python scripts/run_performance_benchmark.py --test all                  # 性能测试

# 📊 监控命令
tensorboard --logdir save/                                             # 可视化监控
tail -f save/*/ContrastiveIDTask/*/log.txt                             # 实时日志
nvidia-smi -l 1                                                        # GPU监控

# 🔧 工具命令
python -c "from src.configs import load_config; print('配置系统就绪')"     # 环境验证
python scripts/prepare_data.py --metadata data.xlsx --output data.h5   # 数据准备
```

### ⚙️ 高级配置命令

```bash
# 📝 参数覆盖
python main.py --pipeline Pipeline_ID --config contrastive \
  --data.batch_size 32 \
  --task.temperature 0.1 \
  --trainer.epochs 50 \
  --notes "参数调优实验"

# 🎯 特定GPU训练
CUDA_VISIBLE_DEVICES=0,1 python main.py \
  --pipeline Pipeline_ID \
  --config contrastive_prod \
  --trainer.devices 2

# 💾 内存优化训练
python main.py --pipeline Pipeline_ID --config contrastive \
  --trainer.precision "16-mixed" \
  --trainer.gradient_checkpointing true \
  --data.batch_size 16

# 🔄 恢复训练
python main.py --pipeline Pipeline_ID --config contrastive \
  --resume_from_checkpoint save/*/checkpoints/last.ckpt
```

### 📊 状态查询命令

```bash
# 查看最新实验
ls -t save/*/ContrastiveIDTask/* | head -5

# 检查实验状态
find save/ -name "metrics.json" -exec tail -1 {} \; -print

# 清理旧实验（保留最新10个）
find save/ -name "ContrastiveIDTask" -type d | \
  head -n -10 | xargs rm -rf

# GPU使用情况
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# 磁盘空间检查
du -sh save/ && df -h
```

---

## 📚 相关文档链接

- [📖 ContrastiveIDTask技术文档](../docs/contrastive_pretrain_guide.md) - 完整的API参考和技术细节
- [⚙️ PHM-Vibench配置系统](../src/configs/CLAUDE.md) - 配置系统使用指南
- [🏭 任务工厂文档](../src/task_factory/CLAUDE.md) - 任务开发和扩展指南
- [📊 性能基准报告](../benchmarks/README_performance_benchmark.md) - 性能测试和优化指南

---

## 📞 获取帮助

如果遇到问题，按以下顺序排查：

1. **检查本指南**的问题诊断流程
2. **查看日志文件**: `save/*/ContrastiveIDTask/*/log.txt`
3. **运行验证脚本**: 确认环境和数据设置
4. **查阅技术文档**: 深入了解实现细节
5. **社区支持**: 在GitHub Issues中寻求帮助

🎉 **祝你使用愉快！ContrastiveIDTask将为你的工业信号分析研究提供强大的对比学习预训练能力。**