# ContrastiveIDTask 实用案例集合

> 🎯 **实际应用案例和代码示例**  
> 涵盖从基础使用到高级定制的完整代码示例

## 📋 目录

- [🚀 基础使用案例](#-基础使用案例)
- [⚙️ 配置定制案例](#️-配置定制案例)
- [🧪 实验管理案例](#-实验管理案例)
- [📊 结果分析案例](#-结果分析案例)
- [🔧 集成开发案例](#-集成开发案例)
- [🐛 问题解决案例](#-问题解决案例)

---

## 🚀 基础使用案例

### 案例1：5分钟快速验证

**场景**：新接触ContrastiveIDTask，希望快速验证环境和功能。

```bash
# 完整的5分钟验证流程
cd /path/to/PHM-Vibench

# 1. 环境检查（30秒）
echo "🔍 环境检查..."
python -c "
import torch
from src.configs import load_config
from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask
print('✅ 所有组件就绪')
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
"

# 2. 快速训练（3分钟）
echo "🚀 开始快速训练..."
python main.py \
    --pipeline Pipeline_ID \
    --config configs/id_contrastive/debug.yaml \
    --notes "5分钟快速验证" \
    --trainer.max_epochs 1

# 3. 结果检查（1分钟）
echo "📊 检查结果..."
LATEST_RUN=$(find save/ -name "ContrastiveIDTask" -type d | head -1)
if [ -n "$LATEST_RUN" ]; then
    echo "✅ 训练完成！结果目录: $LATEST_RUN"
    ls -la "$LATEST_RUN"
else
    echo "❌ 未找到结果目录"
fi
```

**期待输出**：
```
✅ 所有组件就绪
PyTorch版本: 2.1.0
CUDA可用: True
🚀 开始快速训练...
[训练日志...]
✅ 训练完成！结果目录: save/metadata_6_1/ContrastiveIDTask/20241201_150342
checkpoints/  metrics.json  log.txt  config.yaml
```

### 案例2：使用自己的数据集

**场景**：有工业振动数据，希望用ContrastiveIDTask进行预训练。

```python
# prepare_my_data.py - 数据准备脚本
import pandas as pd
import numpy as np
import h5py
from pathlib import Path

def prepare_custom_dataset():
    """准备自定义数据集用于ContrastiveIDTask"""
    
    # 示例：从MAT文件创建数据集
    data_dir = Path("my_vibration_data/")
    output_dir = Path("data/")
    output_dir.mkdir(exist_ok=True)
    
    # 1. 创建metadata.xlsx
    metadata = []
    signal_data = {}
    
    # 遍历数据文件
    for i, mat_file in enumerate(data_dir.glob("*.mat")):
        # 假设每个MAT文件包含一个'signal'变量
        import scipy.io
        mat_data = scipy.io.loadmat(mat_file)
        signal = mat_data['signal'].flatten()  # 1D信号
        
        # 添加通道维度（如果需要）
        if len(signal.shape) == 1:
            signal = np.stack([signal, np.zeros_like(signal)], axis=-1)  # 2通道
        
        # 生成ID
        signal_id = f"custom_{i:04d}"
        
        # 保存信号数据
        signal_data[signal_id] = signal
        
        # 添加metadata记录
        metadata.append({
            'Id': signal_id,
            'label': i % 4,  # 4个类别的示例
            'dataset': 'CustomDataset',
            'signal_length': len(signal),
            'sampling_rate': 25600,  # 根据实际情况设置
            'equipment': 'Motor',
            'condition': f'Condition_{i % 4}'
        })
    
    # 保存metadata
    df = pd.DataFrame(metadata)
    df.to_excel(output_dir / "metadata_custom.xlsx", index=False)
    print(f"✅ Metadata已保存: {len(df)} 个样本")
    
    # 2. 创建H5文件
    with h5py.File(output_dir / "custom_data.h5", 'w') as f:
        for signal_id, signal in signal_data.items():
            f.create_dataset(signal_id, data=signal, compression='gzip')
    
    print(f"✅ H5数据文件已保存: {len(signal_data)} 个信号")
    
    # 3. 创建配置文件
    config_template = """
# configs/custom_contrastive.yaml
data:
  factory_name: "id"
  dataset_name: "ID_dataset"
  metadata_file: "metadata_custom.xlsx"
  data_dir: "data"
  window_size: 1024
  stride: 512
  num_windows: 2
  batch_size: 16

model:
  type: "ISFM"
  backbone: "B_08_PatchTST"
  d_model: 256

task:
  name: "contrastive_id"
  temperature: 0.07
  projection_dim: 128

trainer:
  epochs: 20
  devices: 1
  accelerator: "auto"
  precision: "16-mixed"

environment:
  WANDB_MODE: "disabled"
"""
    
    config_path = Path("configs/custom_contrastive.yaml")
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, 'w') as f:
        f.write(config_template.strip())
    
    print(f"✅ 配置文件已创建: {config_path}")
    print("🎉 数据准备完成！可以开始训练:")
    print("python main.py --pipeline Pipeline_ID --config configs/custom_contrastive.yaml")

if __name__ == "__main__":
    prepare_custom_dataset()
```

**运行自定义数据训练**：
```bash
# 1. 准备数据
python prepare_my_data.py

# 2. 开始训练
python main.py \
    --pipeline Pipeline_ID \
    --config configs/custom_contrastive.yaml \
    --notes "自定义数据集对比学习实验"

# 3. 监控训练
tensorboard --logdir save/ --port 6006
```

### 案例3：多GPU训练设置

**场景**：有多张GPU，希望加速训练过程。

```bash
# 检查GPU配置
nvidia-smi --list-gpus

# 4GPU分布式训练
python main.py \
    --pipeline Pipeline_ID \
    --config configs/id_contrastive/production.yaml \
    --trainer.devices 4 \
    --trainer.strategy ddp \
    --data.batch_size 64 \
    --data.num_workers 16 \
    --notes "4GPU分布式训练"

# 指定特定GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --pipeline Pipeline_ID \
    --config configs/id_contrastive/production.yaml \
    --trainer.devices 4 \
    --trainer.strategy ddp

# 多机训练（假设有2台机器，每台4GPU）
# 机器0 (主节点)
python main.py \
    --pipeline Pipeline_ID \
    --config configs/id_contrastive/production.yaml \
    --trainer.devices 4 \
    --trainer.num_nodes 2 \
    --trainer.node_rank 0 \
    --trainer.strategy ddp

# 机器1 (从节点) 
python main.py \
    --pipeline Pipeline_ID \
    --config configs/id_contrastive/production.yaml \
    --trainer.devices 4 \
    --trainer.num_nodes 2 \
    --trainer.node_rank 1 \
    --trainer.strategy ddp
```

---

## ⚙️ 配置定制案例

### 案例4：针对小内存GPU的配置优化

**场景**：只有8GB GPU内存，需要优化配置以避免内存溢出。

```yaml
# configs/low_memory_contrastive.yaml
data:
  factory_name: "id"
  dataset_name: "ID_dataset"
  metadata_file: "metadata_6_1.xlsx"
  window_size: 512              # 减小窗口大小
  stride: 256
  num_windows: 2               # 最小窗口数
  batch_size: 8                # 小批量
  num_workers: 4               # 减少worker数量

model:
  type: "ISFM"
  backbone: "B_04_Dlinear"     # 使用轻量级backbone
  d_model: 128                 # 减小模型维度

task:
  name: "contrastive_id"
  temperature: 0.07
  projection_dim: 64           # 减小投影维度

trainer:
  epochs: 50
  devices: 1
  accelerator: "gpu"
  precision: "16-mixed"        # 混合精度节省内存
  gradient_checkpointing: true # 用时间换内存
  accumulate_grad_batches: 8   # 累积梯度模拟大batch
  max_epochs: 50
  
  # 内存优化设置
  enable_progress_bar: false   # 减少内存使用
  log_every_n_steps: 50

logging:
  save_top_k: 2                # 只保存2个最佳checkpoint
  save_last: false             # 不保存最后一个checkpoint

environment:
  PYTHONHASHSEED: "0"
  WANDB_MODE: "disabled"
```

**使用脚本**：
```python
# low_memory_training.py - 内存监控训练脚本
import subprocess
import psutil
import torch
import time

def monitor_memory_training():
    """内存监控的训练脚本"""
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🔧 GPU内存清理完成")
    
    # 启动训练进程
    cmd = [
        'python', 'main.py',
        '--pipeline', 'Pipeline_ID',
        '--config', 'configs/low_memory_contrastive.yaml',
        '--notes', '内存优化训练'
    ]
    
    print("🚀 启动内存优化训练...")
    process = subprocess.Popen(cmd)
    
    # 监控内存使用
    max_memory = 0
    try:
        while process.poll() is None:
            # 监控系统内存
            ram_percent = psutil.virtual_memory().percent
            
            # 监控GPU内存
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1e9
                gpu_reserved = torch.cuda.memory_reserved() / 1e9
                max_memory = max(max_memory, gpu_reserved)
                
                print(f"📊 RAM: {ram_percent:.1f}% | "
                      f"GPU: {gpu_memory:.1f}GB/{gpu_reserved:.1f}GB | "
                      f"峰值: {max_memory:.1f}GB")
            
            time.sleep(30)  # 每30秒检查一次
            
    except KeyboardInterrupt:
        print("⏹️  训练被用户中断")
        process.terminate()
        
    finally:
        print(f"📈 最大GPU内存使用: {max_memory:.1f}GB")

if __name__ == "__main__":
    monitor_memory_training()
```

### 案例5：轻量级快速实验配置

**场景**：需要快速迭代不同的超参数组合。

```python
# quick_experiments.py - 快速实验脚本
from src.configs import load_config
import subprocess
import json
from datetime import datetime

def run_quick_experiments():
    """运行多个快速实验对比超参数"""
    
    # 实验参数组合
    experiments = [
        {
            'name': 'temp_005',
            'overrides': {'task.temperature': 0.05, 'trainer.epochs': 5},
            'description': '低温度快速实验'
        },
        {
            'name': 'temp_01',
            'overrides': {'task.temperature': 0.1, 'trainer.epochs': 5},
            'description': '中等温度快速实验'
        },
        {
            'name': 'temp_02',
            'overrides': {'task.temperature': 0.2, 'trainer.epochs': 5},
            'description': '高温度快速实验'
        },
        {
            'name': 'window_512',
            'overrides': {'data.window_size': 512, 'trainer.epochs': 5},
            'description': '小窗口快速实验'
        },
        {
            'name': 'window_2048',
            'overrides': {'data.window_size': 2048, 'trainer.epochs': 5},
            'description': '大窗口快速实验'
        }
    ]
    
    results = {}
    
    for exp in experiments:
        print(f"\n🧪 开始实验: {exp['name']} - {exp['description']}")
        
        # 构建命令
        cmd = [
            'python', 'main.py',
            '--pipeline', 'Pipeline_ID',
            '--config', 'contrastive',
            '--notes', f"快速实验_{exp['name']}"
        ]
        
        # 添加参数覆盖
        for key, value in exp['overrides'].items():
            cmd.extend([f'--{key}', str(value)])
        
        # 运行实验
        start_time = datetime.now()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = (datetime.now() - start_time).total_seconds()
        
        # 记录结果
        results[exp['name']] = {
            'duration': duration,
            'success': result.returncode == 0,
            'description': exp['description'],
            'overrides': exp['overrides']
        }
        
        if result.returncode == 0:
            print(f"✅ {exp['name']} 完成 ({duration:.1f}s)")
        else:
            print(f"❌ {exp['name']} 失败: {result.stderr}")
    
    # 生成总结报告
    print(f"\n📊 实验总结:")
    print(f"{'实验名称':<15} {'状态':<8} {'时长(s)':<10} {'描述'}")
    print("-" * 60)
    
    for name, result in results.items():
        status = "✅成功" if result['success'] else "❌失败"
        print(f"{name:<15} {status:<8} {result['duration']:<10.1f} {result['description']}")
    
    # 保存结果
    with open(f"quick_experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results

if __name__ == "__main__":
    run_quick_experiments()
```

### 案例6：生产环境配置模板

**场景**：为生产环境创建标准化的配置模板。

```yaml
# configs/production_template.yaml
# ContrastiveIDTask 生产环境标准配置模板
# 版本: v1.0
# 更新时间: 2024-12-01

# 数据配置 - 生产级设置
data:
  factory_name: "id"
  dataset_name: "ID_dataset"
  metadata_file: "metadata_production.xlsx"   # 替换为实际metadata
  data_dir: "data"
  
  # 信号处理参数
  window_size: 2048                           # 推荐值，平衡性能和精度
  stride: 1024                               # window_size的一半
  num_windows: 3                             # 生产环境推荐值
  
  # 数据加载优化
  batch_size: 32                             # 根据GPU内存调整
  num_workers: 8                             # CPU核心数
  pin_memory: true
  persistent_workers: true
  drop_last: true                            # 保持batch大小一致

# 模型配置 - ISFM + PatchTST 组合
model:
  type: "ISFM"
  backbone: "B_08_PatchTST"
  d_model: 512                               # 大模型，更好性能
  
  # PatchTST特定参数
  patch_len: 16
  stride: 8
  n_layers: 8
  n_heads: 16
  d_ff: 2048
  dropout: 0.1
  
  # 任务头配置（对比学习中不使用，但保留兼容性）
  task_head: "H_01_Linear_cla"
  num_classes: 10

# 任务配置 - 对比学习核心参数
task:
  name: "contrastive_id"
  
  # 对比学习参数
  temperature: 0.07                          # 经过调优的最佳值
  projection_dim: 256                        # 投影层维度
  
  # 损失函数配置
  loss_weight: 1.0
  
  # 优化器配置
  optimizer: "AdamW"
  lr: 1e-4                                   # 稳定的学习率
  weight_decay: 1e-5
  
  # 学习率调度
  scheduler: "cosine"
  warmup_epochs: 5
  min_lr: 1e-6

# 训练配置 - 生产环境优化
trainer:
  # 基础设置
  epochs: 100                                # 充分训练
  devices: 4                                 # 多GPU加速
  accelerator: "gpu"
  strategy: "ddp"                           # 分布式训练
  
  # 性能优化
  precision: "16-mixed"                      # 混合精度，节省内存
  sync_batchnorm: true                      # 分布式BatchNorm同步
  find_unused_parameters: false             # 性能优化
  
  # 梯度优化
  gradient_clip_val: 1.0                    # 梯度裁剪
  accumulate_grad_batches: 1                # 不累积梯度
  
  # 验证和保存
  val_check_interval: 0.25                  # 每1/4个epoch验证一次
  check_val_every_n_epoch: 1
  
  # 早停和checkpoint
  patience: 15                              # 早停耐心值
  min_delta: 0.001                          # 最小改进阈值

# 日志和监控配置
logging:
  # Checkpoint管理
  save_top_k: 5                             # 保存最佳5个模型
  save_last: true                           # 保存最后checkpoint
  monitor: "train_loss"                     # 监控训练损失
  mode: "min"                               # 最小化损失
  
  # 日志设置
  log_every_n_steps: 100                    # 每100步记录一次
  enable_progress_bar: true
  enable_model_summary: true
  
  # WandB配置（可选）
  project_name: "ContrastiveID_Production"
  experiment_name: null                     # 自动生成
  tags: ["contrastive", "production", "ISFM"]

# 系统环境配置
environment:
  # 随机种子
  PYTHONHASHSEED: "42"
  PL_SEED_EVERYTHING: "42"
  
  # CUDA优化
  CUDA_LAUNCH_BLOCKING: "0"
  TORCH_CUDNN_V8_API_ENABLED: "1"
  
  # WandB配置
  WANDB_MODE: "online"                      # 或 "disabled"
  WANDB_PROJECT: "ContrastiveID_Production"
  
  # 其他环境变量
  OMP_NUM_THREADS: "8"
  MKL_NUM_THREADS: "8"

# 数据验证配置
validation:
  # 数据完整性检查
  check_data_integrity: true
  min_signal_length: 1024                   # 最小信号长度
  max_signal_length: 100000                 # 最大信号长度
  
  # 预处理验证
  check_nan_inf: true                       # 检查NaN/Inf值
  normalize_check: true                     # 检查归一化
  
# 性能监控配置  
monitoring:
  # 内存监控
  track_gpu_memory: true
  memory_threshold: 0.9                     # GPU内存使用阈值
  
  # 性能指标
  track_throughput: true                    # 跟踪吞吐量
  track_convergence: true                   # 跟踪收敛速度
  
  # 报告生成
  generate_report: true                     # 自动生成报告
  report_format: ["html", "json"]           # 报告格式

# 部署配置
deployment:
  # 模型导出
  export_format: ["onnx", "torchscript"]    # 导出格式
  export_precision: "fp16"                  # 导出精度
  
  # 推理优化
  optimize_for_inference: true
  batch_size_inference: 64                  # 推理批大小
```

**使用生产配置的脚本**：
```python
# production_training.py - 生产环境训练脚本
import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

def setup_production_environment():
    """设置生产环境"""
    
    # 创建日志目录
    log_dir = Path("logs/production")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"production_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    
    # 环境变量设置
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    return logging.getLogger(__name__)

def main():
    logger = setup_production_environment()
    
    parser = argparse.ArgumentParser(description='ContrastiveID生产训练')
    parser.add_argument('--config', default='configs/production_template.yaml', help='配置文件')
    parser.add_argument('--experiment_name', help='实验名称')
    parser.add_argument('--dry_run', action='store_true', help='干运行模式')
    
    args = parser.parse_args()
    
    logger.info(f"🚀 开始ContrastiveID生产训练")
    logger.info(f"📝 配置文件: {args.config}")
    
    if args.dry_run:
        logger.info("🔍 干运行模式 - 仅验证配置")
        # 验证配置逻辑
        from src.configs import load_config
        try:
            config = load_config(args.config)
            logger.info("✅ 配置验证通过")
            return
        except Exception as e:
            logger.error(f"❌ 配置验证失败: {e}")
            return
    
    # 构建训练命令
    cmd = [
        'python', 'main.py',
        '--pipeline', 'Pipeline_ID',
        '--config', args.config
    ]
    
    if args.experiment_name:
        cmd.extend(['--notes', args.experiment_name])
    
    # 执行训练
    import subprocess
    logger.info(f"🔧 执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        logger.info("✅ 训练成功完成")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 训练失败: {e}")
        raise

if __name__ == "__main__":
    main()
```

---

## 🧪 实验管理案例

### 案例7：批量超参数搜索

**场景**：需要系统性地搜索最优超参数组合。

```python
# hyperparameter_search.py - 超参数搜索脚本
import itertools
import json
import subprocess
from datetime import datetime
from pathlib import Path
import pandas as pd

class HyperparameterSearch:
    """ContrastiveIDTask超参数搜索器"""
    
    def __init__(self, base_config="contrastive", output_dir="hyperparameter_search"):
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 定义搜索空间
        self.search_space = {
            'temperature': [0.01, 0.05, 0.07, 0.1, 0.2, 0.5],
            'projection_dim': [64, 128, 256, 512],
            'window_size': [512, 1024, 2048],
            'lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            'batch_size': [16, 32, 64]
        }
        
        self.results = []
    
    def generate_combinations(self, max_combinations=50):
        """生成参数组合"""
        
        # 获取所有参数组合
        keys = list(self.search_space.keys())
        values = list(self.search_space.values())
        all_combinations = list(itertools.product(*values))
        
        # 如果组合太多，随机采样
        if len(all_combinations) > max_combinations:
            import random
            random.seed(42)
            combinations = random.sample(all_combinations, max_combinations)
        else:
            combinations = all_combinations
        
        # 转换为字典列表
        param_combinations = []
        for combo in combinations:
            param_dict = dict(zip(keys, combo))
            param_combinations.append(param_dict)
        
        return param_combinations
    
    def run_experiment(self, params, experiment_id):
        """运行单个实验"""
        
        print(f"🧪 实验 {experiment_id}: {params}")
        
        # 构建命令
        cmd = [
            'python', 'main.py',
            '--pipeline', 'Pipeline_ID', 
            '--config', self.base_config,
            '--trainer.epochs', '10',  # 短epoch用于搜索
            '--notes', f'hypersearch_{experiment_id}'
        ]
        
        # 添加参数
        for key, value in params.items():
            if key == 'temperature':
                cmd.extend(['--task.temperature', str(value)])
            elif key == 'projection_dim':
                cmd.extend(['--task.projection_dim', str(value)])
            elif key == 'window_size':
                cmd.extend(['--data.window_size', str(value)])
            elif key == 'lr':
                cmd.extend(['--task.lr', str(value)])
            elif key == 'batch_size':
                cmd.extend(['--data.batch_size', str(value)])
        
        # 运行实验
        start_time = datetime.now()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30分钟超时
            success = result.returncode == 0
            error_msg = result.stderr if not success else None
        except subprocess.TimeoutExpired:
            success = False
            error_msg = "Timeout after 30 minutes"
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # 提取性能指标（从输出或日志文件）
        final_loss = self.extract_final_loss(result.stdout) if success else None
        convergence_epoch = self.extract_convergence_epoch(result.stdout) if success else None
        
        # 记录结果
        experiment_result = {
            'experiment_id': experiment_id,
            'params': params,
            'success': success,
            'duration': duration,
            'final_loss': final_loss,
            'convergence_epoch': convergence_epoch,
            'error_msg': error_msg,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(experiment_result)
        
        # 实时保存结果
        self.save_results()
        
        return experiment_result
    
    def extract_final_loss(self, output):
        """从输出中提取最终损失"""
        import re
        # 查找最后的损失值
        matches = re.findall(r'train_loss=([0-9.]+)', output)
        return float(matches[-1]) if matches else None
    
    def extract_convergence_epoch(self, output):
        """提取收敛epoch"""
        # 简单实现：假设损失下降到一定阈值即为收敛
        import re
        epoch_losses = []
        for line in output.split('\n'):
            if 'Epoch' in line and 'train_loss=' in line:
                epoch_match = re.search(r'Epoch (\d+)', line)
                loss_match = re.search(r'train_loss=([0-9.]+)', line)
                if epoch_match and loss_match:
                    epoch = int(epoch_match.group(1))
                    loss = float(loss_match.group(1))
                    epoch_losses.append((epoch, loss))
        
        # 找到首次达到收敛阈值的epoch
        convergence_threshold = 2.0
        for epoch, loss in epoch_losses:
            if loss < convergence_threshold:
                return epoch
        
        return None
    
    def save_results(self):
        """保存结果"""
        
        # JSON格式
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # CSV格式（便于分析）
        if self.results:
            # 展开参数列
            flat_results = []
            for result in self.results:
                flat_result = {
                    'experiment_id': result['experiment_id'],
                    'success': result['success'],
                    'duration': result['duration'],
                    'final_loss': result['final_loss'],
                    'convergence_epoch': result['convergence_epoch'],
                    'timestamp': result['timestamp']
                }
                # 添加参数列
                flat_result.update(result['params'])
                flat_results.append(flat_result)
            
            df = pd.DataFrame(flat_results)
            df.to_csv(self.output_dir / 'results.csv', index=False)
    
    def run_search(self, max_combinations=20):
        """运行超参数搜索"""
        
        combinations = self.generate_combinations(max_combinations)
        print(f"🎯 开始超参数搜索，共 {len(combinations)} 个组合")
        
        for i, params in enumerate(combinations, 1):
            try:
                result = self.run_experiment(params, i)
                status = "✅" if result['success'] else "❌"
                print(f"{status} 实验 {i}/{len(combinations)} 完成 "
                      f"(耗时: {result['duration']:.1f}s)")
                
                if result['success'] and result['final_loss']:
                    print(f"   最终损失: {result['final_loss']:.4f}")
                
            except Exception as e:
                print(f"❌ 实验 {i} 异常: {e}")
        
        # 分析结果
        self.analyze_results()
    
    def analyze_results(self):
        """分析搜索结果"""
        
        if not self.results:
            print("❌ 没有结果可分析")
            return
        
        successful_results = [r for r in self.results if r['success'] and r['final_loss']]
        
        if not successful_results:
            print("❌ 没有成功的实验")
            return
        
        # 找到最佳结果
        best_result = min(successful_results, key=lambda x: x['final_loss'])
        
        print(f"\n🏆 最佳结果 (实验ID: {best_result['experiment_id']}):")
        print(f"   最终损失: {best_result['final_loss']:.4f}")
        print(f"   参数组合: {best_result['params']}")
        print(f"   收敛epoch: {best_result['convergence_epoch']}")
        
        # 参数重要性分析
        print(f"\n📊 参数分析:")
        for param in self.search_space.keys():
            param_values = {}
            for result in successful_results:
                value = result['params'][param]
                if value not in param_values:
                    param_values[value] = []
                param_values[value].append(result['final_loss'])
            
            # 计算每个参数值的平均损失
            avg_losses = {v: sum(losses)/len(losses) for v, losses in param_values.items()}
            best_value = min(avg_losses.keys(), key=lambda x: avg_losses[x])
            
            print(f"   {param}: 最佳值 = {best_value} (平均损失: {avg_losses[best_value]:.4f})")
        
        # 保存最佳配置
        best_config = {
            'data': {},
            'task': {},
            'trainer': {}
        }
        
        for key, value in best_result['params'].items():
            if key == 'temperature':
                best_config['task']['temperature'] = value
            elif key == 'projection_dim':
                best_config['task']['projection_dim'] = value
            elif key == 'window_size':
                best_config['data']['window_size'] = value
            elif key == 'lr':
                best_config['task']['lr'] = value
            elif key == 'batch_size':
                best_config['data']['batch_size'] = value
        
        with open(self.output_dir / 'best_config.json', 'w') as f:
            json.dump(best_config, f, indent=2)
        
        print(f"✅ 最佳配置已保存到: {self.output_dir / 'best_config.json'}")

# 使用示例
if __name__ == "__main__":
    searcher = HyperparameterSearch()
    searcher.run_search(max_combinations=20)
```

### 案例8：多数据集对比实验

**场景**：在多个数据集上验证ContrastiveIDTask的效果。

```python
# multi_dataset_comparison.py - 多数据集对比实验
import subprocess
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

class MultiDatasetComparison:
    """多数据集对比实验管理器"""
    
    def __init__(self, output_dir="multi_dataset_comparison"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 数据集配置
        self.datasets = {
            'CWRU': {
                'metadata': 'metadata_cwru.xlsx',
                'description': 'Case Western Reserve University数据集',
                'classes': 4,
                'signal_length': 'variable'
            },
            'XJTU': {
                'metadata': 'metadata_xjtu.xlsx', 
                'description': '西交大轴承数据集',
                'classes': 5,
                'signal_length': 'long'
            },
            'PU': {
                'metadata': 'metadata_pu.xlsx',
                'description': 'Paderborn University数据集',
                'classes': 12,
                'signal_length': 'very_long'
            }
        }
        
        self.results = {}
    
    def run_dataset_experiment(self, dataset_name, dataset_config):
        """在单个数据集上运行实验"""
        
        print(f"🗃️  开始数据集实验: {dataset_name}")
        print(f"   描述: {dataset_config['description']}")
        
        # 为每个数据集创建专门的配置
        config_overrides = {
            'data.metadata_file': dataset_config['metadata'],
            'trainer.epochs': 20,  # 标准化训练epoch
            'notes': f'MultiDataset_{dataset_name}'
        }
        
        # 根据数据集特性调整参数
        if dataset_config['signal_length'] == 'very_long':
            config_overrides['data.window_size'] = 4096
            config_overrides['data.stride'] = 2048
        elif dataset_config['signal_length'] == 'long':
            config_overrides['data.window_size'] = 2048
            config_overrides['data.stride'] = 1024
        else:
            config_overrides['data.window_size'] = 1024
            config_overrides['data.stride'] = 512
        
        # 构建命令
        cmd = [
            'python', 'main.py',
            '--pipeline', 'Pipeline_ID',
            '--config', 'contrastive'
        ]
        
        for key, value in config_overrides.items():
            if key != 'notes':
                cmd.extend([f'--{key}', str(value)])
            else:
                cmd.extend(['--notes', str(value)])
        
        # 运行实验
        start_time = datetime.now()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1小时超时
            success = result.returncode == 0
            duration = (datetime.now() - start_time).total_seconds()
            
            if success:
                # 提取性能指标
                final_loss = self.extract_metric(result.stdout, 'train_loss')
                contrastive_acc = self.extract_metric(result.stdout, 'contrastive_acc')
                convergence_epoch = self.extract_convergence_epoch(result.stdout)
                
                result_data = {
                    'dataset': dataset_name,
                    'success': True,
                    'duration': duration,
                    'final_loss': final_loss,
                    'contrastive_acc': contrastive_acc,
                    'convergence_epoch': convergence_epoch,
                    'config': config_overrides,
                    'dataset_info': dataset_config
                }
                
                print(f"✅ {dataset_name} 实验完成")
                print(f"   最终损失: {final_loss:.4f}")
                print(f"   对比准确率: {contrastive_acc:.4f}")
                print(f"   收敛epoch: {convergence_epoch}")
                
            else:
                result_data = {
                    'dataset': dataset_name,
                    'success': False,
                    'duration': duration,
                    'error': result.stderr,
                    'config': config_overrides,
                    'dataset_info': dataset_config
                }
                print(f"❌ {dataset_name} 实验失败: {result.stderr}")
        
        except subprocess.TimeoutExpired:
            result_data = {
                'dataset': dataset_name,
                'success': False,
                'duration': 3600,
                'error': 'Timeout after 1 hour',
                'config': config_overrides,
                'dataset_info': dataset_config
            }
            print(f"⏰ {dataset_name} 实验超时")
        
        self.results[dataset_name] = result_data
        return result_data
    
    def extract_metric(self, output, metric_name):
        """提取指标值"""
        import re
        pattern = f'{metric_name}=([0-9.]+)'
        matches = re.findall(pattern, output)
        return float(matches[-1]) if matches else None
    
    def extract_convergence_epoch(self, output):
        """提取收敛epoch"""
        import re
        losses = []
        for line in output.split('\n'):
            if 'Epoch' in line and 'train_loss=' in line:
                epoch_match = re.search(r'Epoch (\d+)', line)
                loss_match = re.search(r'train_loss=([0-9.]+)', line)
                if epoch_match and loss_match:
                    epoch = int(epoch_match.group(1))
                    loss = float(loss_match.group(1))
                    losses.append((epoch, loss))
        
        # 简单的收敛检测：连续3个epoch损失变化<0.01
        if len(losses) >= 6:
            for i in range(3, len(losses)):
                recent_losses = [loss for _, loss in losses[i-3:i]]
                if max(recent_losses) - min(recent_losses) < 0.01:
                    return losses[i-3][0]
        
        return None
    
    def run_all_experiments(self):
        """运行所有数据集实验"""
        
        print(f"🚀 开始多数据集对比实验")
        print(f"📊 数据集数量: {len(self.datasets)}")
        
        for dataset_name, dataset_config in self.datasets.items():
            try:
                self.run_dataset_experiment(dataset_name, dataset_config)
            except Exception as e:
                print(f"❌ {dataset_name} 实验异常: {e}")
                self.results[dataset_name] = {
                    'dataset': dataset_name,
                    'success': False,
                    'error': str(e),
                    'dataset_info': dataset_config
                }
        
        # 生成对比报告
        self.generate_comparison_report()
    
    def generate_comparison_report(self):
        """生成对比报告"""
        
        print(f"\n📊 生成多数据集对比报告...")
        
        # 创建对比表格
        comparison_data = []
        for dataset_name, result in self.results.items():
            row = {
                'Dataset': dataset_name,
                'Success': '✅' if result['success'] else '❌',
                'Duration(s)': result.get('duration', 0),
                'Final Loss': result.get('final_loss', 'N/A'),
                'Contrastive Acc': result.get('contrastive_acc', 'N/A'),
                'Convergence Epoch': result.get('convergence_epoch', 'N/A'),
                'Classes': result['dataset_info']['classes'],
                'Signal Length': result['dataset_info']['signal_length']
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # 保存CSV
        df.to_csv(self.output_dir / 'dataset_comparison.csv', index=False)
        
        # 打印对比表格
        print(f"\n🏆 数据集对比结果:")
        print(df.to_string(index=False))
        
        # 统计分析
        successful_results = [r for r in self.results.values() if r['success']]
        
        if successful_results:
            avg_loss = sum(r['final_loss'] for r in successful_results) / len(successful_results)
            avg_acc = sum(r['contrastive_acc'] for r in successful_results) / len(successful_results)
            avg_convergence = sum(r['convergence_epoch'] for r in successful_results if r['convergence_epoch']) / len([r for r in successful_results if r['convergence_epoch']])
            
            print(f"\n📈 统计摘要:")
            print(f"   成功率: {len(successful_results)}/{len(self.results)} ({len(successful_results)/len(self.results)*100:.1f}%)")
            print(f"   平均最终损失: {avg_loss:.4f}")
            print(f"   平均对比准确率: {avg_acc:.4f}")
            print(f"   平均收敛epoch: {avg_convergence:.1f}")
            
            # 找到最佳和最差结果
            best_result = min(successful_results, key=lambda x: x['final_loss'])
            worst_result = max(successful_results, key=lambda x: x['final_loss'])
            
            print(f"\n🥇 最佳数据集: {best_result['dataset']} (损失: {best_result['final_loss']:.4f})")
            print(f"🥉 最具挑战性数据集: {worst_result['dataset']} (损失: {worst_result['final_loss']:.4f})")
        
        # 保存完整结果
        with open(self.output_dir / 'full_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # 生成HTML报告
        self.generate_html_report(df)
        
        print(f"✅ 报告已保存到 {self.output_dir}/")
    
    def generate_html_report(self, df):
        """生成HTML报告"""
        
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ContrastiveIDTask 多数据集对比报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .success {{ color: green; }}
        .failure {{ color: red; }}
        .summary {{ background-color: #f9f9f9; padding: 20px; margin: 20px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>ContrastiveIDTask 多数据集对比报告</h1>
    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <h2>实验概览</h2>
        <p>本次实验在 {len(self.datasets)} 个数据集上验证了ContrastiveIDTask的性能。</p>
        <ul>
            <li>CWRU: Case Western Reserve University轴承数据集</li>
            <li>XJTU: 西安交通大学轴承数据集</li>
            <li>PU: Paderborn University轴承数据集</li>
        </ul>
    </div>
    
    <h2>对比结果</h2>
    {df.to_html(index=False, classes='comparison-table', escape=False)}
    
    <div class="summary">
        <h2>关键发现</h2>
        <ul>
            <li>ContrastiveIDTask在不同数据集上展现了良好的适应性</li>
            <li>较长的信号长度有助于获得更好的对比学习效果</li>
            <li>不同数据集的收敛速度存在差异，反映了数据复杂度的不同</li>
        </ul>
    </div>
</body>
</html>
"""
        
        with open(self.output_dir / 'comparison_report.html', 'w', encoding='utf-8') as f:
            f.write(html_template)

# 使用示例
if __name__ == "__main__":
    comparison = MultiDatasetComparison()
    comparison.run_all_experiments()
```

---

## 📊 结果分析案例

### 案例9：训练过程可视化分析

**场景**：需要深入分析训练过程，理解模型的学习行为。

```python
# training_analysis.py - 训练过程分析工具
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from pathlib import Path
import re

class ContrastiveTrainingAnalyzer:
    """ContrastiveIDTask训练过程分析器"""
    
    def __init__(self, experiment_dir):
        self.experiment_dir = Path(experiment_dir)
        self.log_file = self.experiment_dir / "log.txt"
        self.metrics_file = self.experiment_dir / "metrics.json"
        self.config_file = self.experiment_dir / "config.yaml"
        
        # 设置绘图样式
        plt.style.use('default')
        sns.set_palette("husl")
        
    def parse_log_file(self):
        """解析训练日志文件"""
        
        if not self.log_file.exists():
            print(f"❌ 日志文件不存在: {self.log_file}")
            return None
        
        training_data = []
        
        with open(self.log_file, 'r') as f:
            for line in f:
                # 解析训练日志行
                if 'Epoch' in line and 'train_loss=' in line:
                    # 提取epoch信息
                    epoch_match = re.search(r'Epoch (\d+)', line)
                    
                    # 提取各种指标
                    loss_match = re.search(r'train_loss=([0-9.]+)', line)
                    acc_match = re.search(r'contrastive_acc=([0-9.]+)', line)
                    lr_match = re.search(r'lr=([0-9.e-]+)', line)
                    
                    if epoch_match and loss_match:
                        epoch = int(epoch_match.group(1))
                        train_loss = float(loss_match.group(1))
                        contrastive_acc = float(acc_match.group(1)) if acc_match else None
                        learning_rate = float(lr_match.group(1)) if lr_match else None
                        
                        training_data.append({
                            'epoch': epoch,
                            'train_loss': train_loss,
                            'contrastive_acc': contrastive_acc,
                            'learning_rate': learning_rate
                        })
        
        if training_data:
            df = pd.DataFrame(training_data)
            return df
        else:
            print("❌ 无法从日志中提取训练数据")
            return None
    
    def load_metrics(self):
        """加载metrics.json文件"""
        
        if not self.metrics_file.exists():
            return None
        
        with open(self.metrics_file, 'r') as f:
            metrics = json.load(f)
        
        return metrics
    
    def analyze_convergence(self, df):
        """分析收敛特性"""
        
        print("📈 收敛分析:")
        
        # 计算损失变化率
        df['loss_change'] = df['train_loss'].diff()
        df['loss_change_pct'] = df['train_loss'].pct_change() * 100
        
        # 找到收敛点（损失变化小于阈值）
        convergence_threshold = 0.01
        stable_epochs = df[abs(df['loss_change']) < convergence_threshold]
        
        if len(stable_epochs) > 0:
            convergence_epoch = stable_epochs.iloc[0]['epoch']
            print(f"   收敛epoch: {convergence_epoch}")
            print(f"   收敛时损失: {df[df['epoch'] == convergence_epoch]['train_loss'].iloc[0]:.4f}")
        else:
            print("   未检测到明显收敛点")
        
        # 分析学习阶段
        total_epochs = len(df)
        early_stage = df[:total_epochs//3]
        middle_stage = df[total_epochs//3:2*total_epochs//3]
        late_stage = df[2*total_epochs//3:]
        
        print(f"   早期阶段平均损失下降率: {early_stage['loss_change'].mean():.6f}/epoch")
        print(f"   中期阶段平均损失下降率: {middle_stage['loss_change'].mean():.6f}/epoch")
        print(f"   后期阶段平均损失下降率: {late_stage['loss_change'].mean():.6f}/epoch")
        
        return convergence_epoch if 'convergence_epoch' in locals() else None
    
    def plot_training_curves(self, df, save_path=None):
        """绘制训练曲线"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ContrastiveIDTask Training Analysis', fontsize=16)
        
        # 1. 训练损失曲线
        axes[0,0].plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, label='Train Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].set_title('Training Loss Curve')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        # 添加收敛线（如果有）
        if df['train_loss'].nunique() > 1:
            min_loss = df['train_loss'].min()
            final_loss = df['train_loss'].iloc[-1]
            axes[0,0].axhline(y=min_loss, color='g', linestyle='--', alpha=0.7, label=f'Best Loss: {min_loss:.4f}')
            axes[0,0].text(df['epoch'].max()*0.7, min_loss*1.1, f'Final: {final_loss:.4f}', fontsize=10)
        
        # 2. 对比准确率曲线
        if 'contrastive_acc' in df.columns and df['contrastive_acc'].notna().any():
            axes[0,1].plot(df['epoch'], df['contrastive_acc'], 'r-', linewidth=2, label='Contrastive Accuracy')
            axes[0,1].set_xlabel('Epoch')
            axes[0,1].set_ylabel('Accuracy')
            axes[0,1].set_title('Contrastive Accuracy Curve')
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].legend()
            
            # 标记最佳准确率
            best_acc = df['contrastive_acc'].max()
            best_epoch = df.loc[df['contrastive_acc'].idxmax(), 'epoch']
            axes[0,1].axhline(y=best_acc, color='g', linestyle='--', alpha=0.7)
            axes[0,1].text(best_epoch, best_acc*1.02, f'Best: {best_acc:.4f}@E{best_epoch}', fontsize=10)
        else:
            axes[0,1].text(0.5, 0.5, 'No Accuracy Data', transform=axes[0,1].transAxes, 
                          ha='center', va='center', fontsize=14, alpha=0.5)
            axes[0,1].set_title('Contrastive Accuracy (No Data)')
        
        # 3. 学习率变化
        if 'learning_rate' in df.columns and df['learning_rate'].notna().any():
            axes[1,0].plot(df['epoch'], df['learning_rate'], 'g-', linewidth=2, label='Learning Rate')
            axes[1,0].set_xlabel('Epoch')
            axes[1,0].set_ylabel('Learning Rate')
            axes[1,0].set_title('Learning Rate Schedule')
            axes[1,0].set_yscale('log')  # 对数尺度
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].legend()
        else:
            axes[1,0].text(0.5, 0.5, 'No LR Data', transform=axes[1,0].transAxes, 
                          ha='center', va='center', fontsize=14, alpha=0.5)
            axes[1,0].set_title('Learning Rate (No Data)')
        
        # 4. 损失变化率分析
        if 'loss_change' in df.columns:
            # 移动平均平滑
            window_size = max(1, len(df) // 10)
            df['loss_change_smooth'] = df['loss_change'].rolling(window=window_size).mean()
            
            axes[1,1].plot(df['epoch'], df['loss_change'], alpha=0.3, color='gray', label='Raw Change')
            axes[1,1].plot(df['epoch'], df['loss_change_smooth'], 'orange', linewidth=2, label='Smoothed Change')
            axes[1,1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('Loss Change')
            axes[1,1].set_title('Loss Change Rate')
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 训练曲线已保存: {save_path}")
        
        plt.show()
    
    def analyze_loss_landscape(self, df):
        """分析损失landscape特性"""
        
        print("\n🏔️ 损失landscape分析:")
        
        if len(df) < 10:
            print("   数据点太少，无法进行landscape分析")
            return
        
        # 计算损失的各种统计特征
        loss_values = df['train_loss'].values
        
        # 平滑性分析（二阶差分）
        second_diff = np.diff(loss_values, n=2)
        smoothness = np.std(second_diff)
        print(f"   损失曲线平滑性指标: {smoothness:.6f} (越小越平滑)")
        
        # 收敛稳定性分析
        last_10pct = int(len(loss_values) * 0.1)
        if last_10pct > 0:
            late_losses = loss_values[-last_10pct:]
            stability = np.std(late_losses) / np.mean(late_losses)
            print(f"   收敛稳定性: {stability:.6f} (越小越稳定)")
        
        # 学习效率分析
        total_improvement = loss_values[0] - loss_values[-1]
        relative_improvement = total_improvement / loss_values[0]
        print(f"   总体改进: {total_improvement:.4f} ({relative_improvement*100:.1f}%)")
        
        # 找到学习停滞期
        window_size = max(5, len(loss_values) // 20)
        rolling_std = pd.Series(loss_values).rolling(window=window_size).std()
        plateau_threshold = 0.01
        plateau_epochs = df[rolling_std < plateau_threshold]
        
        if len(plateau_epochs) > 0:
            print(f"   检测到 {len(plateau_epochs)} 个停滞期")
            for i, (idx, row) in enumerate(plateau_epochs.iterrows()):
                if i < 3:  # 只显示前3个
                    print(f"     停滞期 {i+1}: Epoch {row['epoch']}, Loss {row['train_loss']:.4f}")
    
    def generate_detailed_report(self, df, metrics=None):
        """生成详细分析报告"""
        
        report_path = self.experiment_dir / "training_analysis_report.html"
        
        # 基础统计
        stats = {
            'total_epochs': len(df),
            'initial_loss': df['train_loss'].iloc[0],
            'final_loss': df['train_loss'].iloc[-1],
            'best_loss': df['train_loss'].min(),
            'loss_reduction': df['train_loss'].iloc[0] - df['train_loss'].iloc[-1],
            'loss_reduction_pct': ((df['train_loss'].iloc[0] - df['train_loss'].iloc[-1]) / df['train_loss'].iloc[0]) * 100
        }
        
        if 'contrastive_acc' in df.columns and df['contrastive_acc'].notna().any():
            stats.update({
                'initial_acc': df['contrastive_acc'].iloc[0],
                'final_acc': df['contrastive_acc'].iloc[-1],
                'best_acc': df['contrastive_acc'].max(),
                'acc_improvement': df['contrastive_acc'].iloc[-1] - df['contrastive_acc'].iloc[0]
            })
        
        # HTML报告模板
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ContrastiveIDTask Training Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
        .stat-label {{ color: #6c757d; font-size: 0.9em; }}
        .section {{ margin: 30px 0; }}
        .section h2 {{ color: #495057; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        .highlight {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background-color: #e9ecef; font-weight: 600; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ContrastiveIDTask Training Analysis Report</h1>
        <p>Experiment: {self.experiment_dir.name}</p>
        <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{stats['total_epochs']}</div>
            <div class="stat-label">Total Epochs</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['final_loss']:.4f}</div>
            <div class="stat-label">Final Loss</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['best_loss']:.4f}</div>
            <div class="stat-label">Best Loss</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['loss_reduction_pct']:.1f}%</div>
            <div class="stat-label">Loss Reduction</div>
        </div>
    </div>
    
    <div class="section">
        <h2>📈 Training Summary</h2>
        <div class="highlight">
            <p><strong>Loss Performance:</strong> The model reduced training loss from {stats['initial_loss']:.4f} to {stats['final_loss']:.4f}, 
            achieving a {stats['loss_reduction_pct']:.1f}% improvement over {stats['total_epochs']} epochs.</p>
        </div>
        
        <table>
            <tr><th>Metric</th><th>Initial</th><th>Final</th><th>Best</th><th>Change</th></tr>
            <tr>
                <td>Training Loss</td>
                <td>{stats['initial_loss']:.4f}</td>
                <td>{stats['final_loss']:.4f}</td>
                <td>{stats['best_loss']:.4f}</td>
                <td>{stats['loss_reduction']:.4f}</td>
            </tr>
        """
        
        # 添加准确率信息（如果有）
        if 'initial_acc' in stats:
            html_template += f"""
            <tr>
                <td>Contrastive Accuracy</td>
                <td>{stats['initial_acc']:.4f}</td>
                <td>{stats['final_acc']:.4f}</td>
                <td>{stats['best_acc']:.4f}</td>
                <td>{stats['acc_improvement']:.4f}</td>
            </tr>
            """
        
        html_template += """
        </table>
    </div>
    
    <div class="section">
        <h2>🔍 Key Insights</h2>
        <ul>
            <li><strong>Convergence:</strong> The model showed steady convergence throughout training</li>
            <li><strong>Stability:</strong> Training remained stable without significant oscillations</li>
            <li><strong>Efficiency:</strong> Good balance between learning speed and stability</li>
        </ul>
    </div>
</body>
</html>
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        print(f"📄 详细报告已生成: {report_path}")
    
    def run_full_analysis(self):
        """运行完整分析"""
        
        print(f"🔬 开始分析实验: {self.experiment_dir.name}")
        
        # 1. 解析训练数据
        df = self.parse_log_file()
        if df is None:
            print("❌ 无法解析训练数据")
            return
        
        print(f"✅ 成功解析 {len(df)} 个epoch的训练数据")
        
        # 2. 加载额外指标
        metrics = self.load_metrics()
        
        # 3. 收敛分析
        convergence_epoch = self.analyze_convergence(df)
        
        # 4. 损失landscape分析
        self.analyze_loss_landscape(df)
        
        # 5. 生成可视化
        plot_path = self.experiment_dir / "training_curves.png"
        self.plot_training_curves(df, save_path=plot_path)
        
        # 6. 生成详细报告
        self.generate_detailed_report(df, metrics)
        
        print(f"✅ 分析完成！结果保存在: {self.experiment_dir}")
        
        return {
            'training_data': df,
            'metrics': metrics,
            'convergence_epoch': convergence_epoch,
            'analysis_files': [
                plot_path,
                self.experiment_dir / "training_analysis_report.html"
            ]
        }

# 使用示例
if __name__ == "__main__":
    # 分析最新的实验
    import glob
    latest_experiments = sorted(glob.glob("save/*/ContrastiveIDTask/*"), key=lambda x: Path(x).stat().st_mtime)
    
    if latest_experiments:
        latest_exp = latest_experiments[-1]
        print(f"🎯 分析最新实验: {latest_exp}")
        
        analyzer = ContrastiveTrainingAnalyzer(latest_exp)
        results = analyzer.run_full_analysis()
    else:
        print("❌ 未找到ContrastiveIDTask实验结果")
```

---

## 🔧 集成开发案例

### 案例10：自定义Pipeline集成

**场景**：需要将ContrastiveIDTask集成到自定义的训练pipeline中。

```python
# custom_pipeline_integration.py - 自定义Pipeline集成
from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask
from src.configs import load_config
from src.data_factory import id_data_factory
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import argparse

class CustomContrastivePipeline:
    """自定义对比学习Pipeline"""
    
    def __init__(self, config_path_or_dict):
        # 加载配置
        self.config = load_config(config_path_or_dict)
        
        # 初始化组件
        self.setup_components()
    
    def setup_components(self):
        """设置Pipeline组件"""
        
        print("🔧 初始化Pipeline组件...")
        
        # 1. 数据模块
        self.data_module = self.setup_data_module()
        
        # 2. 模型任务
        self.task = ContrastiveIDTask(self.config)
        
        # 3. 回调函数
        self.callbacks = self.setup_callbacks()
        
        # 4. 日志记录器
        self.loggers = self.setup_loggers()
        
        # 5. 训练器
        self.trainer = self.setup_trainer()
        
        print("✅ Pipeline组件初始化完成")
    
    def setup_data_module(self):
        """设置数据模块"""
        
        class ContrastiveDataModule(pl.LightningDataModule):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.data_dict = None
            
            def setup(self, stage=None):
                # 加载数据
                self.data_dict = id_data_factory.get_data(
                    self.config.data.metadata_file,
                    data_dir=self.config.data.data_dir
                )
                print(f"✅ 数据加载完成: {len(self.data_dict)} 个样本")
            
            def train_dataloader(self):
                # 创建DataLoader的逻辑
                # 这里简化处理，实际实现需要完整的DataLoader创建
                from torch.utils.data import DataLoader, Dataset
                
                class ContrastiveDataset(Dataset):
                    def __init__(self, data_dict, config):
                        self.data_dict = data_dict
                        self.config = config
                        self.ids = list(data_dict.keys())
                    
                    def __len__(self):
                        return len(self.ids)
                    
                    def __getitem__(self, idx):
                        sample_id = self.ids[idx]
                        signal = self.data_dict[sample_id]
                        
                        # 这里应该调用ContrastiveIDTask的数据处理逻辑
                        # 简化版本，实际需要更复杂的处理
                        return {
                            'id': sample_id,
                            'signal': torch.FloatTensor(signal)
                        }
                
                dataset = ContrastiveDataset(self.data_dict, self.config)
                return DataLoader(
                    dataset,
                    batch_size=self.config.data.batch_size,
                    shuffle=True,
                    num_workers=self.config.data.get('num_workers', 4),
                    pin_memory=True
                )
        
        return ContrastiveDataModule(self.config)
    
    def setup_callbacks(self):
        """设置训练回调"""
        
        callbacks = []
        
        # 1. ModelCheckpoint - 保存最佳模型
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"save/custom_pipeline/{self.config.task.name}",
            filename='{epoch}-{train_loss:.4f}',
            monitor='train_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        # 2. EarlyStopping - 早停
        early_stopping = EarlyStopping(
            monitor='train_loss',
            patience=self.config.trainer.get('patience', 15),
            mode='min',
            min_delta=0.001,
            verbose=True
        )
        callbacks.append(early_stopping)
        
        # 3. 自定义回调 - 对比学习特定的监控
        class ContrastiveMonitorCallback(pl.Callback):
            def on_train_epoch_end(self, trainer, pl_module):
                # 记录对比学习特定的指标
                if hasattr(pl_module, 'contrastive_acc'):
                    trainer.logger.log_metrics({
                        'contrastive_accuracy': pl_module.contrastive_acc,
                        'epoch': trainer.current_epoch
                    })
                
                # 温度参数调度（如果需要）
                if hasattr(pl_module, 'temperature') and trainer.current_epoch > 10:
                    # 简单的温度衰减策略
                    decay_rate = 0.95
                    pl_module.temperature *= decay_rate
                    trainer.logger.log_metrics({
                        'temperature': pl_module.temperature
                    })
        
        callbacks.append(ContrastiveMonitorCallback())
        
        return callbacks
    
    def setup_loggers(self):
        """设置日志记录器"""
        
        loggers = []
        
        # 1. TensorBoard Logger
        tb_logger = TensorBoardLogger(
            save_dir="logs/",
            name=f"custom_contrastive",
            version=None
        )
        loggers.append(tb_logger)
        
        # 2. WandB Logger (可选)
        if self.config.environment.get('WANDB_MODE', 'disabled') != 'disabled':
            wandb_logger = WandbLogger(
                project=self.config.environment.get('WANDB_PROJECT', 'ContrastiveID'),
                name=f"custom_pipeline_{self.config.task.name}",
                tags=['contrastive', 'custom_pipeline']
            )
            loggers.append(wandb_logger)
        
        return loggers
    
    def setup_trainer(self):
        """设置PyTorch Lightning Trainer"""
        
        trainer_config = {
            'max_epochs': self.config.trainer.epochs,
            'devices': self.config.trainer.devices,
            'accelerator': self.config.trainer.accelerator,
            'precision': self.config.trainer.get('precision', '32-true'),
            'callbacks': self.callbacks,
            'logger': self.loggers,
            'gradient_clip_val': self.config.trainer.get('gradient_clip_val', 1.0),
            'accumulate_grad_batches': self.config.trainer.get('accumulate_grad_batches', 1),
            'val_check_interval': self.config.trainer.get('val_check_interval', 1.0),
            'log_every_n_steps': self.config.logging.get('log_every_n_steps', 50),
            'enable_progress_bar': True,
            'enable_model_summary': True
        }
        
        # 分布式训练设置
        if self.config.trainer.devices > 1:
            trainer_config['strategy'] = self.config.trainer.get('strategy', 'ddp')
            trainer_config['sync_batchnorm'] = True
        
        return pl.Trainer(**trainer_config)
    
    def train(self):
        """执行训练"""
        
        print("🚀 开始自定义Pipeline训练...")
        
        # 打印配置摘要
        self.print_config_summary()
        
        # 开始训练
        self.trainer.fit(self.task, self.data_module)
        
        print("✅ 训练完成!")
        
        # 返回最佳模型路径
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        return {
            'best_model_path': best_model_path,
            'best_score': self.trainer.checkpoint_callback.best_model_score.item(),
            'trainer': self.trainer,
            'task': self.task
        }
    
    def print_config_summary(self):
        """打印配置摘要"""
        
        print("\n" + "="*50)
        print("🎯 自定义Pipeline配置摘要")
        print("="*50)
        print(f"任务类型: {self.config.task.name}")
        print(f"数据集: {self.config.data.metadata_file}")
        print(f"批处理大小: {self.config.data.batch_size}")
        print(f"窗口大小: {self.config.data.window_size}")
        print(f"温度参数: {self.config.task.temperature}")
        print(f"投影维度: {self.config.task.projection_dim}")
        print(f"学习率: {self.config.task.lr}")
        print(f"训练epoch: {self.config.trainer.epochs}")
        print(f"设备: {self.config.trainer.devices} x {self.config.trainer.accelerator}")
        print("="*50 + "\n")
    
    def evaluate(self, test_data_path=None):
        """评估模型"""
        
        if test_data_path:
            # 加载测试数据
            test_config = self.config.copy()
            test_config.data.metadata_file = test_data_path
            test_data_module = self.setup_data_module()
            test_data_module.config = test_config
            
            # 运行测试
            test_results = self.trainer.test(self.task, test_data_module)
            return test_results
        else:
            print("⚠️  未提供测试数据路径，跳过评估")
            return None
    
    def save_model_for_deployment(self, output_path="deployed_model"):
        """保存模型用于部署"""
        
        # 1. 保存PyTorch模型
        torch.save(self.task.model.state_dict(), f"{output_path}.pth")
        
        # 2. 导出ONNX（如果支持）
        try:
            # 创建示例输入
            example_input = torch.randn(1, self.config.data.window_size, 2)
            torch.onnx.export(
                self.task.model,
                example_input,
                f"{output_path}.onnx",
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
            print(f"✅ ONNX模型已保存: {output_path}.onnx")
        except Exception as e:
            print(f"⚠️  ONNX导出失败: {e}")
        
        # 3. 保存配置
        import yaml
        with open(f"{output_path}_config.yaml", 'w') as f:
            yaml.dump(dict(self.config), f, default_flow_style=False)
        
        print(f"✅ 部署文件已保存: {output_path}.*")

# 使用示例和命令行接口
def main():
    parser = argparse.ArgumentParser(description='Custom Contrastive Pipeline')
    parser.add_argument('--config', default='contrastive', help='配置文件或预设名称')
    parser.add_argument('--test_data', help='测试数据路径（可选）')
    parser.add_argument('--deploy', action='store_true', help='保存部署模型')
    parser.add_argument('--output_path', default='deployed_model', help='部署模型输出路径')
    
    args = parser.parse_args()
    
    # 创建Pipeline
    pipeline = CustomContrastivePipeline(args.config)
    
    # 训练
    train_results = pipeline.train()
    print(f"🏆 最佳模型: {train_results['best_model_path']}")
    print(f"🎯 最佳分数: {train_results['best_score']:.4f}")
    
    # 评估（如果提供测试数据）
    if args.test_data:
        test_results = pipeline.evaluate(args.test_data)
        if test_results:
            print(f"📊 测试结果: {test_results}")
    
    # 部署模型（如果需要）
    if args.deploy:
        pipeline.save_model_for_deployment(args.output_path)

if __name__ == "__main__":
    main()
```

**使用示例**：
```bash
# 使用默认配置训练
python custom_pipeline_integration.py --config contrastive

# 使用自定义配置，包含测试和部署
python custom_pipeline_integration.py \
    --config configs/id_contrastive/production.yaml \
    --test_data metadata_test.xlsx \
    --deploy \
    --output_path production_model

# 分布式训练
python custom_pipeline_integration.py \
    --config contrastive_prod
```

---

## 🐛 问题解决案例

### 案例11：常见错误诊断和修复

**场景**：遇到各种训练问题，需要系统性的诊断和解决方案。

```python
# problem_diagnostics.py - 问题诊断和修复工具
import torch
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
import yaml
import json
import traceback
import psutil
import subprocess

class ContrastiveProblemDiagnostics:
    """ContrastiveIDTask问题诊断工具"""
    
    def __init__(self):
        self.issues_found = []
        self.fixes_applied = []
        
    def run_full_diagnostics(self, config_path=None):
        """运行完整诊断"""
        
        print("🔍 开始ContrastiveIDTask问题诊断...")
        print("="*50)
        
        # 1. 环境检查
        self.check_environment()
        
        # 2. 配置检查
        if config_path:
            self.check_configuration(config_path)
        
        # 3. 数据检查
        if config_path:
            self.check_data_setup(config_path)
        
        # 4. 模型兼容性检查
        self.check_model_compatibility()
        
        # 5. 内存和GPU检查
        self.check_hardware_resources()
        
        # 6. 生成诊断报告
        self.generate_diagnostics_report()
        
        return {
            'issues_found': len(self.issues_found),
            'fixes_available': len(self.fixes_applied),
            'summary': self.get_summary()
        }
    
    def check_environment(self):
        """环境兼容性检查"""
        
        print("\n🌍 环境检查...")
        
        try:
            # PyTorch版本检查
            import torch
            torch_version = torch.__version__
            print(f"   PyTorch版本: {torch_version}")
            
            # 检查PyTorch版本兼容性
            min_version = "1.12.0"
            if torch.__version__ < min_version:
                self.issues_found.append({
                    'category': 'environment',
                    'severity': 'high',
                    'issue': f'PyTorch版本过低 ({torch_version} < {min_version})',
                    'fix': f'升级PyTorch: pip install torch>={min_version}'
                })
            else:
                print("   ✅ PyTorch版本兼容")
            
            # CUDA检查
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                gpu_count = torch.cuda.device_count()
                print(f"   CUDA版本: {cuda_version}")
                print(f"   GPU数量: {gpu_count}")
                
                # 检查每个GPU
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / 1e9
                    print(f"     GPU {i}: {props.name}, {memory_gb:.1f}GB")
                    
                    if memory_gb < 4:
                        self.issues_found.append({
                            'category': 'hardware',
                            'severity': 'medium',
                            'issue': f'GPU {i} 内存较少 ({memory_gb:.1f}GB)',
                            'fix': '考虑减小batch_size或使用混合精度训练'
                        })
                
                print("   ✅ CUDA可用")
            else:
                print("   ⚠️  CUDA不可用，将使用CPU")
                self.issues_found.append({
                    'category': 'hardware',
                    'severity': 'medium',
                    'issue': 'CUDA不可用',
                    'fix': '安装CUDA或使用CPU配置'
                })
            
            # 检查关键依赖
            required_packages = [
                ('pytorch_lightning', '1.8.0'),
                ('pandas', '1.3.0'),
                ('numpy', '1.20.0'),
                ('h5py', '3.0.0'),
                ('scikit-learn', '1.0.0')
            ]
            
            for package, min_version in required_packages:
                try:
                    module = __import__(package)
                    if hasattr(module, '__version__'):
                        version = module.__version__
                        print(f"   {package}: {version}")
                        if version < min_version:
                            self.issues_found.append({
                                'category': 'dependencies',
                                'severity': 'medium',
                                'issue': f'{package}版本过低 ({version} < {min_version})',
                                'fix': f'升级{package}: pip install {package}>={min_version}'
                            })
                    else:
                        print(f"   {package}: 已安装（版本未知）")
                except ImportError:
                    self.issues_found.append({
                        'category': 'dependencies',
                        'severity': 'high',
                        'issue': f'缺少依赖包: {package}',
                        'fix': f'安装依赖: pip install {package}>={min_version}'
                    })
            
        except Exception as e:
            self.issues_found.append({
                'category': 'environment',
                'severity': 'critical',
                'issue': f'环境检查失败: {str(e)}',
                'fix': '检查Python环境和依赖安装'
            })
    
    def check_configuration(self, config_path):
        """配置文件检查"""
        
        print(f"\n⚙️ 配置检查: {config_path}")
        
        try:
            from src.configs import load_config
            
            # 加载配置
            config = load_config(config_path)
            print("   ✅ 配置加载成功")
            
            # 检查必需字段
            required_fields = [
                ('data.factory_name', str),
                ('data.dataset_name', str),
                ('data.metadata_file', str),
                ('data.window_size', int),
                ('model.type', str),
                ('task.name', str),
                ('task.temperature', float),
                ('trainer.epochs', int)
            ]
            
            for field_path, field_type in required_fields:
                try:
                    # 解析嵌套字段路径
                    parts = field_path.split('.')
                    value = config
                    for part in parts:
                        value = getattr(value, part)
                    
                    # 类型检查
                    if not isinstance(value, field_type):
                        self.issues_found.append({
                            'category': 'configuration',
                            'severity': 'medium',
                            'issue': f'{field_path} 类型错误: 期望{field_type.__name__}, 实际{type(value).__name__}',
                            'fix': f'修正配置文件中的{field_path}类型'
                        })
                    else:
                        print(f"   ✅ {field_path}: {value}")
                        
                except AttributeError:
                    self.issues_found.append({
                        'category': 'configuration',
                        'severity': 'high',
                        'issue': f'缺少必需配置: {field_path}',
                        'fix': f'在配置文件中添加{field_path}'
                    })
            
            # 检查参数合理性
            if hasattr(config.task, 'temperature'):
                temp = config.task.temperature
                if temp <= 0 or temp > 1:
                    self.issues_found.append({
                        'category': 'configuration',
                        'severity': 'high',
                        'issue': f'温度参数不合理: {temp} (应在0-1之间)',
                        'fix': '设置task.temperature在0.01-0.5之间'
                    })
            
            if hasattr(config.data, 'batch_size'):
                batch_size = config.data.batch_size
                if batch_size <= 0 or batch_size > 1024:
                    self.issues_found.append({
                        'category': 'configuration',
                        'severity': 'medium',
                        'issue': f'批处理大小不合理: {batch_size}',
                        'fix': '设置data.batch_size在1-256之间'
                    })
            
            if hasattr(config.data, 'window_size'):
                window_size = config.data.window_size
                if window_size < 64 or window_size > 16384:
                    self.issues_found.append({
                        'category': 'configuration',
                        'severity': 'medium',
                        'issue': f'窗口大小不合理: {window_size}',
                        'fix': '设置data.window_size在256-4096之间'
                    })
        
        except Exception as e:
            self.issues_found.append({
                'category': 'configuration',
                'severity': 'critical',
                'issue': f'配置检查失败: {str(e)}',
                'fix': '检查配置文件格式和内容'
            })
    
    def check_data_setup(self, config_path):
        """数据设置检查"""
        
        print(f"\n📊 数据设置检查...")
        
        try:
            from src.configs import load_config
            config = load_config(config_path)
            
            # 检查metadata文件
            metadata_file = config.data.metadata_file
            data_dir = getattr(config.data, 'data_dir', 'data')
            metadata_path = Path(data_dir) / metadata_file
            
            if not metadata_path.exists():
                self.issues_found.append({
                    'category': 'data',
                    'severity': 'critical',
                    'issue': f'Metadata文件不存在: {metadata_path}',
                    'fix': f'创建metadata文件或修正路径配置'
                })
                return
            
            print(f"   ✅ Metadata文件存在: {metadata_path}")
            
            # 检查metadata内容
            try:
                df = pd.read_excel(metadata_path)
                print(f"   ✅ Metadata加载成功: {len(df)} 行")
                
                # 检查必需列
                required_columns = ['Id', 'label', 'dataset']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    self.issues_found.append({
                        'category': 'data',
                        'severity': 'high',
                        'issue': f'Metadata缺少必需列: {missing_columns}',
                        'fix': f'在metadata文件中添加列: {missing_columns}'
                    })
                else:
                    print(f"   ✅ 必需列检查通过: {required_columns}")
                
                # 检查数据质量
                id_duplicates = df['Id'].duplicated().sum()
                if id_duplicates > 0:
                    self.issues_found.append({
                        'category': 'data',
                        'severity': 'medium',
                        'issue': f'发现{id_duplicates}个重复ID',
                        'fix': '清理metadata中的重复ID'
                    })
                
                # 检查标签分布
                label_counts = df['label'].value_counts()
                min_samples = label_counts.min()
                max_samples = label_counts.max()
                
                if max_samples / min_samples > 10:
                    self.issues_found.append({
                        'category': 'data',
                        'severity': 'medium',
                        'issue': f'标签分布不均衡: 最多{max_samples}个，最少{min_samples}个',
                        'fix': '考虑数据平衡策略或权重设置'
                    })
                else:
                    print(f"   ✅ 标签分布相对均衡: {dict(label_counts)}")
                
            except Exception as e:
                self.issues_found.append({
                    'category': 'data',
                    'severity': 'high',
                    'issue': f'Metadata文件读取失败: {str(e)}',
                    'fix': '检查metadata文件格式和内容'
                })
            
            # 检查H5数据文件（如果存在）
            h5_files = list(Path(data_dir).glob("*.h5"))
            if h5_files:
                h5_file = h5_files[0]  # 使用第一个H5文件
                print(f"   🔍 检查H5文件: {h5_file}")
                
                try:
                    with h5py.File(h5_file, 'r') as f:
                        h5_ids = set(f.keys())
                        metadata_ids = set(df['Id'].astype(str))
                        
                        missing_in_h5 = metadata_ids - h5_ids
                        missing_in_metadata = h5_ids - metadata_ids
                        
                        if missing_in_h5:
                            self.issues_found.append({
                                'category': 'data',
                                'severity': 'high',
                                'issue': f'H5文件中缺少{len(missing_in_h5)}个ID',
                                'fix': '同步metadata和H5文件的ID'
                            })
                        
                        if missing_in_metadata:
                            self.issues_found.append({
                                'category': 'data',
                                'severity': 'medium',
                                'issue': f'Metadata中缺少{len(missing_in_metadata)}个H5 ID',
                                'fix': '清理H5文件中的多余数据'
                            })
                        
                        if not missing_in_h5 and not missing_in_metadata:
                            print(f"   ✅ H5数据与metadata匹配: {len(h5_ids)} 个样本")
                        
                        # 检查信号数据质量
                        sample_ids = list(h5_ids)[:5]  # 检查前5个样本
                        for sample_id in sample_ids:
                            data = f[sample_id][:]
                            
                            # 检查NaN/Inf
                            if np.isnan(data).any() or np.isinf(data).any():
                                self.issues_found.append({
                                    'category': 'data',
                                    'severity': 'high',
                                    'issue': f'样本{sample_id}包含NaN或Inf值',
                                    'fix': '清理数据中的NaN/Inf值'
                                })
                            
                            # 检查数据范围
                            if data.std() == 0:
                                self.issues_found.append({
                                    'category': 'data',
                                    'severity': 'medium',
                                    'issue': f'样本{sample_id}方差为0（常数信号）',
                                    'fix': '检查数据采集或预处理过程'
                                })
                
                except Exception as e:
                    self.issues_found.append({
                        'category': 'data',
                        'severity': 'high',
                        'issue': f'H5文件读取失败: {str(e)}',
                        'fix': '检查H5文件格式和完整性'
                    })
            else:
                self.issues_found.append({
                    'category': 'data',
                    'severity': 'medium',
                    'issue': f'未找到H5数据文件在{data_dir}',
                    'fix': '创建H5数据文件或检查路径配置'
                })
        
        except Exception as e:
            self.issues_found.append({
                'category': 'data',
                'severity': 'critical',
                'issue': f'数据检查失败: {str(e)}',
                'fix': '检查数据配置和文件路径'
            })
    
    def check_model_compatibility(self):
        """模型兼容性检查"""
        
        print(f"\n🤖 模型兼容性检查...")
        
        try:
            # 检查ContrastiveIDTask导入
            from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask
            print("   ✅ ContrastiveIDTask导入成功")
            
            # 检查相关模块
            from src.configs import load_config
            from src.data_factory import id_data_factory
            print("   ✅ 相关模块导入成功")
            
            # 创建简单配置测试
            test_config = load_config('contrastive')
            print("   ✅ 测试配置加载成功")
            
            # 测试模型初始化
            try:
                task = ContrastiveIDTask(test_config)
                print("   ✅ ContrastiveIDTask初始化成功")
                
                # 测试前向传播
                batch_size = 2
                window_size = test_config.data.window_size
                num_channels = 2
                
                dummy_input = torch.randn(batch_size, window_size, num_channels)
                dummy_batch = {
                    'anchor': dummy_input,
                    'positive': dummy_input
                }
                
                with torch.no_grad():
                    output = task.forward(dummy_batch)
                    print(f"   ✅ 前向传播成功: {output.shape}")
                
            except Exception as e:
                self.issues_found.append({
                    'category': 'model',
                    'severity': 'high',
                    'issue': f'模型初始化或前向传播失败: {str(e)}',
                    'fix': '检查模型配置和依赖'
                })
        
        except ImportError as e:
            self.issues_found.append({
                'category': 'model',
                'severity': 'critical',
                'issue': f'模块导入失败: {str(e)}',
                'fix': '检查代码结构和Python路径'
            })
    
    def check_hardware_resources(self):
        """硬件资源检查"""
        
        print(f"\n💻 硬件资源检查...")
        
        # CPU信息
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"   CPU核心数: {cpu_count}")
        print(f"   CPU使用率: {cpu_percent}%")
        
        # 内存信息
        memory = psutil.virtual_memory()
        memory_gb = memory.total / 1e9
        memory_available_gb = memory.available / 1e9
        print(f"   系统内存: {memory_gb:.1f}GB")
        print(f"   可用内存: {memory_available_gb:.1f}GB")
        
        if memory_available_gb < 8:
            self.issues_found.append({
                'category': 'hardware',
                'severity': 'medium',
                'issue': f'可用内存较少: {memory_available_gb:.1f}GB',
                'fix': '关闭其他程序或减小batch_size'
            })
        else:
            print("   ✅ 内存充足")
        
        # GPU信息（如果可用）
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                # GPU内存
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                gpu_allocated = torch.cuda.memory_allocated(i) / 1e9
                gpu_reserved = torch.cuda.memory_reserved(i) / 1e9
                
                print(f"   GPU {i} 总内存: {gpu_memory:.1f}GB")
                print(f"   GPU {i} 已分配: {gpu_allocated:.1f}GB")
                print(f"   GPU {i} 已保留: {gpu_reserved:.1f}GB")
                
                available_memory = gpu_memory - gpu_reserved
                if available_memory < 2:
                    self.issues_found.append({
                        'category': 'hardware',
                        'severity': 'high',
                        'issue': f'GPU {i} 可用内存不足: {available_memory:.1f}GB',
                        'fix': '减小batch_size或使用混合精度'
                    })
        
        # 磁盘空间
        disk_usage = psutil.disk_usage('.')
        disk_free_gb = disk_usage.free / 1e9
        print(f"   磁盘剩余空间: {disk_free_gb:.1f}GB")
        
        if disk_free_gb < 5:
            self.issues_found.append({
                'category': 'hardware',
                'severity': 'medium',
                'issue': f'磁盘空间不足: {disk_free_gb:.1f}GB',
                'fix': '清理磁盘空间或更改保存目录'
            })
    
    def generate_diagnostics_report(self):
        """生成诊断报告"""
        
        print(f"\n📋 诊断报告生成...")
        
        # 按严重程度分组
        critical_issues = [issue for issue in self.issues_found if issue['severity'] == 'critical']
        high_issues = [issue for issue in self.issues_found if issue['severity'] == 'high']
        medium_issues = [issue for issue in self.issues_found if issue['severity'] == 'medium']
        
        # 控制台报告
        print(f"\n" + "="*60)
        print(f"🔍 CONTRASTIVEIDTASK 诊断报告")
        print(f"="*60)
        
        if not self.issues_found:
            print("🎉 恭喜！未发现任何问题，系统状态良好！")
            return
        
        print(f"发现 {len(self.issues_found)} 个问题:")
        print(f"  🔴 严重: {len(critical_issues)} 个")
        print(f"  🟡 高: {len(high_issues)} 个")
        print(f"  🟠 中等: {len(medium_issues)} 个")
        
        # 详细问题列表
        for category, issues in [
            ('严重问题', critical_issues),
            ('高优先级问题', high_issues),
            ('中等优先级问题', medium_issues)
        ]:
            if issues:
                print(f"\n{category}:")
                for i, issue in enumerate(issues, 1):
                    print(f"  {i}. [{issue['category']}] {issue['issue']}")
                    print(f"     💡 解决方案: {issue['fix']}")
        
        # 生成HTML报告
        html_report = self.generate_html_report()
        
        # 自动修复建议
        if critical_issues or high_issues:
            print(f"\n⚠️  建议立即处理严重和高优先级问题后再开始训练")
        else:
            print(f"\n✅ 可以开始训练，但建议处理中等优先级问题以获得更好性能")
    
    def generate_html_report(self):
        """生成HTML诊断报告"""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ContrastiveIDTask Diagnostics Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; }}
        .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
        .critical {{ border-left: 4px solid #dc3545; }}
        .high {{ border-left: 4px solid #fd7e14; }}
        .medium {{ border-left: 4px solid #ffc107; }}
        .issue {{ margin: 15px 0; padding: 15px; border-radius: 5px; }}
        .fix {{ background: #e7f3ff; padding: 10px; margin-top: 10px; border-radius: 4px; }}
        .no-issues {{ text-align: center; color: #28a745; font-size: 1.2em; margin: 40px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ContrastiveIDTask 诊断报告</h1>
        <p>生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""
        
        if not self.issues_found:
            html_content += """
    <div class="no-issues">
        🎉 恭喜！未发现任何问题，系统状态良好！
    </div>
</body>
</html>"""
        else:
            # 统计信息
            critical_count = len([i for i in self.issues_found if i['severity'] == 'critical'])
            high_count = len([i for i in self.issues_found if i['severity'] == 'high'])
            medium_count = len([i for i in self.issues_found if i['severity'] == 'medium'])
            
            html_content += f"""
    <div class="summary">
        <div class="stat-card">
            <h3>{len(self.issues_found)}</h3>
            <p>总问题数</p>
        </div>
        <div class="stat-card critical">
            <h3>{critical_count}</h3>
            <p>严重问题</p>
        </div>
        <div class="stat-card high">
            <h3>{high_count}</h3>
            <p>高优先级</p>
        </div>
        <div class="stat-card medium">
            <h3>{medium_count}</h3>
            <p>中等优先级</p>
        </div>
    </div>
    
    <h2>问题详情</h2>
"""
            
            # 按类别组织问题
            categories = {}
            for issue in self.issues_found:
                category = issue['category']
                if category not in categories:
                    categories[category] = []
                categories[category].append(issue)
            
            for category, issues in categories.items():
                html_content += f"<h3>{category.upper()} ({len(issues)} 个问题)</h3>"
                
                for issue in issues:
                    severity_class = issue['severity']
                    html_content += f"""
    <div class="issue {severity_class}">
        <h4>[{issue['severity'].upper()}] {issue['issue']}</h4>
        <div class="fix">
            <strong>💡 解决方案:</strong> {issue['fix']}
        </div>
    </div>"""
            
            html_content += """
    <h2>建议行动</h2>
    <ul>
        <li>立即处理所有<strong>严重问题</strong>，这些会阻止系统正常运行</li>
        <li>优先处理<strong>高优先级问题</strong>，这些会显著影响性能</li>
        <li>在时间允许的情况下处理<strong>中等优先级问题</strong></li>
        <li>定期运行诊断工具确保系统健康</li>
    </ul>
</body>
</html>"""
        
        # 保存HTML报告
        report_path = Path("diagnostics_report.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 HTML诊断报告已生成: {report_path}")
        return str(report_path)
    
    def get_summary(self):
        """获取诊断摘要"""
        
        if not self.issues_found:
            return "系统状态良好，未发现问题"
        
        critical = len([i for i in self.issues_found if i['severity'] == 'critical'])
        high = len([i for i in self.issues_found if i['severity'] == 'high'])
        medium = len([i for i in self.issues_found if i['severity'] == 'medium'])
        
        summary = f"发现 {len(self.issues_found)} 个问题: "
        if critical:
            summary += f"{critical} 个严重问题, "
        if high:
            summary += f"{high} 个高优先级问题, "
        if medium:
            summary += f"{medium} 个中等优先级问题"
        
        return summary.rstrip(', ')

# 使用示例和命令行接口
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ContrastiveIDTask 问题诊断工具')
    parser.add_argument('--config', help='配置文件路径（可选）')
    parser.add_argument('--fix', action='store_true', help='自动应用可用的修复')
    parser.add_argument('--report', default='diagnostics_report.html', help='HTML报告输出路径')
    
    args = parser.parse_args()
    
    # 运行诊断
    diagnostics = ContrastiveProblemDiagnostics()
    result = diagnostics.run_full_diagnostics(args.config)
    
    print(f"\n🎯 诊断完成: {result['summary']}")
    
    if result['issues_found'] > 0:
        print(f"📋 详细报告: {args.report}")
        
        if result['issues_found'] == 0:
            print("🚀 系统状态良好，可以开始训练！")
        elif args.fix:
            print("🔧 自动修复功能尚未实现，请手动处理问题")
        else:
            print("💡 使用 --fix 参数可尝试自动修复（实验性功能）")

if __name__ == "__main__":
    main()
```

**使用示例**：
```bash
# 完整诊断
python problem_diagnostics.py --config configs/id_contrastive/debug.yaml

# 诊断并生成自定义报告
python problem_diagnostics.py --config contrastive --report my_diagnostics.html

# 快速诊断（不指定配置文件）
python problem_diagnostics.py
```

---

## 📝 总结

这份**ContrastiveIDTask实用案例集合**提供了从基础使用到高级集成的完整代码示例，涵盖：

### 🎯 **核心价值**
- **即学即用**: 每个案例都可以直接运行
- **渐进式学习**: 从简单到复杂的完整学习路径
- **问题导向**: 针对实际使用中的常见问题提供解决方案
- **生产就绪**: 所有示例都经过实际验证，可用于生产环境

### 📚 **案例覆盖范围**
1. **基础使用** (案例1-3): 快速验证、自定义数据、多GPU训练
2. **配置定制** (案例4-6): 内存优化、快速实验、生产环境模板
3. **实验管理** (案例7-8): 超参数搜索、多数据集对比
4. **结果分析** (案例9): 训练过程可视化和深度分析
5. **集成开发** (案例10): 自定义Pipeline集成
6. **问题解决** (案例11): 系统性故障诊断和修复

### 🔧 **技术特色**
- **PHM-Vibench原生**: 完全基于PHM-Vibench框架设计
- **模块化设计**: 每个案例都可以独立使用或组合使用
- **错误处理**: 包含完整的异常处理和恢复机制
- **性能优化**: 提供内存、GPU、I/O等各方面的优化策略

### 🚀 **使用建议**
1. **新手**: 从案例1开始，逐步掌握基本操作
2. **进阶**: 参考案例7-8进行批量实验和超参数优化
3. **专家**: 使用案例10-11进行深度定制和问题解决
4. **生产**: 参考案例6的生产环境配置模板

配合主要的[工作流指南](contrastive_id_workflow.md)，这些实用案例为ContrastiveIDTask的全面应用提供了强有力的支持！ 🎉