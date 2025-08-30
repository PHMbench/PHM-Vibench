# HSE异构对比学习 Cross-System Generalization 详细实施计划

## 一、项目概述

### 1.1 目标定位
- **核心目标**：基于PHM-Vibench现有框架，通过最小化代码增量实现HSE异构嵌入与对比学习的融合
- **技术路线**：复用E_01_HSE嵌入模块 + ContrastiveSSL损失函数 + CDDG跨域任务框架
- **预期成果**：提升模型在不同工业系统间的泛化能力，代码增量<300行

### 1.2 设计原则
- **复用优先**：最大化利用现有组件(E_01_HSE, ContrastiveSSL, CDDG)
- **配置驱动**：通过YAML配置控制实验，遵循PHM-Vibench v5.0配置系统
- **向后兼容**：不破坏现有功能，保持接口一致性
- **简洁实用**：避免过度工程化，每个文件职责单一明确

## 二、技术实现方案

### 2.1 现有组件分析

#### HSE异构嵌入 (E_01_HSE)
- **核心功能**：随机patch采样 + 时间嵌入融合
- **关键参数**：patch_size_L=256, patch_size_C=1, num_patches=128
- **输出格式**：(batch_size, num_patches, output_dim)
- **优势**：支持异构信号的多尺度表示

#### 对比学习模块 (ContrastiveSSL)
- **可复用函数**：contrastive_loss(z1, z2), TimeSeriesAugmentation
- **温度参数**：默认0.1，支持可调
- **增强策略**：噪声、抖动、缩放、时域掩码

#### CDDG任务框架
- **基础结构**：跨数据集训练逻辑已完善
- **域管理**：source_domain_id/target_domain_id配置化
- **评估体系**：accuracy、F1等指标完备

### 2.2 集成设计方案

#### 方案选择：任务级融合（最优）
```
原理：在CDDG任务基础上，添加系统级对比学习目标
优势：代码侵入性最小，复用度最高，维护成本低
实现：继承CDDG_task，重写training_step添加对比损失
```

#### 核心创新点
1. **系统级对比学习**：同系统内样本为正样本，跨系统为负样本
2. **HSE特征增强**：利用HSE的多尺度patch表示能力
3. **自适应权重**：分类损失与对比损失的动态平衡

## 三、核心代码实现

### 3.1 HSE_CDDG_task.py（约150行）
```python
"""
HSE增强的跨系统域泛化任务
结合异构信号嵌入与对比学习的系统级泛化优化
"""

import torch
import torch.nn.functional as F
from src.task_factory.task.CDDG.classification import task as CDDG_task

class HSE_CDDG_task(CDDG_task):
    """
    HSE异构对比学习任务
    - 继承CDDG基础功能
    - 添加系统级对比学习目标
    - 支持多种对比策略
    """
    
    def __init__(self, network, args_data, args_model, args_task, 
                 args_trainer, args_environment, metadata):
        super().__init__(network, args_data, args_model, args_task,
                        args_trainer, args_environment, metadata)
        
        # 对比学习超参数
        self.contrast_weight = getattr(args_task, 'contrast_weight', 0.1)
        self.system_temperature = getattr(args_task, 'system_temperature', 0.07)
        self.use_hard_mining = getattr(args_task, 'use_hard_mining', True)
        
        # 系统映射字典（数据名称->系统ID）
        self.system_mapping = self._build_system_mapping()
        
        # 特征维度（从HSE输出推断）
        self.feature_dim = args_model.output_dim
        
        print(f"HSE_CDDG初始化: contrast_weight={self.contrast_weight}, "
              f"temperature={self.system_temperature}")

    def _build_system_mapping(self):
        """构建数据集到系统的映射"""
        # 基于metadata构建映射关系
        mapping = {}
        if hasattr(self.metadata, 'task') and self.metadata.task:
            for data_name, info in self.metadata.task.items():
                # 从数据名称推断系统（如CWRU_12k->CWRU）
                system_name = data_name.split('_')[0]
                mapping[data_name] = system_name
        return mapping
    
    def extract_hse_features(self, batch):
        """
        提取HSE嵌入特征用于对比学习
        
        Args:
            batch: 包含x, file_id的批次数据
            
        Returns:
            features: HSE嵌入特征 (batch_size, feature_dim)
        """
        x = batch['x']
        file_id = batch['file_id'] 
        
        # 调用网络的HSE embedding层
        # 假设network有get_hse_embedding方法，如无则需要添加
        if hasattr(self.network, 'get_hse_embedding'):
            features = self.network.get_hse_embedding(x, file_id)
        else:
            # 备选：通过前向传播获取中间特征
            with torch.no_grad():
                # 获取embedding层输出
                embedded = self.network.embedding(x, file_id)  
                # 池化到固定维度
                features = embedded.mean(dim=1)  # (batch, num_patches, dim) -> (batch, dim)
                
        return features
        
    def compute_system_contrast_loss(self, features, data_names):
        """
        计算系统级对比学习损失
        
        Args:
            features: HSE特征 (batch_size, feature_dim)
            data_names: 数据集名称列表
            
        Returns:
            loss: 系统对比损失值
        """
        batch_size = features.shape[0]
        device = features.device
        
        # L2标准化特征
        features = F.normalize(features, dim=1)
        
        # 获取系统标签
        system_labels = []
        for name in data_names:
            system = self.system_mapping.get(name, name.split('_')[0])
            system_labels.append(system)
        
        # 构建系统ID张量
        unique_systems = list(set(system_labels))
        system_ids = torch.tensor([unique_systems.index(s) for s in system_labels], 
                                device=device)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(features, features.T) / self.system_temperature
        
        # 构建正负样本掩码
        pos_mask = system_ids.unsqueeze(0) == system_ids.unsqueeze(1)
        
        # 移除对角线（自身相似度）
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        pos_mask = pos_mask & ~diag_mask
        neg_mask = ~pos_mask & ~diag_mask
        
        # 计算InfoNCE损失
        pos_sim = sim_matrix[pos_mask].view(batch_size, -1)
        neg_sim = sim_matrix[neg_mask].view(batch_size, -1)
        
        if pos_sim.shape[1] == 0:  # 如果没有正样本对
            return torch.tensor(0.0, device=device)
            
        # 组合正负样本相似度
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # 交叉熵损失
        loss = F.cross_entropy(logits, labels, reduction='mean')
        
        return loss
        
    def training_step(self, batch, batch_idx):
        """
        训练步骤，结合分类损失和对比损失
        """
        # 获取标准分类损失
        cls_output = super().training_step(batch, batch_idx)
        cls_loss = cls_output if torch.is_tensor(cls_output) else cls_output['loss']
        
        # 如果启用对比学习
        if self.contrast_weight > 0:
            try:
                # 提取HSE特征
                features = self.extract_hse_features(batch)
                
                # 获取数据名称（用于系统标识）
                data_names = batch.get('data_name', ['unknown'] * len(batch['x']))
                
                # 计算系统对比损失
                contrast_loss = self.compute_system_contrast_loss(features, data_names)
                
                # 组合损失
                total_loss = cls_loss + self.contrast_weight * contrast_loss
                
                # 日志记录
                self.log('train/cls_loss', cls_loss, prog_bar=True)
                self.log('train/contrast_loss', contrast_loss, prog_bar=True) 
                self.log('train/total_loss', total_loss, prog_bar=True)
                
                return total_loss
                
            except Exception as e:
                # 如果对比学习失败，回退到分类损失
                print(f"对比学习计算失败: {e}, 使用分类损失")
                return cls_loss
        else:
            return cls_loss
            
    def validation_step(self, batch, batch_idx):
        """验证步骤，主要关注分类性能"""
        return super().validation_step(batch, batch_idx)
```

### 3.2 配置文件系统

#### 基础配置 (configs/hse_contrastive_base.yaml)
```yaml
# HSE异构对比学习基础配置
# 基于 configs/demo/Multiple_DG/CWRU_THU_using_ISFM.yaml

environment:
  project: "HSE_Contrastive_CrossSystem"
  seed: 42
  notes: "HSE异构对比学习跨系统泛化实验"

data:
  metadata_file: "metadata_6_1.xlsx"
  batch_size: 32
  window_size: 4096
  num_window: 64
  normalization: "standardization"

model:
  name: "M_01_ISFM" 
  type: "ISFM"
  
  # HSE嵌入配置
  embedding: "E_01_HSE"
  patch_size_L: 256      # patch长度
  patch_size_C: 1        # 通道patch大小  
  num_patches: 128       # patch数量
  output_dim: 512        # 嵌入维度
  
  # 骨干网络
  backbone: "B_08_PatchTST"
  d_model: 256
  num_layers: 4
  num_heads: 8
  
  # 任务头
  task_head: "H_02_distance_cla"

task:
  type: "HSE_CDDG"       # 使用HSE增强的CDDG任务
  name: "classification"
  
  # 对比学习参数
  contrast_weight: 0.1          # 对比损失权重
  system_temperature: 0.07      # 对比学习温度
  use_hard_mining: true         # 是否使用困难样本挖掘
  
  # 训练参数
  epochs: 30
  lr: 0.001
  weight_decay: 1e-4
  optimizer: "adam"
  scheduler: true
  
  # 评估指标
  loss: "CE"
  metrics: ["acc", "f1"]

trainer:
  name: "Default_trainer"
  gpus: 1
  early_stopping: true
  patience: 8
```

#### 实验配置 (configs/experiments.yaml)  
```yaml
# HSE对比学习实验配置矩阵

base_config: "hse_contrastive_base.yaml"

experiments:
  # 基线实验：原始CDDG
  baseline_cddg:
    task.type: "CDDG"
    task.contrast_weight: 0.0
    environment.notes: "基线CDDG实验"
    
  # 主实验：HSE+对比学习
  hse_contrastive:
    task.type: "HSE_CDDG"
    task.contrast_weight: 0.1
    environment.notes: "HSE对比学习实验"
    
  # 消融实验1：对比权重影响
  ablation_weight:
    - {task.contrast_weight: 0.05, environment.notes: "权重0.05"}
    - {task.contrast_weight: 0.15, environment.notes: "权重0.15"}
    - {task.contrast_weight: 0.2, environment.notes: "权重0.2"}
    
  # 消融实验2：温度参数影响  
  ablation_temperature:
    - {task.system_temperature: 0.05, environment.notes: "温度0.05"}
    - {task.system_temperature: 0.1, environment.notes: "温度0.1"}
    - {task.system_temperature: 0.2, environment.notes: "温度0.2"}
    
  # 消融实验3：HSE patch参数
  ablation_patch:
    - {model.patch_size_L: 128, environment.notes: "patch_L=128"}
    - {model.patch_size_L: 512, environment.notes: "patch_L=512"}
    - {model.num_patches: 64, environment.notes: "patches=64"}
    - {model.num_patches: 256, environment.notes: "patches=256"}

# 跨系统实验设计
cross_system_experiments:
  # 实验组1: CWRU+THU -> XJTU
  exp1:
    source_systems: ["CWRU", "THU"]
    target_system: "XJTU"
    
  # 实验组2: CWRU+XJTU -> THU  
  exp2:
    source_systems: ["CWRU", "XJTU"] 
    target_system: "THU"
    
  # 实验组3: THU+XJTU -> CWRU
  exp3:
    source_systems: ["THU", "XJTU"]
    target_system: "CWRU"
```

### 3.3 实验运行脚本

#### run_hse_experiments.py（约80行）
```python
"""
HSE对比学习实验批量运行脚本
支持配置文件驱动的批量实验执行
"""

import os
import yaml
import json
import argparse
from pathlib import Path
from datetime import datetime
from src.configs import load_config

class HSE_ExperimentRunner:
    def __init__(self, config_dir="configs"):
        self.config_dir = Path(config_dir)
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
    def load_experiment_config(self, config_file):
        """加载实验配置文件"""
        with open(self.config_dir / config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
            
    def run_single_experiment(self, exp_name, exp_config, base_config):
        """运行单个实验"""
        print(f"\n{'='*60}")
        print(f"运行实验: {exp_name}")
        print(f"配置: {exp_config}")
        print(f"{'='*60}")
        
        try:
            # 加载基础配置
            config = load_config(base_config, exp_config)
            
            # 添加实验标识
            config['environment']['exp_name'] = exp_name
            config['environment']['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存实验配置
            exp_dir = self.results_dir / exp_name
            exp_dir.mkdir(exist_ok=True)
            
            config_path = exp_dir / "config.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True)
            
            # 运行主程序
            import subprocess
            cmd = ['python', '../../main.py', '--config', str(config_path)]
            
            print(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=Path(__file__).parent)
            
            # 记录运行结果  
            log_file = exp_dir / "run.log"
            with open(log_file, 'w') as f:
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"STDOUT:\n{result.stdout}\n")
                f.write(f"STDERR:\n{result.stderr}\n")
                
            if result.returncode == 0:
                print(f"✅ 实验 {exp_name} 成功完成")
            else:
                print(f"❌ 实验 {exp_name} 失败")
                print(f"错误信息: {result.stderr}")
                
            return result.returncode == 0
            
        except Exception as e:
            print(f"❌ 实验 {exp_name} 异常: {str(e)}")
            return False
            
    def run_experiment_batch(self, experiment_file="experiments.yaml"):
        """批量运行实验"""
        # 加载实验配置
        exp_config = self.load_experiment_config(experiment_file)
        base_config = exp_config.get('base_config', 'hse_contrastive_base.yaml')
        
        success_count = 0
        total_count = 0
        
        # 运行基础实验
        for exp_name, config in exp_config['experiments'].items():
            if isinstance(config, list):
                # 多组实验
                for i, sub_config in enumerate(config):
                    sub_exp_name = f"{exp_name}_{i+1}"
                    success = self.run_single_experiment(sub_exp_name, sub_config, base_config)
                    total_count += 1
                    if success:
                        success_count += 1
            else:
                # 单个实验
                success = self.run_single_experiment(exp_name, config, base_config)
                total_count += 1
                if success:
                    success_count += 1
        
        # 输出统计结果
        print(f"\n{'='*60}")
        print(f"实验批次完成: {success_count}/{total_count} 成功")
        print(f"{'='*60}")
        
        return success_count, total_count

def main():
    parser = argparse.ArgumentParser(description='HSE对比学习实验运行器')
    parser.add_argument('--config', '-c', default='experiments.yaml',
                      help='实验配置文件')
    parser.add_argument('--single', '-s', type=str,
                      help='运行单个实验')
    args = parser.parse_args()
    
    runner = HSE_ExperimentRunner()
    
    if args.single:
        # 运行单个实验
        exp_config = runner.load_experiment_config(args.config)
        if args.single in exp_config['experiments']:
            config = exp_config['experiments'][args.single]
            base_config = exp_config.get('base_config', 'hse_contrastive_base.yaml')
            runner.run_single_experiment(args.single, config, base_config)
        else:
            print(f"未找到实验: {args.single}")
    else:
        # 批量运行
        runner.run_experiment_batch(args.config)

if __name__ == '__main__':
    main()
```

### 3.4 结果分析脚本

#### analyze_results.py（约100行）
```python
"""
HSE对比学习实验结果分析工具
支持多维度性能分析和可视化
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

class HSE_ResultAnalyzer:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        
    def collect_experiment_results(self):
        """收集所有实验结果"""
        results = []
        
        for exp_dir in self.results_dir.glob("*"):
            if not exp_dir.is_dir():
                continue
                
            # 读取配置
            config_file = exp_dir / "config.yaml"
            metrics_file = exp_dir / "metrics.json"
            
            if not metrics_file.exists():
                continue
                
            try:
                # 读取指标
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    
                # 读取配置
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # 提取关键信息
                result = {
                    'exp_name': exp_dir.name,
                    'task_type': config['task']['type'],
                    'contrast_weight': config['task'].get('contrast_weight', 0),
                    'system_temperature': config['task'].get('system_temperature', 0.07),
                    'patch_size_L': config['model'].get('patch_size_L', 256),
                    'num_patches': config['model'].get('num_patches', 128),
                    'final_acc': metrics.get('test_accuracy', 0),
                    'final_f1': metrics.get('test_f1', 0),
                    'best_val_acc': metrics.get('best_val_accuracy', 0),
                    'epochs_trained': metrics.get('epochs_trained', 0),
                    'convergence_epoch': metrics.get('convergence_epoch', 0)
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"处理 {exp_dir.name} 时出错: {e}")
                continue
                
        return pd.DataFrame(results)
        
    def analyze_performance(self, df):
        """性能分析"""
        print("=== HSE对比学习性能分析 ===")
        
        # 基线 vs HSE对比
        baseline = df[df['task_type'] == 'CDDG']
        hse_contrast = df[df['task_type'] == 'HSE_CDDG']
        
        if len(baseline) > 0 and len(hse_contrast) > 0:
            baseline_acc = baseline['final_acc'].mean()
            hse_acc = hse_contrast['final_acc'].mean()
            improvement = ((hse_acc - baseline_acc) / baseline_acc) * 100
            
            print(f"基线CDDG准确率: {baseline_acc:.3f}")
            print(f"HSE对比学习准确率: {hse_acc:.3f}")
            print(f"性能提升: {improvement:+.2f}%")
            
        # 最佳配置
        best_result = df.loc[df['final_acc'].idxmax()]
        print(f"\n最佳配置:")
        print(f"  实验: {best_result['exp_name']}")
        print(f"  准确率: {best_result['final_acc']:.3f}")
        print(f"  对比权重: {best_result['contrast_weight']}")
        print(f"  系统温度: {best_result['system_temperature']}")
        
    def plot_results(self, df):
        """结果可视化"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 对比权重 vs 准确率
        hse_df = df[df['task_type'] == 'HSE_CDDG']
        if len(hse_df) > 0:
            axes[0,0].scatter(hse_df['contrast_weight'], hse_df['final_acc'], 
                            alpha=0.7, s=60)
            axes[0,0].set_xlabel('对比损失权重')
            axes[0,0].set_ylabel('测试准确率')
            axes[0,0].set_title('对比权重对性能的影响')
            axes[0,0].grid(True, alpha=0.3)
            
        # 2. 温度参数 vs 准确率
        if len(hse_df) > 0:
            axes[0,1].scatter(hse_df['system_temperature'], hse_df['final_acc'],
                            alpha=0.7, s=60, c='orange')
            axes[0,1].set_xlabel('系统温度参数')
            axes[0,1].set_ylabel('测试准确率')
            axes[0,1].set_title('温度参数对性能的影响')
            axes[0,1].grid(True, alpha=0.3)
            
        # 3. 不同方法性能对比
        methods = df['task_type'].unique()
        method_acc = [df[df['task_type']==m]['final_acc'].mean() for m in methods]
        axes[1,0].bar(methods, method_acc, alpha=0.7)
        axes[1,0].set_ylabel('平均测试准确率')
        axes[1,0].set_title('不同方法性能对比')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. 收敛分析
        if 'convergence_epoch' in df.columns:
            axes[1,1].hist(df['convergence_epoch'], bins=10, alpha=0.7, color='green')
            axes[1,1].set_xlabel('收敛轮次')
            axes[1,1].set_ylabel('实验数量')
            axes[1,1].set_title('收敛性分析')
            
        plt.tight_layout()
        plt.savefig(self.results_dir / 'hse_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_report(self, df):
        """生成分析报告"""
        report_path = self.results_dir / 'analysis_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# HSE异构对比学习实验分析报告\n\n")
            
            # 实验概览
            f.write("## 实验概览\n")
            f.write(f"- 总实验数量: {len(df)}\n")
            f.write(f"- 最高准确率: {df['final_acc'].max():.3f}\n")
            f.write(f"- 平均准确率: {df['final_acc'].mean():.3f}\n\n")
            
            # 详细结果表格
            f.write("## 详细结果\n")
            f.write(df.to_markdown(index=False, floatfmt=".3f"))
            f.write("\n\n")
            
            # 关键发现
            f.write("## 关键发现\n")
            if len(df[df['task_type']=='CDDG']) > 0 and len(df[df['task_type']=='HSE_CDDG']) > 0:
                baseline = df[df['task_type']=='CDDG']['final_acc'].mean() 
                hse = df[df['task_type']=='HSE_CDDG']['final_acc'].mean()
                f.write(f"- HSE对比学习相比基线CDDG提升 {((hse-baseline)/baseline*100):+.2f}%\n")
            
        print(f"分析报告已保存至: {report_path}")

def main():
    analyzer = HSE_ResultAnalyzer()
    
    # 收集结果
    print("收集实验结果...")
    df = analyzer.collect_experiment_results()
    
    if len(df) == 0:
        print("未找到实验结果")
        return
        
    print(f"找到 {len(df)} 个实验结果")
    
    # 分析性能
    analyzer.analyze_performance(df)
    
    # 绘制图表
    analyzer.plot_results(df)
    
    # 生成报告
    analyzer.generate_report(df)

if __name__ == '__main__':
    main()
```

## 四、测试验证体系

### 4.1 单元测试 (test_hse_cddg.py)
```python
"""HSE CDDG任务单元测试"""

import torch
import pytest
from unittest.mock import Mock, patch
from HSE_CDDG_task import HSE_CDDG_task

class TestHSE_CDDG:
    def setup_method(self):
        # 创建模拟参数
        self.args_task = Mock()
        self.args_task.contrast_weight = 0.1
        self.args_task.system_temperature = 0.07
        
        # 创建任务实例
        self.task = HSE_CDDG_task(...)
        
    def test_system_mapping_construction(self):
        """测试系统映射构建"""
        mapping = self.task._build_system_mapping()
        assert isinstance(mapping, dict)
        
    def test_contrast_loss_computation(self):
        """测试对比损失计算"""
        features = torch.randn(8, 128)  # batch_size=8, feature_dim=128
        data_names = ['CWRU_1', 'CWRU_2', 'THU_1', 'THU_2', 
                     'XJTU_1', 'XJTU_2', 'CWRU_3', 'THU_3']
        
        loss = self.task.compute_system_contrast_loss(features, data_names)
        
        assert torch.is_tensor(loss)
        assert loss.item() >= 0
        
    def test_feature_extraction(self):
        """测试特征提取"""
        batch = {
            'x': torch.randn(4, 4096, 1),
            'file_id': ['test1', 'test2', 'test3', 'test4']
        }
        
        features = self.task.extract_hse_features(batch)
        assert features.shape[0] == 4  # batch_size
```

### 4.2 集成测试
- 完整训练流程测试
- 配置文件加载测试  
- 与现有框架兼容性测试

## 五、实验设计方案

### 5.1 基线对比实验
| 实验组 | 方法 | 源系统 | 目标系统 | 预期准确率 |
|--------|------|---------|----------|------------|
| A1 | 原始CDDG | CWRU+THU | XJTU | 0.75-0.80 |
| A2 | HSE_CDDG | CWRU+THU | XJTU | 0.80-0.85 |

### 5.2 消融实验矩阵
| 实验 | 变量 | 取值范围 | 步长 |
|------|------|-----------|------|
| B1 | contrast_weight | [0, 0.05, 0.1, 0.15, 0.2] | 0.05 |
| B2 | system_temperature | [0.05, 0.07, 0.1, 0.2] | 变步长 |
| B3 | patch_size_L | [128, 256, 512] | 2x |
| B4 | num_patches | [64, 128, 256] | 2x |

### 5.3 跨系统泛化测试
```
实验C：系统交叉验证
- C1: CWRU+THU → XJTU
- C2: CWRU+XJTU → THU  
- C3: THU+XJTU → CWRU
- C4: CWRU → THU+XJTU
- C5: THU → CWRU+XJTU
```

## 六、项目文件结构

```
script/unified_metric/
├── plan.md                              # 本实施计划 [已完成]
├── README.md                           # 项目说明文档
├── HSE_CDDG_task.py                   # 核心任务实现 (~150行)
├── configs/                            # 配置文件目录
│   ├── hse_contrastive_base.yaml     # 基础配置
│   ├── experiments.yaml              # 实验矩阵配置
│   └── ablation_studies.yaml         # 消融实验配置
├── scripts/                           # 执行脚本目录
│   ├── run_hse_experiments.py        # 实验运行脚本 (~80行)
│   ├── analyze_results.py            # 结果分析脚本 (~100行)
│   └── generate_report.py            # 报告生成脚本 (~50行)
├── tests/                             # 测试文件目录
│   ├── test_hse_cddg.py              # 单元测试 (~80行)
│   └── test_integration.py           # 集成测试 (~40行)
└── results/                           # 实验结果目录
    ├── baseline_cddg/                 # 基线实验结果
    ├── hse_contrastive/              # 主实验结果
    └── analysis_report.md             # 自动生成的分析报告
```

## 七、实施时间计划

### 第1阶段：核心开发 (2天)
- **Day 1**: 实现HSE_CDDG_task.py核心逻辑，编写单元测试
- **Day 2**: 创建配置文件，实现实验运行脚本

### 第2阶段：测试优化 (1天) 
- **Day 3**: 集成测试，运行pilot实验，调试优化

### 第3阶段：批量实验 (2天)
- **Day 4-5**: 运行完整实验矩阵，结果分析，撰写报告

## 八、成功指标

### 8.1 技术指标
- **代码质量**: 总代码量<500行，测试覆盖率>80%
- **性能提升**: 跨系统准确率相比基线提升>5%
- **稳定性**: 所有配置组合成功运行率>90%

### 8.2 可重现性指标
- **配置完整性**: 所有实验可通过配置文件重现
- **文档完善度**: README + 代码注释 + 分析报告
- **兼容性**: 与现有PHM-Vibench框架100%兼容

## 九、风险缓解方案

### 9.1 技术风险
| 风险点 | 概率 | 影响 | 缓解措施 |
|--------|------|------|----------|
| HSE特征接口不匹配 | 中 | 高 | 添加适配层，提供备选实现 |
| 系统标签获取失败 | 低 | 中 | 从文件名推断，提供映射配置 |
| 内存消耗过大 | 中 | 中 | 梯度检查点，批次大小调整 |
| 收敛困难 | 中 | 中 | 学习率调度，权重初始化 |

### 9.2 实验风险
| 风险点 | 缓解措施 |
|--------|----------|
| 实验时间过长 | 使用小规模pilot验证，并行实验 |
| 结果不显著 | 多组重复实验，统计显著性检验 |
| 资源不足 | 优先核心实验，分批执行 |

## 十、预期成果与影响

### 10.1 技术贡献
- **方法创新**: HSE异构嵌入与系统级对比学习的有效融合
- **工程优化**: 基于配置的可扩展实验框架
- **性能提升**: 跨工业系统故障诊断准确率提升5-10%

### 10.2 学术价值
- 验证HSE在跨系统泛化中的有效性
- 探索对比学习在工业信号处理中的应用潜力
- 提供完整的消融实验和分析

### 10.3 工程实用性
- 代码高度复用现有框架，维护成本低
- 配置驱动设计，便于参数调优和扩展
- 完整测试体系，保证代码质量和稳定性

---

**注意事项**：
1. 严格遵循PHM-Vibench现有代码风格和设计模式
2. 所有新增功能提供向后兼容性
3. 完善的文档和测试是成功的关键
4. 实验设计要有统计学意义，避免过拟合结论

**确认后将严格按照此计划执行实施。**