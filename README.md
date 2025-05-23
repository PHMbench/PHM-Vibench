# Vbench: 工业设备振动信号基准平台

<div align="center">
  <img src="pic/Vbench.png" alt="Vbench Logo" width="300"/>
  <p><strong>🏭 工业领域端到端可复现、模块化的故障诊断与预测性维护基准测试平台 🏭</strong></p>
  <p><em>⚠️ 内测阶段 - 仅限邀请访问 ⚠️</em></p>

  <p>
    <img src="https://img.shields.io/badge/状态-内测中-orange" alt="Status: Alpha"/>
    <img src="https://img.shields.io/badge/版本-0.1.0--alpha-blue" alt="Version"/>
    <img src="https://img.shields.io/badge/许可-Apache%202.0-green" alt="License"/>
    <img src="https://img.shields.io/badge/数据集-15+-purple" alt="Datasets"/>
    <img src="https://img.shields.io/badge/算法-30+-red" alt="Algorithms"/>
  </p>

  <p>
    <a href="#-快速开始">快速开始</a> •
    <a href="#-使用指南">使用文档</a> •
    <a href="#-项目亮点">核心特性</a> •
    <a href="#-开发指南">参与贡献</a> •
    <a href="#-常见问题">常见问题</a>
  </p>
</div>

---

## 📖 目录
- [✨ 项目亮点](#-项目亮点)
- [📝 项目背景与简介](#-项目背景与简介)
- [🔄 支持的模型与数据集](#-支持的模型与数据集)
- [🔔 技术动态](#-技术动态)
- [🛠️ 安装指南](#️-安装指南)
- [🚀 快速开始](#-快速开始)
- [📘 使用指南](#-使用指南)
- [📂 项目结构](#-项目结构)
- [🧑‍💻 开发指南](#-开发指南)
- [❓ 常见问题](#-常见问题)
- [📃 用了该项目发表的文章](#-用了该项目发表的文章)
- [🔮 项目路线图](#-项目路线图)
- [👥 贡献者与社区](#-贡献者与社区)
- [🏛  许可证](#-许可证)
- [📎 引用方式](#-引用方式)

## ✨ 项目亮点

<!-- <div align="center">
  <img src="pic/features.png" alt="Vbench Features" width="700"/>
</div> -->

- 🧩 **先进的模块化设计**：采用工厂设计模式实现数据集、模型、任务和训练器的高度模块化，为后续功能扩展提供了灵活架构
- 🔄 **多样化任务支持**：内置对故障分类、异常检测和剩余使用寿命预测等多种故障诊断相关任务的全面支持
- 📊 **丰富的工业数据集集成**：整合15+经典与前沿的工业设备故障诊断数据集，覆盖轴承、齿轮、电机等多种工业部件
- 📏 **精确的评估框架**：提供针对不同故障诊断场景优化的评估指标和专业可视化工具，支持结果的定量分析与比较
- 🖱️ **简洁高效的用户体验**：基于配置文件的实验设计，使研究人员无需修改代码即可快速配置与运行实验
- 📈 **一键复现与基准测试**：内置30+经典和最新算法实现，只需一行命令即可复现论文结果并进行公平比较

<details>
<summary><b>为什么选择Vbench？</b> (点击展开)</summary>
<table>
  <tr>
    <th>特性</th>
    <th>Vbench</th>
    <th>传统PHM工具</th>
  </tr>
  <tr>
    <td>模块化设计</td>
    <td>✅ 高度模块化，组件可随意组合</td>
    <td>❌ 通常耦合紧密，难以扩展</td>
  </tr>
  <tr>
    <td>配置驱动</td>
    <td>✅ 通过YAML文件配置，无需编码</td>
    <td>❌ 多需修改代码，配置繁琐</td>
  </tr>
  <tr>
    <td>一致性评估</td>
    <td>✅ 统一的数据处理和评估标准</td>
    <td>❌ 评估标准不一致</td>
  </tr>
  <tr>
    <td>可复现性</td>
    <td>✅ 完整实验链追踪，结果可复现</td>
    <td>❌ 缺乏完整实验环境记录</td>
  </tr>
  <tr>
    <td>多任务支持</td>
    <td>✅ 分类、检测、寿命预测等多种任务</td>
    <td>⚠️ 通常专注于单一类型任务</td>
  </tr>
</table>
</details>

## 📝 项目背景与简介

**❓为什么需要 Vbench**

### 🎯 A. 项目定位与价值

工业设备故障诊断和预测性维护技术在工业4.0时代具有重要的战略意义，对提高生产效率、降低维护成本和延长设备使用寿命至关重要。然而，随着机器学习和深度学习技术在该领域的广泛应用，研究成果的评估与比较面临以下挑战：

1. 🔍 **实验环境碎片化**：不同研究使用各自的数据预处理流程、模型实现和评估指标
2. 🔄 **可复现性困难**：缺乏标准化的实验流程和完整的实现细节
3. ⚖️ **公平比较的障碍**：数据划分、预处理和评估标准的不一致性导致结果难以直接比较

<!-- <div align="center">
  <img src="pic/motivation.png" alt="Vbench Motivation" width="600"/>
  <p><em>PHM研究面临的挑战</em></p>
</div> -->

Vbench 作为 PHMbench 生态系统中专注于工业设备故障诊断的基准测试平台，旨在提供一个标准化、可复现且易于使用的实验环境，以解决上述挑战。

### 🛠️ B. 核心功能与特性

1. 🔌 **统一的接口设计**：标准化的数据加载、模型训练和评估流程，简化实验实施
2. 🔄 **可复现的实验框架**：基于配置的实验管理确保研究结果可精确复现
3. ⚖️ **公平的比较环境**：统一的数据划分策略和评估指标保证不同方法之间的公平比较
4. 🚀 **快速原型开发支持**：模块化设计使研究人员能高效实施和验证新思路与方法

<!-- <div align="center">
  <img src="pic/workflow.png" alt="Vbench Workflow" width="700"/>
  <p><em>Vbench工作流程</em></p>
</div> -->

## 🔄 支持的模型与数据集

### 📊 支持的数据集 见[Model scope](https://www.modelscope.cn/datasets/RichieTHU/Vbench_data)



### 🧠 支持的算法模型





## 🛠️ 安装指南

> ⚠️ **注意**：当前项目处于内测阶段，仅限获得邀请的用户安装使用。

### 环境要求
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.1+ 

### 依赖安装

```bash
# 克隆仓库
git clone https://github.com/PHMbench/Vbench.git
cd Vbench

# 安装依赖
pip install -r requirements.txt

```

## 🚀 快速开始

通过以下步骤快速体验 Vbench 的功能：

<!-- <div align="center">
  <img src="pic/quickstart.png" alt="Vbench Quick Start" width="650"/>
</div> -->

```bash
# demo pipeline
python main.py --config configs/demo/dummy.yaml

# CWRU 分类任务
python main.py --config configs/demo/CWRU.yaml

# Cross-dataset genealization
python main.py --config configs/demo/CWRU_THU_using_ISFM.yaml
```

### 📊 性能基准示例

<!-- <div align="center">
  <img src="pic/benchmark_results.png" alt="Benchmark Results" width="700"/>
  <p><em>不同模型在CWRU数据集上的性能对比</em></p>
</div> -->

## 📘 使用指南

### 1. 配置文件详解 ⚙️

Vbench 使用 YAML 配置文件定义实验，包含以下主要部分：

```yaml
experiment:
  name: "bearing_fault_diagnosis"
  seed: 42
  
dataset:
  name: "CWRU"  # 案例西储大学轴承数据集
  args:
    task_type: "classification"
    split_ratio: [0.7, 0.1, 0.2]  # 训练/验证/测试集比例
    sampling_rate: 12000  # 采样率(Hz)
    window_size: 1024  # 信号窗口长度

model:
  name: "CNN1D"  # 一维卷积神经网络模型
  args:
    input_channels: 1
    hidden_channels: [16, 32, 64]
    kernel_size: 3
    output_dim: 10

task:
  name: "ClassificationTask"  # 故障分类任务
  args:
    num_classes: 10
    class_weights: null  # 可选，处理类别不平衡

trainer:
  name: "ModularTrainer"  # 训练器
  args:
    epochs: 100
    batch_size: 64
    optimizer: "adam"
    lr: 0.001
    metrics: ["accuracy", "precision", "recall", "f1", "confusion_matrix"]
    early_stopping: true
    patience: 10
```

<!-- <div align="center">
  <img src="pic/config_structure.png" alt="Configuration Structure" width="550"/>
  <p><em>Vbench配置文件结构</em></p>
</div> -->

### 2. 运行实验 🧪

```bash
# 基本用法
python main.py --config configs/your_config.yaml

# 多次重复实验增强结果稳定性
python main.py --config configs/your_config.yaml --iterations 5 --seeds 42,43,44,45,46

# 启用WandB实验跟踪
python main.py --config configs/your_config.yaml --wandb --project "vbench-experiments"

# 使用特定GPU
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/your_config.yaml
```

### 3. 结果分析 📊

实验结果保存在 `results/` 目录下，每次实验会创建以下文件：

- 📁 **模型权重与检查点**：`{experiment_name}/checkpoints/`
- 📄 **评估指标报告**：`{experiment_name}/metrics.json`
- 📝 **详细日志**：`{experiment_name}/log.txt`
- 📊 **可视化结果**：`{experiment_name}/figures/`，包括混淆矩阵、学习曲线等
- 🔄 **实验配置备份**：`{experiment_name}/config.yaml`

<div align="center">
  <img src="pic/results_visualization.png" alt="Results Visualization" width="700"/>
  <p><em>Vbench结果可视化示例</em></p>
</div>

### 4. 结果可视化 📈

<!-- ```bash
# 生成实验结果可视化报告
python scripts/visualize_results.py --result_dir results/experiment_name --output report.pdf

# 生成模型性能比较报告
python scripts/compare_models.py --experiments exp1,exp2,exp3 --metric accuracy

# 导出结果为LaTeX表格（用于论文）
python scripts/export_latex.py --result_dir results/experiment_name
``` -->

## 📂 项目结构

```bash
📂 Vbench
├── 📄 README.md                 # 项目说明
├── 📄 main.py                   # 主入口程序
├── 📄 main_dummy.py             # 功能测试程序
├── 📄 benchmark.py              # 性能基准测试工具
├── 📂 configs                   # 配置文件目录
│   ├── 📂 demo                  # 示例配置
│   │   ├── 📄 cwru_classification.yaml  # CWRU分类实验
│   │   ├── 📄 dummy_test.yaml   # 测试配置
│   │   └── 📄 paderborn_rul.yaml # RUL预测实验
│   └── 📂 experiments           # 实验配置
├── 📂 src                       # 源代码目录
│   ├── 📂 data_factory          # 数据集工厂
│   ├── 📂 model_factory         # 模型工厂
│   ├── 📂 task_factory          # 任务工厂
│   ├── 📂 trainer_factory       # 训练器工厂
│   ├── 📂 visualization         # 可视化工具
│   └── 📂 utils                 # 工具函数
├── 📂 test                      # 测试代码
├── 📂 data                      # 数据目录
├── 📂 results                   # 结果目录
├── 📂 save                      # 模型保存目录
└── 📂 scripts                   # 脚本目录
```

<div align="center">
  <img src="pic/project_structure.png" alt="Project Structure" width="600"/>
  <p><em>Vbench项目结构概览</em></p>
</div>

## 🧑‍💻 开发指南

Vbench 采用模块化设计，遵循工厂模式，便于扩展和定制。如果您希望贡献代码，请参考[贡献者指南](./contributing.md)。

### 扩展数据集 📊 见[数据集贡献指南](./data_factory/contributing.md)

### 添加新模型 🧠 见[模型贡献指南](./model_factory/contributing.md)

### 调试与测试 🐞 见[测试指南](./test/README.md)

## ❓ 常见问题

<!-- <details>
<summary><b>如何处理自定义数据集?</b></summary>
<p>
创建自定义数据集需要继承<code>BaseDataset</code>类并实现所需方法。详细步骤请参考<a href="#扩展数据集-">扩展数据集</a>部分或查看我们的<a href="docs/custom_dataset.md">自定义数据集教程</a>。
</p>
</details>

<details>
<summary><b>实验结果不可复现怎么办?</b></summary>
<p>
请确保设置了相同的随机种子，并使用相同的配置文件。如果问题依然存在，可能是由于硬件差异或PyTorch版本不同导致的。尝试使用我们提供的Docker镜像可以减少环境差异带来的影响。
</p>
</details>

<details>
<summary><b>Vbench是否支持分布式训练?</b></summary>
<p>
是的，Vbench支持基于PyTorch DDP的分布式训练。使用<code>--distributed</code>参数启动训练，例如：<code>python main.py --config your_config.yaml --distributed</code>
</p>
</details>

<details>
<summary><b>如何引用使用Vbench的研究成果?</b></summary>
<p>
请使用本页底部提供的引用格式。同时，建议在论文方法部分明确说明使用了Vbench平台进行实验，并指明所用配置文件和版本号。
</p>
</details> -->

## 📃 用了该项目发表的文章

1. 张三, 李四. (2023). *基于深度学习的轴承故障早期诊断方法研究*. 机械工程学报, 59(3), 131-142.

## 🔮 项目路线图

- **2023 Q3**: 
  - 公开测试版发布
  - 增加在线演示系统
  - 扩展至20+数据集支持

- **2023 Q4**: 
  - 添加预训练模型库
  - 实现自动超参数优化
  - 增加迁移学习与跨域诊断模块

- **2024 Q1**:
  - 推出低代码Web界面
  - 提供云端一键部署解决方案
  - 发布完整文档与教程

## 👥 贡献者与社区

### 核心团队
- [Qi Li](https://github.com/liq22)

### 参与贡献
我们非常欢迎各种形式的贡献！无论是新功能开发、文档改进还是问题反馈。请参阅[贡献指南](CONTRIBUTING.md)了解详情。

### 社区交流
- 加入我们的[Slack频道](https://phmbench.slack.com)讨论问题和新点子
- 加入我们的[飞书群组](https://phmbench.feishu.cn/invite/2d8e0f3b-4a5c-4b1c-9a6f-7d2e0f3b4a5c)获取最新动态
<!-- - 关注我们的[微信公众号](https://mp.weixin.qq.com/phmbench)获取最新资讯
- 参与每月的[线上研讨会](https://phmbench.com/webinars) -->

<div align="center">
  <br>
  <p>🌟 欢迎内测用户提供宝贵反馈! 🌟</p>
  <img src="pic/contact_qrcode.png" alt="联系方式" width="150"/>
  <p><em>扫描二维码加入内测讨论组</em></p>
</div>

## 🏛 许可证

该基准测试平台采用 [Apache License (Version 2.0)](https://github.com/PHMbench/Vbench/blob/master/LICENSE) 许可。对于模型和数据集，请参考原始资源页面并遵循相应的许可证。

## 📎 引用方式

> 📝 **注意**: 项目尚未正式发布，以下引用格式仅供内测用户参考，正式引用格式将随项目公开发布提供。

```bibtex
@misc{vbench2023,
  title={Vbench: A Modular Benchmark for Industrial Fault Diagnosis and Prognosis},
  author={PHMbench Team},
  year={2023},
  howpublished={Internal Testing Version},
  url={https://github.com/PHMbench/Vbench}
}
```

---

## ⭐ Star历史

[![Star History Chart](https://api.star-history.com/svg?repos=PHMbench/Vbench&type=Date)](https://star-history.com/#PHMbench/Vbench&Date)

<p align="center">如有任何问题或建议，请联系我们</a>或提交<a href="https://github.com/PHMbench/Vbench/issues">Issue</a>。</p>
