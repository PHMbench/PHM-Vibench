# PHMbench Documentation Overview

## § Project Overview

PHMbench (Prognostics and Health Management Benchmark) 提供一个基准平台，用于在统一标准下评估故障诊断与剩余寿命预测等PHM算法。框架整合了常用的工业数据集和基准模型，并通过配置驱动的工作流支持实验的复现与结果对比。

## § Target Audience and Roles (Agents)

以下小节定义了项目中常见的五类角色，描述他们在PHMbench中的职责、目标以及与框架的主要交互方式。

### 1. PHM 研究员 / 数据科学家
- **角色定位**：在学术或工业场景中使用PHMbench开展算法研究和实验验证。
- **目标**：
  - 在标准化数据集上运行并复现实验；
  - 公平比较不同算法的性能；
  - 产生可靠、可发布的研究结果。
- **主要交互**：
  - 通过 `configs/` 配置实验；
  - 使用 `src/data_factory/reader/` 与 `src/model_factory/` 中的组件；
  - 分析日志和评估输出。

### 2. PHMbench 开发者 / 贡献者
- **角色定位**：参与框架本身的开发、修复和功能扩展。
- **目标**：
  - 支持新的数据类型、模型或任务；
  - 提升框架的稳定性与可维护性；
  - 完善文档并保持代码质量。
- **主要交互**：
  - 深入阅读 `src/` 目录源码；
  - 遵循 `contributing.md` 流程提交更改；
  - 编写并运行测试（见 `test/`）。

### 3. 数据集策展人 / 提供者
- **角色定位**：向PHMbench引入和维护新的数据集。
- **目标**：
  - 编写或更新 `metadata_*.csv` 元数据；
  - 开发新的读取脚本放在 `src/data_factory/reader/`；
  - 撰写数据集使用说明。
- **主要交互**：
  - 按 `src/data_factory/contributing.md` 集成数据；
  - 同步更新文档。

### 4. AI 模型开发者 (PHM)
- **角色定位**：在框架中实现并测试新型PHM模型。
- **目标**：
  - 集成最新模型架构；
  - 通过标准流程验证模型效果；
  - 确保与现有任务兼容。
- **主要交互**：
  - 在 `src/model_factory/` 下新增模型并更新注册；
  - 新增相应配置文件。

### 5. 基准测试分析师
- **角色定位**：设计并执行大规模基准实验，撰写分析报告。
- **目标**：
  - 系统比较多种算法与数据集；
  - 输出可信的性能评估与洞见。
- **主要交互**：
  - 大批量运行实验并汇总结果；
  - 编写自动化脚本和可视化工具。

## § Tech Stack

- Python 3.8+
- PyTorch 与 PyTorch Lightning
- Pandas、NumPy 等科学计算库
- Hydra 用于配置管理
- 日志与实验跟踪工具：W&B、TensorBoard 等

## § Project Structure

```
phm-vibench/
├── configs/          # 实验配置文件
├── data/             # 数据集与元数据
├── doc/              # 开发者文档
├── script/           # 实用脚本
├── src/              # 框架源码
│   ├── data_factory/
│   ├── model_factory/
│   ├── task_factory/
│   └── trainer_factory/
└── test/             # 测试代码
```

## § Development Guidelines

- 遵循 PEP 8 风格，保持代码整洁。
- 新功能请附带相应测试和文档。
- 提交PR前请运行 `main_dummy.py`进行基本验证。
- [data_factory](./src/data_factory/contributing.md)、[model_factory](./src/model_factory/contributing.md)、[task_factory](./src/task_factory/contributing.md) 和 [trainer_factory](./src/trainer_factory/contributing.md) 的贡献请遵循各自的 `contributing.md` 文档。
## § Environment Setup

```bash
# 克隆仓库并进入目录
$ git clone <repo-url>
$ cd phm-vibench

# 创建虚拟环境并安装依赖
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

如需更多使用示例，请查阅 `README.md` 和 `doc/developer_guide.md`。
