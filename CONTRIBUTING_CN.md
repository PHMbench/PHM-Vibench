# 为 PHM-Vibench 做贡献

<div align="center">
  <p>
    <a href="CONTRIBUTING.md">English</a> |
    <a href="CONTRIBUTING_CN.md"><strong>中文</strong></a>
  </p>
</div>

我们欢迎对 PHM-Vibench 的各种贡献！本指南将帮助您了解如何有效地为项目做出贡献。

## 目录

1. [快速开始](#快速开始)
2. [开发环境设置](#开发环境设置)
3. [贡献指南](#贡献指南)
4. [添加组件](#添加组件)
   - [新数据集](#新数据集)
   - [新模型](#新模型)
   - [新任务](#新任务)
   - [新流水线](#新流水线)
5. [配置系统](#配置系统)
6. [代码规范](#代码规范)
7. [测试](#测试)
8. [文档](#文档)
9. [Pull Request 流程](#pull-request-流程)

## 快速开始

### 前置要求

- Python 3.8+
- PyTorch 2.0+
- Git
- 对深度学习和时序分析的基本了解

### 架构概览

PHM-Vibench 使用**工厂设计模式**实现模块化扩展：

```
PHM-Vibench/
├── src/
│   ├── data_factory/      # 数据集加载与预处理
│   ├── model_factory/     # 模型（嵌入、骨干、任务头）
│   ├── task_factory/      # 训练逻辑与评估指标
│   └── trainer_factory/   # PyTorch Lightning Trainer 配置
├── configs/               # YAML 配置文件
└── docs/                  # 文档
```

详细架构说明请参阅 [`CLAUDE.md`](CLAUDE.md)。

## 开发环境设置

1. **Fork 并克隆仓库**
```bash
git clone https://github.com/your-username/PHM-Vibench.git
cd PHM-Vibench
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **运行测试**
```bash
python -m pytest test/
```

5. **离线冒烟测试**（无需下载数据）
```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml
```

## 贡献指南

### 贡献类型

我们欢迎以下类型的贡献：

1. **新组件**：数据集、模型、任务、流水线
2. **Bug 修复**：修复现有代码中的问题
3. **文档改进**：改进文档、示例和教程
4. **性能优化**：优化代码效率和性能
5. **配置添加**：新的实验配置和预设

### 开始之前

1. **检查现有 Issue**：查找相关问题或讨论
2. **创建 Issue**：对于新功能或重大变更
3. **先讨论**：对于重大变更，请先与维护者讨论
4. **阅读 CLAUDE.md**：了解项目架构和变更策略

### 核心设计原则

- **配置优先**：所有实验通过 YAML 配置定义
- **工厂模式**：注册组件，不要硬编码导入
- **单一真实来源**：更新注册表 → 图谱 → 文档
- **本科生能跑 + 博士生能改**：保持易用且可扩展

## 添加组件

### 新数据集

详细指南请参阅 [`src/data_factory/contributing.md`](src/data_factory/contributing.md)。

**快速步骤**：
1. 在 `src/data_factory/dataset_task/` 中创建数据集类
2. 在 `src/data_factory/dataset_task/__init__.py` 中注册
3. 在 `data/metadata.xlsx` 中添加元数据条目
4. 在 `configs/base/data/` 中创建配置

### 新模型

详细指南请参阅 [`src/model_factory/contributing.md`](src/model_factory/contributing.md)。

**模型组件遵循注册表风格的 ID**：
- 嵌入：`E_**_*`
- 骨干网络：`B_**_*`
- 任务头：`H_**_*`

**快速步骤**：
1. 创建模型类，使用 NumPy 风格的文档字符串
2. 在相应的 `__init__.py` 中注册
3. 在 `configs/demo/` 中添加配置预设

### 新任务

**任务类型**（通过 `task.type` + `task.name` 选择）：
- `DG`：域泛化
- `CDDG`：跨数据集域泛化
- `FS`/`GFS`：少样本学习 / 广义少样本学习
- `ID`：基于 ID 的数据加载
- `MT`：多任务学习

**快速步骤**：
1. 在 `src/task_factory/task/<TYPE>/` 中创建任务
2. 继承基础任务类
3. 在 `src/task_factory/task/<TYPE>/__init__.py` 中注册

### 新流水线

流水线按固定顺序组装工厂：
1. 加载配置
2. 构建数据
3. 构建模型
4. 构建任务
5. 构建训练器

**快速步骤**：
1. 创建 `src/Pipeline_<name>.py`
2. 通过 YAML 选择：`pipeline: <name>`

## 配置系统

PHM-Vibench 使用 **v5.x 五模块配置模型**：
- `environment` / `data` / `model` / `task` / `trainer`

### 添加新配置

1. **创建配置 YAML**：在 `configs/demo/` 或 `configs/experiments/` 中
2. **添加到注册表**：更新 `configs/config_registry.csv`
3. **重新生成图谱**：`python -m scripts.gen_config_atlas`
4. **验证**：`python -m scripts.validate_configs`

### 配置组合规则（优先级从低到高）

1. `base_configs.*` YAML 文件
2. Demo YAML 自身的模块覆盖
3. 可选的本地覆盖 `configs/local/local.yaml`
4. CLI `--override key=value`

### 配置检查工具

```bash
# 查看解析后的配置 + 来源 + 目标
python -m scripts.config_inspect --config <yaml> --override key=value

# 验证所有配置
python -m scripts.validate_configs

# 生成 CONFIG_ATLAS.md
python -m scripts.gen_config_atlas
```

## 代码规范

### Python 代码风格

遵循 PEP 8 规范，并做以下调整：

1. **行长度**：最多 100 字符
2. **导入顺序**：分组导入（标准库、第三方、本地）
3. **命名规范**：
   - 类：`PascalCase`
   - 函数/变量：`snake_case`
   - 常量：`UPPER_CASE`

### 文档字符串标准

使用 NumPy 风格的文档字符串：

```python
def function_name(param1: int, param2: str = "default") -> bool:
    """函数简要描述。

    更详细的描述，说明函数的用途和行为。

    Parameters
    ----------
    param1 : int
        param1 的描述
    param2 : str, optional
        param2 的描述（默认："default"）

    Returns
    -------
    bool
        返回值的描述

    Raises
    ------
    ValueError
        当 param1 为负数时抛出

    Examples
    --------
    >>> result = function_name(5, "test")
    >>> print(result)
    True
    """
```

## 测试

### 测试结构

```
test/
├── test_end_to_end_integration.py
├── test_parameter_consistency.py
└── ...
```

### 运行测试

```bash
# 运行所有维护的测试
python -m pytest test/

# 运行特定测试文件
python -m pytest test/test_parameter_consistency.py

# 运行测试并生成覆盖率报告
python -m pytest test/ --cov=src --cov-report=html
```

### 编写测试

```python
import pytest
import torch
from argparse import Namespace

class TestYourComponent:
    """YourComponent 的测试套件。"""

    @pytest.fixture
    def config(self):
        """测试用的配置。"""
        return Namespace(
            param1="value1",
            param2=42
        )

    def test_basic_functionality(self, config):
        """测试基本功能。"""
        # 准备
        component = YourComponent(config)

        # 执行
        result = component.method()

        # 断言
        assert result is not None
```

## 文档

### 文档要求

1. **API 文档**：NumPy 风格的文档字符串
2. **使用示例**：可运行的代码示例
3. **配置文档**：更新注册表和图谱
4. **双语支持**：英文和中文（`_CN.md` 后缀）

### 文档结构

```
PHM-Vibench/
├── README.md / README_CN.md           # 主项目 README
├── CONTRIBUTING.md / CONTRIBUTING_CN.md  # 本文件
├── CLAUDE.md                           # 架构和变更策略
├── AGENTS.md                           # 开发命令手册
├── configs/README.md                   # 配置系统指南
├── docs/
│   ├── CONFIG_ATLAS.md                 # 生成的配置参考
│   ├── developer_guide.md
│   └── testing.md
└── src/
    ├── data_factory/README_CN.md
    ├── model_factory/README_CN.md
    └── task_factory/README_CN.md
```

## Pull Request 流程

### 提交之前

1. **运行测试**：确保所有测试通过
2. **检查代码风格**：遵循代码规范
3. **更新文档**：添加/更新相关文档
4. **添加测试**：为新功能包含测试
5. **更新注册表**：对于新配置，更新 `config_registry.csv`

### PR 模板

```markdown
## 描述
变更的简要描述和动机。

## 变更类型
- [ ] Bug 修复
- [ ] 新功能
- [ ] 破坏性变更
- [ ] 文档更新
- [ ] 配置添加

## 测试
- [ ] 本地测试通过
- [ ] 为新功能添加了测试
- [ ] 配置已验证（如适用）

## 文档
- [ ] 代码遵循风格指南
- [ ] 已完成自审
- [ ] 文档已更新
- [ ] 注册表/图谱已更新（如适用）
```

### 审查流程

1. **自动检查**：CI/CD 管道运行测试和验证
2. **代码审查**：维护者审查代码质量和设计
3. **文档审查**：检查文档是否完整准确
4. **批准**：至少需要一名维护者批准

### 批准后

1. **压缩并合并**：我们通常会压缩提交
2. **更新变更日志**：维护者更新变更日志
3. **发布说明**：重大变更将包含在发布说明中

## 必要的变更顺序

对于配置相关的变更：
1. 注册表 → 2. 图谱 → 3. 检查 → 4. 模式验证 → 5. README → 6. CI/测试

## 社区准则

### 行为准则

- 保持尊重和包容
- 专注于建设性反馈
- 帮助新人学习和贡献
- 保持专业的沟通方式

### 获取帮助

- **GitHub Issues**：报告 Bug 和功能请求
- **Discussions**：提问和一般讨论
- **CLAUDE.md**：了解架构和变更策略
- **AGENTS.md**：开发命令参考

## 联系方式

- **维护者**：[Qi Li](https://github.com/liq22)、[Xuan Li](https://github.com/Xuan423)
- **GitHub**：[PHM-Vibench 仓库](https://github.com/PHMbench/PHM-Vibench)

感谢您为 PHM-Vibench 做出贡献！
