# 贡献者指引

*欢迎为Vbench提供Feature PR、Bug反馈、文档补充或其他类型的贡献！*

## 目录

- [代码规约](#-代码规约)
- [贡献流程](#-贡献流程)
- [环境配置](#-环境配置)

## 📖 代码规约

请查看我们的[代码规约文档](./CODE_OF_CONDUCT.md)。

## 🔁 贡献流程

### 我们需要什么
- 新模型与数据集：Vbench需要支持更多的故障诊断模型和工业设备数据集，如果您有相关资源或实现，可以提交PR给我们。
- 新功能模块：我们欢迎对训练流程、数据处理或评估方法的改进，特别是能提高模型性能或易用性的功能。
- 文档与教程：如果您擅长技术写作，欢迎帮助我们完善文档或提供示例教程，帮助用户更好地使用Vbench。
- Bug修复：如果您发现了问题或有改进建议，请提交Issue或直接提供修复代码。

### 开发规范

#### 代码风格
- 变量命名采用下划线分隔的命名方式（如`model_name`），类名采用首字母大写的驼峰命名法（如`BaseModel`）
- Python代码缩进统一使用4个空格
- 每个函数和类都应有清晰的文档字符串，说明其功能、参数和返回值
- 复杂逻辑需要添加注释，使代码易于理解

#### 模块组织
- 新的模型应放在`model_factory`目录下，并在`__init__.py`中注册
- 新的数据集应放在`data_factory`目录下，并在`__init__.py`中注册
- 新的任务类型应放在`task_factory`目录下，并在`__init__.py`中注册
- 工具函数应放在`utils`目录下合适的模块中

### 提交PR（Pull Requests）

1. **Fork**：将Vbench代码库fork到您的个人账户
2. **Clone**：将您fork的代码库clone到本地并创建新的分支进行开发
3. **开发**：按照开发规范进行代码编写，并添加必要的测试用例
4. **测试**：使用`test/test.ipynb`或`main_dummy.py`对您的代码进行测试
5. **提交PR**：开发测试完成后，提交Pull Request到主分支
6. **描述**：在PR中详细描述您的修改内容、解决的问题及测试结果
7. **Review**：等待维护者审核您的代码并根据反馈进行修改

### 测试流程

在提交PR前，请确保您的代码通过了以下测试：

```shell
# 使用main_dummy.py测试特定模块
python main_dummy.py --module <您修改的模块名称>

# 或使用全面测试
python main_dummy.py --all_modules

# 使用Jupyter Notebook进行交互式测试
jupyter notebook test/test.ipynb
```

## 🔧 环境配置

### 依赖安装

```shell
pip install -r requirements.txt
```

### 目录结构
确保您了解Vbench的目录结构：
- `configs/`: 配置文件目录
- `src/`: 源代码目录
  - `data_factory/`: 数据集实现
  - `model_factory/`: 模型实现
  - `task_factory/`: 任务定义
  - `trainer_factory/`: 训练器实现
  - `utils/`: 工具函数
- `test/`: 测试代码
- `results/`: 实验结果存储
- `data/`: 数据存储

### 开发建议
- 尽量保持代码模块化和可扩展性
- 添加新功能前，先查看现有实现，避免重复造轮子
- 确保您的代码具有良好的可读性和注释
- 为新功能添加示例配置文件和使用说明

# Contributor Covenant Code of Conduct

## Our Pledge

We as members, contributors, and leaders pledge to make participation in our
community a harassment-free experience for everyone, regardless of age, body
size, visible or invisible disability, ethnicity, sex characteristics, gender
identity and expression, level of experience, education, socio-economic status,
nationality, personal appearance, race, religion, or sexual identity
and orientation.

We pledge to act and interact in ways that contribute to an open, welcoming,
diverse, inclusive, and healthy community.

## Our Standards

Examples of behavior that contributes to a positive environment for our
community include:

* Demonstrating empathy and kindness toward other people
* Being respectful of differing opinions, viewpoints, and experiences
* Giving and gracefully accepting constructive feedback
* Accepting responsibility and apologizing to those affected by our mistakes,
  and learning from the experience
* Focusing on what is best not just for us as individuals, but for the
  overall community

Examples of unacceptable behavior include:

* The use of sexualized language or imagery, and sexual attention or
  advances of any kind
* Trolling, insulting or derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information, such as a physical or email
  address, without their explicit permission
* Other conduct which could reasonably be considered inappropriate in a
  professional setting

## Enforcement Responsibilities

Community leaders are responsible for clarifying and enforcing our standards of
acceptable behavior and will take appropriate and fair corrective action in
response to any behavior that they deem inappropriate, threatening, offensive,
or harmful.

Community leaders have the right and responsibility to remove, edit, or reject
comments, commits, code, wiki edits, issues, and other contributions that are
not aligned to this Code of Conduct, and will communicate reasons for moderation
decisions when appropriate.

## Scope

This Code of Conduct applies within all community spaces, and also applies when
an individual is officially representing the community in public spaces.
Examples of representing our community include using an official e-mail address,
posting via an official social media account, or acting as an appointed
representative at an online or offline event.

## Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be
reported to the community leaders responsible for enforcement at
[INSERT CONTACT METHOD].
All complaints will be reviewed and investigated promptly and fairly.

All community leaders are obligated to respect the privacy and security of the
reporter of any incident.

## Enforcement Guidelines

Community leaders will follow these Community Impact Guidelines in determining
the consequences for any action they deem in violation of this Code of Conduct:

### 1. Correction

**Community Impact**: Use of inappropriate language or other behavior deemed
unprofessional or unwelcome in the community.

**Consequence**: A private, written warning from community leaders, providing
clarity around the nature of the violation and an explanation of why the
behavior was inappropriate. A public apology may be requested.

### 2. Warning

**Community Impact**: A violation through a single incident or series
of actions.

**Consequence**: A warning with consequences for continued behavior. No
interaction with the people involved, including unsolicited interaction with
those enforcing the Code of Conduct, for a specified period of time. This
includes avoiding interactions in community spaces as well as external channels
like social media. Violating these terms may lead to a temporary or
permanent ban.

### 3. Temporary Ban

**Community Impact**: A serious violation of community standards, including
sustained inappropriate behavior.

**Consequence**: A temporary ban from any sort of interaction or public
communication with the community for a specified period of time. No public or
private interaction with the people involved, including unsolicited interaction
with those enforcing the Code of Conduct, is allowed during this period.
Violating these terms may lead to a permanent ban.

### 4. Permanent Ban

**Community Impact**: Demonstrating a pattern of violation of community
standards, including sustained inappropriate behavior, harassment of an
individual, or aggression toward or disparagement of classes of individuals.

**Consequence**: A permanent ban from any sort of public interaction within
the community.

## Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage],
version 2.0, available at
[https://www.contributor-covenant.org/version/2/0/code_of_conduct.html][v2.0].

Community Impact Guidelines were inspired by [Mozilla's code of conduct
enforcement ladder][mozilla ladder].

For answers to common questions about this code of conduct, see the FAQ at
[https://www.contributor-covenant.org/faq][faq]. Translations are available
at [https://www.contributor-covenant.org/translations][translations].

[homepage]: https://www.contributor-covenant.org
[v2.0]: https://www.contributor-covenant.org/version/2/0/code_of_conduct.html
[mozilla ladder]: https://github.com/mozilla/diversity
[faq]: https://www.contributor-covenant.org/faq
[translations]: https://www.contributor-covenant.org/translations
