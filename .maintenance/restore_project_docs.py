"""One-time homepage restoration; not part of the PHMFactory runtime."""
from pathlib import Path


def insert_before(path: str, anchor: str, text: str) -> None:
    file = Path(path)
    old = file.read_text(encoding="utf-8")
    if old.count(anchor) != 1:
        raise ValueError(f"Expected one insertion point in {path}: {anchor}")
    file.write_text(old.replace(anchor, text.rstrip() + "\n\n" + anchor), encoding="utf-8")


for directory in (
    "phmfactory", "configs", "src/data_factory", "src/model_factory",
    "src/task_factory", "src/trainer_factory", "src/runtime", "data", "test",
    "apps/streamlit", "docs", "doc/changelog", "paper/project",
):
    if not Path(directory).is_dir():
        raise FileNotFoundError(directory)

insert_before("README.md", "## Runtime structure", """## Project structure

```text
PHM-Vibench/
├── phmfactory/           # Public commands and configuration entrypoint
├── configs/              # Demo and research experiment configurations
├── src/
│   ├── data_factory/     # Readers, datasets, sampling and loaders
│   ├── model_factory/    # Models and representation modules
│   ├── task_factory/     # Objectives, metrics and optimization
│   ├── trainer_factory/  # Devices, callbacks and model selection
│   └── runtime/          # Experiment execution
├── data/                 # Bundled Dummy data and data-layout guide
├── test/                 # Runtime and component tests
├── apps/streamlit/       # Optional browser workspace
├── docs/                 # User and developer guides
├── doc/changelog/        # Upgrade notes
└── paper/project/        # Research source and migration notes
```

Start with `configs/` to run an experiment and the relevant Factory to add a
component. Run outputs are written to the paths printed by the command, not to a
fixed directory implied by this tree.
""")
insert_before("README_CN.md", "## 运行结构", """## 项目结构

```text
PHM-Vibench/
├── phmfactory/           # 公共命令与配置入口
├── configs/              # 示例和研究实验配置
├── src/
│   ├── data_factory/     # 数据读取、数据集、采样与加载
│   ├── model_factory/    # 模型与表示模块
│   ├── task_factory/     # 目标函数、指标与优化策略
│   ├── trainer_factory/  # 设备、回调与模型选择
│   └── runtime/          # 实验执行
├── data/                 # 内置 Dummy 数据与数据布局说明
├── test/                 # 运行路径与组件测试
├── apps/streamlit/       # 可选浏览器工作区
├── docs/                 # 用户与开发文档
├── doc/changelog/        # 升级记录
└── paper/project/        # 研究源码与迁移说明
```

运行实验从 `configs/` 开始；新增组件从相应 Factory 开始。实验结果以命令返回的实际路径
为准，目录树不规定固定的结果保存位置。
""")
insert_before("README.md", "## Citation and license", """## Publications and research

### Project paper

Qi Li, Bojian Chen, Xuan Li, Qitong Chen, Liang Chen, Changqing Shen, Lu Lu,
Zhaoye Qin, and Fulei Chu.
**[PHM-Vibench: A Unified and Factory-Style Vibration Benchmarking Framework for the Foundation Model Era](https://papers.phmsociety.org/index.php/phmap/article/view/4303)**.
*PHM Society Asia-Pacific Conference*, 5(1), 2025 proceedings;
published online January 13, 2026. DOI: [10.36001/phmap.2025.v5i1.4303](https://doi.org/10.36001/phmap.2025.v5i1.4303).

The paper describes PHM-Vibench. For capabilities of the current PHMFactory source,
use the [supported combinations](SUPPORTED_COMBINATIONS.md) and [known limitations](KNOWN_LIMITATIONS.md).

### Related method

Qi Li, Bojian Chen, Qitong Chen, Xuan Li, Zhaoye Qin, and Fulei Chu.
**[HSE: A plug-and-play module for unified fault diagnosis foundation models](https://doi.org/10.1016/j.inffus.2025.103277)**.
*Information Fusion*, 123, 103277, 2025.

HSE is listed as a related representation method, not as a claim that all published
experiments used the current software. In-progress work and historical paper sources
are separate from published results; see [research source notes](paper/project/README.md).

### Research using PHMFactory

To add a study, [open an issue](https://github.com/PHMbench/PHM-Vibench/issues) with its
publication link, code or experiment configuration, and the software version used.
Only studies with a documented use of this project belong in this category.

## Roadmap

| Stage | Focus |
| --- | --- |
| Available | Configuration-first CLI, offline Dummy first run, and direct result paths |
| Next | Complete declared metrics and result semantics; requalify the real-data reference experiment |
| Research | Evaluate interpretable-model explanation and heterogeneous-signal extensions before promoting them to maintained examples |

[Upgrade notes](doc/changelog/) record completed changes.
[Release readiness](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md) records current blockers.
Research candidates are not release promises.

## Contributors and community

Core project contributors include [Qi Li](https://github.com/liq22) and
[Xuan Li](https://github.com/Xuan423). See [all contributors](https://github.com/PHMbench/PHM-Vibench/graphs/contributors)
for the full contribution history.

Use [Issues](https://github.com/PHMbench/PHM-Vibench/issues) for reproducible bugs and
bounded feature proposals, and [Discussions](https://github.com/PHMbench/PHM-Vibench/discussions)
for usage questions and research ideas. Contributions follow the
[contribution guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).

[Star history](https://www.star-history.com/#PHMbench/PHM-Vibench&Date)
""")
insert_before("README_CN.md", "## 引用与许可", """## 论文与研究

### 项目论文

Qi Li, Bojian Chen, Xuan Li, Qitong Chen, Liang Chen, Changqing Shen, Lu Lu,
Zhaoye Qin, Fulei Chu.
**[PHM-Vibench: A Unified and Factory-Style Vibration Benchmarking Framework for the Foundation Model Era](https://papers.phmsociety.org/index.php/phmap/article/view/4303)**.
*PHM Society Asia-Pacific Conference*, 5(1)，2025 年会议论文集；
在线发表日期为 2026 年 1 月 13 日。DOI：[10.36001/phmap.2025.v5i1.4303](https://doi.org/10.36001/phmap.2025.v5i1.4303)。

该论文介绍 PHM-Vibench。当前 PHMFactory 源码的能力范围，以
[支持组合](SUPPORTED_COMBINATIONS.md)和[已知限制](KNOWN_LIMITATIONS.md)为准。

### 相关方法

Qi Li, Bojian Chen, Qitong Chen, Xuan Li, Zhaoye Qin, Fulei Chu.
**[HSE: A plug-and-play module for unified fault diagnosis foundation models](https://doi.org/10.1016/j.inffus.2025.103277)**.
*Information Fusion*, 123, 103277, 2025。

HSE 在此列为相关表示方法，不表示论文的全部实验使用了当前软件版本。在研项目和历史
论文源码不属于已发表结果，相关说明见[研究源码入口](paper/project/README.md)。

### 使用本项目的研究

收录研究时，请通过 [Issue](https://github.com/PHMbench/PHM-Vibench/issues) 提供论文链接、
代码或实验配置，以及实际使用的软件版本。本栏目只收录具有明确项目使用关系的研究。

## 项目路线图

| 阶段 | 内容 |
| --- | --- |
| 已有能力 | 配置驱动 CLI、离线 Dummy 首跑、直接结果路径 |
| 下一步 | 补齐声明指标与结果语义，重新验证真实数据参考实验 |
| 研究方向 | 验证可解释模型的语言解释与异构信号扩展，再决定是否纳入维护示例 |

[升级记录](doc/changelog/)说明已完成的修改，
[发布状态](docs/PHMFACTORY_V0_3_RELEASE_READINESS.md)说明当前阻塞。研究方向不代表发布承诺。

## 贡献者与社区

项目核心贡献者包括 [Qi Li](https://github.com/liq22) 和
[Xuan Li](https://github.com/Xuan423)。完整贡献历史见
[全部贡献者](https://github.com/PHMbench/PHM-Vibench/graphs/contributors)。

可复现问题与具体功能建议请提交 [Issue](https://github.com/PHMbench/PHM-Vibench/issues)，
使用问题与研究讨论请前往 [Discussions](https://github.com/PHMbench/PHM-Vibench/discussions)。
参与开发前请阅读[贡献指南](CONTRIBUTING_CN.md)和[行为准则](CODE_OF_CONDUCT.md)。

[Star 历史](https://www.star-history.com/#PHMbench/PHM-Vibench&Date)
""")
paper = Path("paper/project/README.md")
old = paper.read_text(encoding="utf-8")
anchor = "Positive claims\nstill require fresh, hash-bound runs recorded by the corresponding paper\nrepository."
if old.count(anchor) != 1:
    raise ValueError("Paper source note changed; inspect before replacing")
paper.write_text(old.replace(anchor, "Performance and explanation claims require current-source experiments recorded\nby the corresponding paper repository: exact configuration, data population,\nsplit, selected checkpoint, estimators, seeds, and results. A source snapshot\nor a valid identifier alone does not establish a scientific result."), encoding="utf-8")
Path("doc/changelog/2026-09-07-project-homepage.md").write_text("""# Project homepage recovery — 2026-09-07

The English and Chinese homepages now show the current directory layout, the
published PHM-Vibench project paper, the related HSE method, a capability-based
roadmap, and contributor/community links. Published work, studies using the
software, and in-progress research are separated.

The project paper belongs to the 2025 proceedings and was published online on
January 13, 2026. Software citation remains in `CITATION.cff`; the homepage does
not invent a release date, DOI, author list, or benchmark result for PHMFactory.

The paper source note no longer requires hash-bound runs. It identifies the
configuration, data, split, checkpoint, estimators, seeds, and actual results
needed to evaluate a claim.

No installation command, experiment configuration, training code, data protocol,
or release status changed. Old placeholder publications, unverified chat invites,
and the credential-bearing Star History embed were not restored.
""", encoding="utf-8")
print("Restored four documentation files; runtime and experiment configuration unchanged.")
