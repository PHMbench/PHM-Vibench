# Goal: P08 TII — essential evidence, starting with the real-HSE interface

本 Goal 归属唯一实现仓 `PHMbench/PHM-Vibench`。沿用该仓 `docs/` 中实验 Goal 的位置，不恢复已移除的 Agent 工作区。论文为 `AI4Engineering-L/P08-HSE-Prompt-CDDG`，当前稿件 `paper/tex/main.tex`，判断入口 `paper/STATUS.md`。本文件是执行正文；论文仓只维护缺口表和链接。

核验基线：实现 `dev@89f3885a7c5e66c2ea17fa7677b9542bf03662d9`；论文 `dev@b1494d2f0443e46ab116cb6b0ffabafdc4a099f5`。执行时先查实际 dev、活跃 PR 和相关差异，不自动覆盖本地改动。

## Core claim and essential gap

C2 研究部署物理条件相对合格信号基线的诊断增量，融合结构优势另与同信息后融合比较。C1 的时间坐标分量和 C3 的条件路径解释不能替代真实诊断成绩。

**当前唯一获准立即执行的阶段为 G0：原有三项真实 HSE 接口资格测试。** 不通过时，后续训练、消融与基线成绩均不能按当前方法解释。通过仅解除接口阻塞。本批不启动训练、数据下载或完整五系统比较。

可复用：论文 A1–A5 与 `paper/experiments/results/conditioning_boundary.json` 的分析边界；原 `upstream_runtime_attempt.txt` 记录三项导入失败。旧交付的11项测试不构成真实 HSE 资格，合成诊断不构成工业结果。未访问的本地结果为 UNKNOWN，已有有效同协议结果应先检查再决定是否复用。

## Inputs and protocol

完整实现 checkout 必须包含 `src/model_factory/ISFM/embedding/E_01_HSE.py` 和其真实依赖。标准安装定义见根 `pyproject.toml` / `requirements.txt`；不要逐文件拼出假上游。

测试与调用方来自已交付的 `P08_TII_TPAMI_theory_and_code_20260907.zip` 内 `P08_research_revision_20260907/P4/`：

- `run.sh` 的 `test-runtime` 分支；
- `tests/test_upstream_runtime.py`；
- `p08/model.py` 中的 `Model`。

这份 P4 是已有的**接口测试输入和待迁移参考**，不是已被接纳的第二运行系统。不要在论文仓继续训练或维护一套新的 loader。后续实际算法、配置、评价与绘图只在 PHMFactory 的既有 owner 下接入；本批不执行该迁移，不更改子模块 gitlink。

三项测试保持现有种子3、7、23，各自对应坐标、detachment、共享初始化的确定性断言，不是三次性能重复。输入为测试已有的小张量，不读取工业数据。固定模型、输入、patch、共享组件；分别改变坐标所用采样信息或条件路径。

验收对象为实际导入、构造与前向语义；无精度指标、置信区间或方法胜出门槛。资源沿用原约束：CPU、单线程、三项一次有效执行；发现具体接口错误只允许最小修复后的受影响重测。无 GPU、API、超参数搜索或40×40训练。机器路径、联网、环境版本和内存实际可用性由本地填写，当前未知。

## Execution — G0 only

在真实工作区设置三个绝对路径环境变量。`P4_ROOT` 指向原交付，不是推测仓库中存在 P4。`P08_RUN_DIR` 为新的本地输出目录。

```bash
set -euo pipefail
: "${PHMFACTORY_ROOT:?Set the complete implementation checkout}"
: "${P4_ROOT:?Set the existing delivered P4 directory}"
: "${P08_RUN_DIR:?Set a new absolute local output directory}"
case "$P08_RUN_DIR" in /*) ;; *) echo 'P08_RUN_DIR must be absolute' >&2; exit 2;; esac
test -f "$PHMFACTORY_ROOT/src/model_factory/ISFM/embedding/E_01_HSE.py"
test -f "$P4_ROOT/tests/test_upstream_runtime.py"
test ! -e "$P08_RUN_DIR"
mkdir -p "$P08_RUN_DIR"
git -C "$PHMFACTORY_ROOT" status --short > "$P08_RUN_DIR/worktree.txt"
git -C "$PHMFACTORY_ROOT" rev-parse HEAD > "$P08_RUN_DIR/runtime_commit.txt"
```

先查看 `worktree.txt` 和当前分支。相关未提交代码必须记录、审查；不能宣称只有 commit 即完整描述代码版本。无需为无关脏文件停下，也不能自动 stash/reset/pull。已正确安装则复用；缺依赖时只在本地独立环境按上游执行一次 `python -m pip install -e "$PHMFACTORY_ROOT"`，失败记录后停止，不自动替换依赖或模型。

```bash
cd "$P4_ROOT"
# Isolate the intended upstream import root; do not mutate sys.modules.
PYTHONPATH="$PHMFACTORY_ROOT" python -c \
  'import inspect; from src.model_factory.ISFM.embedding.E_01_HSE import E_01_HSE; print(inspect.getfile(E_01_HSE))' \
  > "$P08_RUN_DIR/import_path.txt" 2> "$P08_RUN_DIR/import_stderr.txt"
# Inspect import_path.txt: it must resolve inside PHMFACTORY_ROOT.
set +e
PYTHONPATH="$PHMFACTORY_ROOT" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  bash run.sh test-runtime > "$P08_RUN_DIR/pytest.txt" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$P08_RUN_DIR/exit_code.txt"
if [ "$rc" -ne 0 ]; then
  printf 'Real-HSE qualification failed; see pytest.txt\n' > "$P08_RUN_DIR/failure.txt"
fi
exit "$rc"
```

命令依据已核验原文件；本设计阶段只运行了帮助解析，没有重新运行上述 HSE 测试。导入命令失败时 shell 会停止，原错误在 `import_stderr.txt`；本地补记依赖缺项，不伪造测试通过。

## Artifacts and validation

G0 保存上述版本、实际导入路径、pytest原始输出、退出码及必要失败说明，另记录 Python/PyTorch/相关包版本和CPU信息。不得存凭据或完整私有环境变量。三项均须实际构造并前向；skip、mock、替代 HSE 或只收集到测试均不合格。

逐项验收：索引模型不随仅用于坐标的采样率变化、HSE路径随其变化；detached条件在同一checkpoint下不改变输出；FiLM与token-CONCAT的共享组件初始化一致。目标标签、同信息后融合和原始数据入口不在这三项测试覆盖内。

## Required comparisons after G0 — NOT queued by this Goal

以下是投稿缺口的最小设计，不是当前可执行命令，也不授予训练预算。前置未解决前不运行。来源为论文 `paper/tex/sections/04_method.tex` 与 `05_experiments.tex`。

| ID / 不做的后果 | 比较、变量及指标 | 必须满足的运行前条件 |
|---|---|---|
| G1 数据与唯一执行入口；否则 C1/C2 的诊断成绩含义不确定 | 候选 CWRU、Ottawa-19、THU、JNU、HUST24；原始标签到共同物理类、预定通道、采样率单位、原始记录与物理单元、部署字段。核验源端至少一条真实 reader 输出；不据目标结果删系统/改类别 | 使用实际 PHMFactory reader、元数据与配置。数据绝对路径、各系统本体/单元/字段来源尚未绑定；没有自动导出器可广告为已完成。跨样本拟合只用源训练；按单元先分组再切窗 |
| G2 实用竞争力；否则 C2 不能支持实际增量 | HSE-only B1、P0、同字段后融合；近邻依论文定义作任务匹配适配，不冒称所有 DI-ERM 变体的官方复现。主指标记录级 macro-F1；估计每系统 P0−B1 与 P0−后融合的成对差；同源数据、信息、类空间与合理调优机会 | 各方法独立源验证选择；提前填写有限 trials、候选学习率/正则、训练上限、checkpoint与停止规则。源验证以等系统等物理单元 Brier；未知数据/算力下本批训练配额为0，不能拿40epochs默认当已调优。可比近期诊断参照的完整实现/权限未核验，缺口仍保留 |
| G3 必要拆分；否则采样时间与条件效应混合 | B0/B1/F01/P0 两因素，固定数据、处理、骨干、patch、抽样与训练/选择预算；分别报告时间主效应、条件主效应、交互，不预设交互为正 | 同一组源端选定设置用于匹配实验，不以各自最优配置充当严格拆分。只复用与该组设置完全一致的 G2 单元；无重复计数。最终种子集合在目标解封前按源端稳定性与资源冻结 |
| G4 条件内容诊断；否则不能把分支敏感性解释为物理内容作用 | 同一已选 P0 checkpoint 的 correct 与 detached，加一个有明确物理含义的错误/置换条件；必要时metadata-only排除字段—标签代理。报告成对分数、表示/logit变化分开。正确条件的性能不是从位移推出 | 不重新训练做冻结干预。供体、兼容性、权重只由来源知识/源训练确定；先各供体评分再按权重平均，不能先平均概率。不可识别则记录并收缩对应机制主张，不强制全部 I1–I9 |

**已定位的执行差距，不能用旧命令掩盖：** P4 `p08/run.py::execute` 在每次 `fit` 后立即读取目标记录、预测并评分；它没有独立的 source-only 调参阶段。`--suite interventions` 同样先训练，不是读取既有 checkpoint 的独立评价入口。当前 `fusion='concat'` 是token残差，不是后融合。未来接入唯一实现时复用现有训练、评价与checkpoint机制，补足明确的选择/冻结边界；不要用重复调用旧suite冒充盲调参或checkpoint复用。不能仅因目标成绩被计算就断言已发生泄漏，关键是阻止它参与选择。

### Statistical and decision contract for G2–G4

外层留出完整系统，源训练/验证按独立物理单元隔离；所有方法使用同一划分。窗概率先平均到原记录；物理单元为重采样聚类，抽中单元时带上全部记录。报告每系统效应和描述性均值。LOSO共享训练系统，不做假定各折独立的精确符号检验。种子是优化重复，不能替代设备数量。

本批G0没有性能重复。G2/G3重复次数尚未发布：必须在目标评价前写入冻结配置，并仅根据源训练随机性/预算确定；没有方差依据不强制统一≥3。最终报告优化重复范围，以及在固定模型下按目标物理单元配对重采样的区间，说明不包含新机器总体或重新训练不确定性。实际意义阈值须有诊断决策依据，不能套用旧0.01/4-of-5门槛；没有依据时仅报告效应与精度，不宣布工业显著性。

G2及以后的输出沿用所接纳runtime的原生产物；迁移参考的字段包括 `settings.json`、`protocol.json`、`checkpoint.pt`、`training.json`、`predictions.csv`、`metrics.csv`。失败前也保留实际配置和协议，不只在成功末尾写入；新目录不得覆盖已有失败。诊断数组与checkpoint可留原存储，dev保留必要小表和可访问引用。

有效负结果同样验收；后融合解释收益则收缩FiLM特异优势。缺数据/导入失败不是科学反证；区间宽为INCONCLUSIVE，不反复追加seed救结论。成本优势、多骨干、全部SOTA、更多数据集、全面噪声网格和原型方向诊断均未入队，除非新的核心主张需要它们。

## Failure and sync

缺完整checkout/原P4测试输入则 BLOCKED，列出精确缺项；不重造P4、不拼接loader、不合并旧研究分支。G0通过后交回G1，不自动解封G2。TII受阻不阻止独立TPAMI数值Goal。

在实现仓工作分支保存有效必要代码/小证据，按已有PR检查后合入dev，记录实际版本。论文仓只更新既有实验映射及STATUS中的下一步，不复制Goal或撰写Results。绘图只读取冻结的真实预测/指标；本批不画结果图，不调用模型来绘图。
