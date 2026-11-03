# G08 — P01 投稿必需的独立诊断证据

本文件是 P01 当前唯一执行正文。论文仓只保留链接和贡献—证据表，不复制本 Goal。对应稿件为 `AI4Engineering-L/P01-UXFD-Multimodal-Alignment` 的 `paper/draft/`，`paper/tex` 是其生成的 IEEE 格式版本。目标期刊仍为 IEEE TII。

## 核心主张、缺口及本批范围

问题：完整 TSPN 上的分支算子残差 I 是否优于合格参照与非线性重拟合；新增包络描述量是否提供必要增量；导出的分支项是否忠实解释实际类间 log-odds 变化？

已有：模型、描述量、源资格检查、初始参照候选、概率/分支导出和软件测试已实现。原 PU D1/G07 已完成，但属于旧 replacement-head，不支持新 I 的工业效果；原七个测试轴承保持封存。只新增以下投稿必需证据，不重复旧实验或搭建第二套运行系统。

| 必需项 | 不做时缺失的核心结论 | 最小比较 / 产物 |
|---|---|---|
| 实际诊断增益 | 无法声称 I 改善诊断 | I 对同一 p0、MLP16(F)；准确率/Brier/F1及物理组配对不确定性 |
| 描述量消融 | 无法确认新增描述量的实用作用 | I 对同一残差结构去掉 envelope.diagnostics 的 I-base |
| 忠实解释 | 物理名称不能证明解释对应预测 | 实际导出分支和重构单窗口 log-odds；算子支持及参数 |
| 最近工业方法 | 无法支撑与现有可解释诊断工作的差异 | 优先合法 FIRNet；不可得时有明确原文映射的 TON-based adapter，并收缩 FIRNet-specific 优越性主张 |

I-F、dense same-input MLP、更多数据集、全部 loss/routing、完整物理因果归因、延迟优势和新的分布无关理论均为可选；本批不自动执行。若后续要声明相应独立机制/效率/安全保证，再提出具体缺口，不在本轮扩大列表。

## Inputs and protocol

实际工作目录是 Benchmark checkout 根目录。开始先只读核对 branch/status/HEAD、相关 PR、父仓 gitlink 和 Python imports；保留无关工作、image.png、src/vibench/ 和全部历史产物。不要 reset/clean/自动 stash/强推。

数据入口已知且只读：

```text
/home/user/data/PHMbenchdata/PHM-Vibench/README.md
/home/user/data/PHMbenchdata/PHM-Vibench/metadata.xlsx
```

读取它们及本地已有配置/产物，解析真实 H5、sheet/列、Id、class map、channel/layout、采样、窗口、物理试件或独立 run、工况和曝光历史。不得猜路径、复制 loader、改标签或把窗口/Id当独立试件。先全局物理组划分，再取窗口；已揭盲 PU 的其他窗口或工况不是新独立数据。

优先绑定一个真正独立的新任务，不强制若干数据集。任务集合、源训练/选择/最终测试组、类支持、基线和候选的有限开发预算，必须在查看增强结果或新 test 前写入既有 run 配置/说明。无法确认独立队列就停止该任务的拟合，继续可以独立完成的接口/最近邻依据核验；不从旧 test 挑有利子集。

p0 使用现有 `experiments.p01.train_source_classifier --role reference` 或复用同协议合格 TSPN checkpoint，保留完整配置、温度及开发历史。每个声明的 source-selection condition 准确率至少 .80，同时报告全部类别支持、混淆矩阵和 macro-F1。该点估计是开发资格，不是独立保证。不得按 test 分数选择任务；最终基线低于 .80 仍报告，不删任务。

固定当前结构：`operator_residual`、hidden=16、cap=5、ordinary paired CE+.25 Brier、tau=1、mean_source、lambda_delta=0。频带、lag和epsilon从真实源观测及机理约束确定；示例512点/lag不是数据事实。配对仅在实际标签保持成立时使用；循环时间起点变化不得称转速增强。

本批 I、I-base、MLP16 使用固定 seeds 42/123/456（已有 PU 训练波动足以要求这些核心比较重复），不是所有实验的统一 seed 门槛。同一 p0 只训练/计费一次，不当成三个独立参照。解释核验复用三臂已有概率，不新增拟合。最近邻重复数和理由在运行前固定；若只承担有限预算比较，允许不同重复数，但不得混成相同训练总体或只报告好 seed。

基线与各候选采用同一预声明的有限源搜索上限、优化更新/选择机会和观测。已有旧 PU 的20×50更新不自动等于新任务充分收敛。先用实际可复用配置确定预算并保留曲线；任何小型源调参只能在共同声明的上限内。不得因分数不好单臂延长或继续加新配置。参数量、参照预训练资源和额外初始候选分别披露。

主要效果为 I−p0、I−MLP16、I−I-base 的组平衡 acquisition accuracy 差与 window Brier 差，准确率正值、Brier负值有利；macro-F1与class支持单列。物理组是统计重采样单位；同组跨条件共同重采样、两臂共享索引。固定三seed的有限均值、seed SD、条件于冻结模型的描述性组配对95%区间分开报告。复用既有组聚合/重采样函数，沿用2000次、analysis seed20260919；原 analyze_d1 的 CORE/CONTRASTS 固定为旧五臂，需按下述接口步骤显式适配本批清单，不声称旧CLI已支持新差值。核心比较全部呈现，不用“任一有利指标”决定成功；不将区间跨零写成等价。

分布无关准确率/Brier确认报告 `diagnostic_gain` 可作为额外证据，不是经验方法论文的通用合并或运行门槛。若确实使用它声明保证，误差族、候选/seed/条件/数据集分配及独立抽样假设必须预先固定，不能看结果后换半径。

## Execution

1. 读取当前模型/loss/数据接口和已有完整运行，复用兼容产物；记录实际 code commit。不要从过往交接的 commit 名字推断最新能力。
2. 先完成输入绑定和预声明预算。用现有 `preflight_fusion` 做 metadata/H5 keys/算子支持检查，不加载保护 test 波形。配置依现有 schema，缺失字段不靠自动值修复。
3. 核对既有 `test/test_p01_interpretable.py`、`test/test_p01_interpretable_runtime.py` 的受影响路径。模型主体不要重实现。当前原 `frozen_export` 的旧15槽位/K4入口不适用新矩阵；本地如无已验证的新入口，只给原 `load_model/predict_records/save_bundle/verify_vectors` 加本批清单适配，不复制 loader/evaluator、不重写旧 D1 合同。同时为原 analyze_d1 的旧 CORE/CONTRASTS 增加本批显式清单适配，复用 load_artifact、组聚合及配对重采样，不复制评价器。用构造数组验证 I−p0、I−MLP16、I−I-base 都被实际输出，p0只是一条共享参照，缺臂明确报错，旧D1默认行为不变。这两项适配应先验证，再启动拟合。
4. 建立/复用 p0，查看 source 资格与曲线。失败时保存记录，不开始该任务的候选训练；继续其他已预声明的独立必需工作。资格不达标并不授权无上限搜索。
5. 生成三种真实模型配置：I 保留完整分支及 diagnostics；I-base 只移除新 diagnostics；MLP16 使用原 mlp、branches=[]、reference features=true。I与I-base同类输出零初始化且参照候选保留。维数/参数数目可能不同，如实记录，不称纯因果信息消融。
6. 直接调用现有 trainer，传 `--reference-min-accuracy 0.8`。不要通过未传播此参数的旧 replicate/study wrapper 假定已经资格检查。示例下方变量必须来自实际冻结配置。
7. source完备后验证每个模型的严格保存/新进程恢复，核对 reference参数/buffer与原概率不变，以及分支项对类间log-odds的精确重构。保留 retained checkpoint -1，这是无新增源增量的合法结果。不得在解释图里把它写成改善。
8. 冻结本批全部函数、比较角色、统计定义和数据分区，完成最近邻状态决定后统一释放新test。每个函数只导出一次完整预测；中断仅恢复同一冻结函数的缺失输出，不选另一模型、不改seed。已在本批范围内的缺失核心臂阻塞完整测试结论，不按幸存目录生成“完整表”。
9. 最近邻优先获取合法 FIRNet 方法/实现。失败可使用原文充分可核对的 TON-based adapter，明确哪些结构/预算经过适配；不能把现有 TSPN 文件存在视为忠实 TON 复现。不得声称胜过未执行的 FIRNet。若替代仍无法支撑关键差异，继续标缺口；不无故阻塞可独立完成的三臂源拟合。
10. 分析/绘图只读保存的概率、分支贡献和统计表，不打开 H5/checkpoint或调用模型。解释是 direct单窗口log-odds；采集平均或中间alpha混合必须按保存概率另算，不线性套用分支项。完成本批后停止，不自动扩数据集或方法。

### 已核验的源训练接口

先在真实 Benchmark 根目录与 LQ_signal 环境核对 `--help`。下面不假装已绑定数据或预算。

```bash
conda activate LQ_signal
: "${MODEL_CFG:?实际模型配置}"
: "${DATA_CFG:?实际数据配置}"
: "${DATASET:?真实dataset name}"
: "${RUN_OUT:?本次未存在的输出目录}"
: "${SEED:?固定seed}"
: "${ARM:?固定臂名 I 或 I-base 或 MLP16}"
: "${EPOCHS:?预声明预算}"
: "${STEPS:?预声明预算}"
: "${PAIR_SHIFT:?标签保持的偏移}"
CUDA_VISIBLE_DEVICES=0 python -m experiments.p01.train_tspn_fusion_v2 \
  --model-config "$MODEL_CFG" --data-config "$DATA_CFG" --dataset "$DATASET" \
  --output "$RUN_OUT" --device cuda:0 --seed "$SEED" --arm "$ARM" \
  --epochs "$EPOCHS" --steps-per-epoch "$STEPS" --units-per-domain 2 \
  --lr 0.001 --pair-shift "$PAIR_SHIFT" \
  --selection-predictor candidate --selection-brier-weight 0.25 \
  --reference-min-accuracy 0.8
```

默认物理 GPU0；GPU2 禁止，训练不自动换卡或回退CPU。CPU读取/分析和软件测试不属于训练fallback。不存在的“全候选”CLI不在本文伪造；导出和分析的少量清单适配是运行前待核验接口，不是已经完成的实验。arm从训练、checkpoint到NPZ及分析清单必须完全一致，不能训练后改名掩盖语义错误。

## Artifacts and validation

复用 command.json、model/data config、development_groups.csv、initial/selected state、reference_qualification.json、selected_source_validation_windows.npz、sampling/training表、result_scope.json、导出清单与恢复日志。保留每个臂/seed的 raw/candidate概率、稳定log-prob、class names、group/acquisition/window/domain IDs、logit_contribution__*、组/条件指标、成对差和不确定性、参数/时间及失败记录。不用概率截断伪造有限CE。

运行有效性看：真实输入及物理组一致、没有保护集访问/调参、冻结p0、比较实际配置/预算一致、三臂输出完整、权重归约正确、解释重构成立。I-base缺描述量也改变参数量，作为限制保留。验收不要求正增益；基础设施失败与有效负结果分开。不要用软件PASS代替80%或工业效果。

结果回填：论文现有 `paper/experiments/evidence_matrix.md` 的当前模型必需行、稿件方法后的独立新模型结果。原PU/G07证据章节仍归旧结构。这里只交付产物，不自行写Results或重新决定论文题目/贡献。

## Failure and sync

被数据、GPU、方法许可或依赖阻塞时，记录具体原因并继续本批独立必需任务。协议不变的环境修复可继续；需要改预算/划分/方法时明确交回科学设计，不静默替换。失败目录与旧结果不删、不覆盖；不反复运行到分数好看。

实现、配置、评价和图源码只在Benchmark。按topic branch/PR和现有检查正常同步dev，父仓只更新已可取回的依赖和证据引用。明确stage相关文件，不git add -A，不强推，不夹带历史Draft。大产物沿用授权存储，公开预测/标识/checkpoint前核对许可；dev保留必要可访问引用，不强塞原始数据到Git。

最终只返回真实版本、绑定任务及预算、资格/完成槽位、效果与不确定性、原始产物位置、解释重构、失败/偏差、同步状态和剩余投稿必需缺口。然后等待证据B，不开启下一研究轮次。
