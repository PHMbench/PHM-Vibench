# P01：可解释算子归纳偏置实现

## 已实现与当前边界

模型入口为 `src/model_factory/X_model/P01OperatorBias.py:Model`。它遵守 PHMFactory 的 `Model(args_model, metadata)` 接口，输入为 `[B,L,C]`，输出为 `[B,num_classes]`。这是基于 TSPN 算子思路的独立物理参数化变体，不覆盖 `TSPN.py` 或 `TSPN_UXFD.py`，也不是原 TSPN 权重可直接迁移的同构网络。

四条命名路径：原波形统计、阶次频带、固定 Hz 共振带→Hilbert 包络→阶次频带、STFT 频带统计。每条路径的线性读出按类别中心化；偏置与所有路径贡献在同次前向中相加得到 logits。贡献是固定模型的计算分解，不是唯一的物理因果解释。共享侧信息也会影响路径贡献。

已实现频带中心/带宽的阶次绑定、固定 Hz 载波、一致性损失、单边路径 margin 保持及必要对照。当前版本处理窗内近似恒定的测量转速；没有把变速轨迹替换成均值。尚未实现完整 CWT 路径、角域重采样、速度匹配的可变 STFT 窗或 LPR/SYNC。固定 STFT 已保留频带和帧统计，不声称严格尺度等变。

## 一键验证

在本分支根目录，使用已准备的 Python 环境：

```bash
python -m pip install -e .
bash scripts/p01/run.sh test
bash scripts/p01/run.sh demo results/p01_demo_01
```

`demo` 明确生成合成数据，并采用 `standalone_demo` 模式。失败不会切换数据或模型。合成结果只检查执行与机制构造，不是正式故障诊断成绩。

只拿到补丁包时，可直接在包根目录运行 demo；依赖为 PyTorch、NumPy、pandas、PyYAML、pytest。安装完整 PHMFactory 后再使用真实数据模式。不要在真实实验失败后切换到 demo。

## 真实数据：先改一个本地配置

复制 `configs/experiments/p01/local.example.yaml` 为自己的配置。填实际数据路径、metadata 列名、source domains 和新工况顺序。参数上下界必须由源数据和采集物理范围确定；示例中的 320 Hz 不是任何真实轴承的默认共振频率。

原始 NPZ 文件的信号数组名、shape 必须显式声明。也可以设置 `format: phmfactory_reader`，使用 PHMFactory 已有的实际 `Name` reader；不猜测 reader 名称。metadata 本身存在不等于原数据可用。

```bash
python scripts/p01/run_experiments.py --config configs/experiments/p01/local.yaml \
  --mode offline --output results/p01_plan --dry-run

CUDA_VISIBLE_DEVICES=0 bash scripts/p01/run.sh all \
  configs/experiments/p01/local.yaml results/p01_real_01 cuda:0
```

默认不自动选择 GPU。只暴露一张显式指定的非物理 GPU 2；CUDA 不可用时直接失败。每次使用新输出目录，不覆盖、不跳过已有实验。

可独立运行：

```bash
bash scripts/p01/run.sh offline CONFIG NEW_OUTPUT cpu
bash scripts/p01/run.sh continual CONFIG NEW_OUTPUT cpu
```

修改 `seeds`、`offline_arms`、`continual_arms` 和 `datasets` 控制有限矩阵。三个数据集可以有不同标签空间：分别用各自的配置、输出类别数和标签映射运行，不强行拼接不等价标签。三份配置依次调用同一个命令即可，不新增实验控制框架。

## 该脚本与 PHMFactory 的关系

真实模式通过已有 Model Factory 创建模型，通过已有 reader 读取原始记录。`run_experiments.py` 是论文级、独立实验单元采样和顺序更新驱动，不是通用 PHMFactory Pipeline/TrainerFactory 的替换，也不声称已通过整仓全部测试。

提供了原生 DG 接入文件 `src/task_factory/task/DG/p01_operator_bias.py` 与配套 dataset 文件。已有合法 PHMFactory DG 配置可以改为：

```yaml
model:
  type: X_model
  name: P01OperatorBias
  num_classes: 3
  in_channels: 1
  fs_field: sample_rate_hz
  rpm_field: rotation_speed_rpm
  # 其余频带/窗参数见 demo.yaml 的 model 块，需按真实物理范围修改。
task:
  type: DG
  name: p01_operator_bias
  model_task_id: classification
  loss: CE
  p01_paired_supervision: true
  p01_cov_weight: 0.1
  p01_shift_samples: 32
data:
  normalization: none
  p01_unit_field: 实际试件或运行列
  p01_fs_field: 实际采样率列
  p01_rpm_field: 实际转速RPM列
  split:
    strategy: grouped_metadata
    # 沿用该数据集已核验的完整 group/split 配置，不从本文猜测。
```

这不是可直接替代完整原生配置的文件。当前新增原生 Task 仅覆盖离线成对监督；持续回放由论文驱动实现。原生 Task 的 unit-level reduction 要求已有采样器对实验单元公平采样，不会修复任意不平衡窗口采样。原生全 Pipeline 接入尚待本地真实配置核验。

## 数据协议

每行表示一个物理单元在一个工况下的一次记录。必需字段由 `columns` 显式映射：`file, unit_id, label, domain, split, sample_rate_hz, rotation_speed_rpm`。`split` 只能为 `update/validation/test`，且同一物理单元在全部工况中保持相同 split。

同一记录先确定 split 再切窗；多个窗口不构成独立重复。若一个物理单元有多个文件，先按明确科学规则形成记录，当前驱动不自行猜测合并。同一物理单元跨工况重复测量时，跨域统计仍须按物理单元配对，不能把它们当独立试件。

初始只训练 source domains。对每个新工况：先计算永久 test 单元的 pre-update 预测，然后读取 update 和 validation 进行更新。test 从不进入回放或模型选择。所有模型按当前阶段 validation CE 选 checkpoint，阶段间重置 Adam；这是共同的预声明训练策略。

当前增强只做圆周时间起点移动，适用于标签保持的周期窗口；它不是实际加速、传递路径变化或完整速度协变增强。不能用它单独证明未见速度泛化。

## 输出与解释

每个条件保存训练曲线、阶段 checkpoint、逐域指标、逐单元预测、实际配置、运行时间与内存开销。Brier/CE 先按窗口算损失、再按物理单元平均；macro-F1 基于每个单元的平均预测概率。另报类平衡 Brier 和实际覆盖类别数。

`tables/domain_stage_metrics.csv` 保留各工况结果；`seed_summary.csv` 的标准差是训练重复的离散程度，不是物理总体置信区间；单 seed 的 std 留空，不补零。pre-update 性能不冒称 FWT。

冻结可加模型后运行：

```bash
python scripts/p01/explain.py --config ACTUAL_CONFIG \
  --checkpoint SELECTED_MODEL --dataset DATASET_NAME --output NEW_EXPLAIN_DIR
```

删除每条路径并进行同类别/不同类别、相同 donor-domain 的贡献替换。共同 anchor 和未配对数量被记录。这是贡献级预测依赖测试，包含路径的侧信息影响，不是机械系统物理反事实。

## 分支建议

当前账号可写上游，不必先 fork。使用 `research/p01-operator-bias-20260907`，只在该分支开发，不改 main/dev；通过针对性检查后再提小 PR。若要长期隔离，fork 一次并用上游 remote 同步；不要每篇论文复制一次完整平台。已有 `liq22/PHM-Vibench` 是独立旧副本，不要覆盖。

本包不修改 P01 论文仓库的子模块指针，不创建额外 registry/manager/adapter，不增加结果完整性校验体系。
