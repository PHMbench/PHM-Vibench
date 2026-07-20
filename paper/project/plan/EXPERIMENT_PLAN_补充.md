# 1D-2D_fusion_explainable 实验补充计划

> 更新日期: 2026-05-12
> 作用范围: Paper02 的执行计划和证据验收边界。本文档不是已完成实验结果。

## 当前判定

- 现有 `submission_prep/baseline_ablation_matrix.yaml` 已声明 6 个 baseline 和 7 个 ablation。
- 当前仓库只具备 wiring/smoke 级证据；尚无可支撑 IEEE Transactions 投稿的 accepted CWRU/XJTU 多 seed 结果。
- 当前会话 GPU preflight 未通过：`nvidia-smi -L` 不可用，PyTorch CUDA device count 为 0。因此本计划只能进入准备和静态检查，不能生成 accepted GPU evidence。
- `best_model.pth`、历史图表、manuscript 草稿和脚本脏改不得自动当作 accepted evidence；必须先通过 artifact gate。

## 硬性前置条件

1. 从 parent repo 根目录执行命令，入口固定为 `python main.py --config ...`。
2. Q0 资源检查必须通过：

```bash
nvidia-smi -L
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

验收条件：本机 GPU `0,1` 均可见，且 PyTorch 输出 `True` 和 `2`。

3. 每个 accepted run 必须包含：
   - `run_meta.yaml`
   - `metrics.json` 或 `metrics.csv`
   - stdout/stderr 或 trainer log
   - config path、command、seed、dataset split、preprocessing signature
   - `CUDA_VISIBLE_DEVICES`、GPU model、GPU count、batch size、precision、runtime
   - submodule SHA 或 parent git SHA
   - OOM/dependency/data failure reason if any

## P0 实验矩阵

### 主方法

| ID | 方法 | 命令来源 | 状态 |
|---|---|---|---|
| P00 | 1D-2D fusion proposed method | `configs/vibench/min.yaml` | pending same-protocol GPU evidence |

### Baselines

| ID | Baseline | 目标 |
|---|---|---|
| B01 | 1D-only / no 2D signal path | 验证 2D 分支贡献 |
| B02 | ResNet | 强诊断模型 baseline |
| B03 | SincNet | 频域归纳偏置 baseline |
| B04 | TFN | TOP representative frequency proxy |
| B05 | WKN | TOP representative wavelet/frequency proxy |
| B06 | ConvTransformer | competitive architecture baseline |

### Ablations

| ID | Ablation | 目标 |
|---|---|---|
| A01 | disable 2D signal-processing path | 真正的 1D-only 消融 |
| A02 | smaller STFT window | time-frequency resolution sensitivity |
| A03 | smaller STFT hop length | alignment sensitivity |
| A04 | concat fusion switch | fusion operator contribution |
| A05 | paper-local demo class-count sanity | sanity only; not accepted evidence |
| A06 | FFT-only signal layer | frequency-only proxy |
| A07 | legacy 1D-only / 2D-only / no-statistical surface | rewrite into current-root real-data ablation before acceptance |

## Recent TOP Work Binding

Paper02 至少绑定以下 2024-2026 TOP-source 方法或本地 faithful representative：

| Work ID | Role | Acceptance boundary |
|---|---|---|
| RWTOP2024-TIMEMIXER | multiscale temporal baseline | representative run or exact integration |
| RWTOP2024-MOMENT | foundation representation baseline | local frozen/compact proxy until exact integration |
| RWTOP2025-CATCH | channel/frequency baseline | TFN/WKN/frequency-patching representative evidence |
| RWTOP2025-DADA | bottleneck/anomaly baseline | local bottleneck/anomaly representative evidence |
| RWTOP2026-PGRFNET | prototype/relational diagnostic comparator | local prototype/channel proxy evidence |
| RWTOP2026-GTM | frequency-domain attention comparator | TOP-Q2 binding; pending artifacts |
| RWTOP2026-CSLSTM | contextual/seasonal anomaly baseline | local LSTM/time-frequency proxy evidence |

No SOTA claim is allowed until P00 beats all declared baselines and runnable TOP representatives under the same split, seed, preprocessing, and metric protocol.

## Execution Order

1. Q0 GPU preflight. Stop if GPUs `0,1` are not visible.
2. One-epoch GPU smoke on the current maintained config:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

3. Same-protocol CWRU/XJTU runs for P00 and B01-B06 with at least three seeds.
4. Same-protocol A01-A07 ablations, with A05 marked sanity-only unless rewritten to emit full accepted metadata.
5. TOP-Q2 representative evidence for `RWTOP2026-GTM` through B04/B05/A06.
6. Artifact gate over `paper/UXFD_paper/results/accepted_runs`.
7. Manuscript table/figure update only after accepted artifacts exist.

## Completion Standard

- [ ] CWRU and XJTU each have at least three accepted seeds for P00 and B01-B06.
- [ ] A01-A07 have accepted same-protocol artifacts or explicit blocker records.
- [ ] At least one TOP representative is accepted, and all TOP bindings have exact/representative/blocker status.
- [ ] Result tables are generated only from accepted artifacts.
- [ ] GPU metadata proves the 2x4090 resource policy.
- [ ] SOTA wording is used only if accepted evidence beats every declared baseline and runnable TOP representative.
