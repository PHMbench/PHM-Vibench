# VIBENCH 映射与一键复现（模板）

> 放置位置（在对应 paper submodule 内）：`paper/UXFD_paper/<paper_id>/VIBENCH.md`

## 1) 基本信息

- `paper_id`：`<paper_id>`
- 上游真相源：
  - README：`./README.md`
  - 关键脚本/配置：`./`（按本 submodule 结构填写）
- 主仓库版本（建议填写 commit）：`<phm-vibench-commit>`
- submodule 版本（建议填写 commit/tag）：`<submodule-commit-or-tag>`

## 2) 主仓库一键命令（唯一推荐入口）

配置文件（保存在本 paper submodule 内，避免污染主仓库 configs）：
- `paper/UXFD_paper/<paper_id>/configs/vibench/<config>.yaml`

最小可跑（建议先 1 epoch）：
```bash
python main.py --config paper/UXFD_paper/<paper_id>/configs/vibench/<config>.yaml --override trainer.num_epochs=1
```

## 3) operator_graph 摘要（稳定口径）

- stage 列表：`preprocess_1d / preprocess_2d / feature / fusion / router / reasoner / head`
- 本 paper 使用的关键 operators：
  - `<operator_id_1>@<version>`
  - `<operator_id_2>@<version>`

## 4) Explain 依赖的 data metadata（ExplainReady 门控）

最小需求（缺失时必须可审计降级）：
- `sampling_rate`：用于 Hz 轴/频带解释
- `sensor`：通道名/单位/安装位
- `window_length` / `stride`
- `operating_condition`（如转速/负载）
- `domain`（DG 时必需）
- `transform`（如 STFT/Envelope/Filter）

若 metadata 缺失，检查输出目录：
- `artifacts/data_metadata_snapshot.json`
- `artifacts/explain/eligibility.json`

## 5) 证据链产物（必须存在）

跑完后，在输出目录中应看到：
- `config_snapshot.yaml`
- `metrics.json` 或 `metrics.csv`
- `artifacts/manifest.json`
- `artifacts/data_metadata_snapshot.json`
- （若 explain 启用）`artifacts/explain/eligibility.json` 与 `artifacts/explain/summary.json`

## 6) 与 submodule 原始复现的差异说明（必填）

| 项目 | submodule 原始复现 | 主仓库 vibench 复现 |
|---|---|---|
| 入口命令 | `<cmd>` | `python main.py --config ...` |
| 配置体系 | `<yaml/args>` | 5‑block + `model.operator_graph` + `trainer.extensions` |
| 输出目录 | `<path>` | `save/...`（或 `environment.output_dir`） |
| 依赖 | `<pip/conda>` | 以主仓库 `requirements.txt` 为上限 |

## 7) 常见问题（FAQ）

- Q: submodule 未初始化会怎样？
  - A: 主仓库不应报错；只有进入 submodule 目录执行其脚本时才需要 init。
- Q: explain 报不可用？
  - A: 查看 `eligibility.json` 的 `reasons`，按 `suggestion` 补齐 metadata 或启用对应 hooks。
