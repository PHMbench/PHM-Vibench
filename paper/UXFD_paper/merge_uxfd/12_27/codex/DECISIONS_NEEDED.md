# 需要确认的决策点（12_27）

这些点如果不先确认，后续容易返工；建议尽早拍板。

---

## 1) UXFD 组件目录命名（是否引入 `UXFD_component/`）

现状：
- 代码当前已有：`src/model_factory/X_model/UXFD/`
- 12_23 README 提到未来“组件区统一命名为 `src/model_factory/X_model/UXFD_component/` 并用 shim 逐步淘汰 UXFD/”
  - 见：@`paper/UXFD_paper/merge_uxfd/12_23/README.md`

需确认：
- 是否真的要做这个 rename（会波及 import/registry/文档），还是保持 `UXFD/` 不动（更 KISS）。

---

## 2) pilot paper 选择（WP0）

建议默认选：
- `paper/UXFD_paper/1D-2D_fusion_explainable`

需确认：
- pilot 是否需要 2D/fusion/op-att/fuzzy 哪些组件（决定 WP1 的最小移植范围）。

---

## 3) post-run / 绘图产物落位

当前脚本默认：
- 产物检查 + 绘图：`scripts/uxfd_postrun.py`
- 绘图输出：`<run_dir>/figures/`
- 审计输出：`<run_dir>/artifacts/plots/plot_eligibility.json`

需确认：
- `figures/` 是否作为统一绘图输出目录（建议保持）。

参考：@`paper/UXFD_paper/merge_uxfd/12_23/plot_factory_migration_plan.md`

---

## 4) 是否需要自动落盘 `predictions.npz`（用于混淆矩阵等）

当前策略（KISS）：
- 没有 predictions 就跳过混淆矩阵绘图，不强行改训练/任务代码。

需确认：
- 是否必须让主流程自动写 `artifacts/predictions.npz`（会更侵入）。

---

## 5) `.gitignore` 与证据样例目录

现状：
- `.gitignore` 的 `results/` 规则会忽略 `paper/UXFD_paper/merge_uxfd/12_22/results/**`（样例证据链文件可能无法被跟踪）

需确认：
- 样例证据链是否要纳入版本控制；如果要，需要改目录名或增加反忽略规则。

