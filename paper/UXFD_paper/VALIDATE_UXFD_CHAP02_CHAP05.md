# Validate UXFD → Thesis Support (Chap02–Chap05)

目标：用**可复制命令 + 可检查产物**证明本仓库已具备支撑论文第 2–5 章（理论/算子网络/专家网络/信息融合）的工程闭环。

---

## 0) 环境准备（一次性）

建议在仓库根目录创建并使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

备注：
- `wandb` / `swanlab` 不是硬依赖；未安装时会跳过对应 logger（不应阻塞训练/测试）。

---

## 1) 配置/文档/单测门禁（必须通过）

```bash
python -m scripts.validate_configs
python -m scripts.validate_docs
python -m pytest test/ -q
```

通过标准：
- 上述命令均退出码为 0。

---

## 2) Chap02/Chap03（算子库 + 透明算子网络主干）最小可跑证据

跑 UXFD 最小 demo（仅 1D operator graph + features；不强依赖 UXFD 扩展模块）：

```bash
python main.py --config configs/demo/uxfd/00_smoke_tspn_uxfd.yaml --override trainer.num_epochs=1
```

检查产物（任意最新 run_dir 即可）：
- `<run_dir>/config_snapshot.yaml`
- `<run_dir>/artifacts/manifest.json`
- `<run_dir>/test_result_0.csv`

算子/特征 SSOT（人工检查）：
- `src/model_factory/X_model/UXFD/OPERATOR_CATALOG.md`
- `src/model_factory/X_model/UXFD/FACT_TABLE.md`

---

## 3) Chap05（TIFN：SP2D + fusion）最小可跑证据

跑 SP2D demo（STFT 分支 + fusion；并产出 predictions）：

```bash
python main.py --config configs/demo/uxfd/10_smoke_tspn_uxfd_sp2d.yaml --override trainer.num_epochs=1
```

检查产物：
- `<run_dir>/artifacts/predictions.npz`（应存在）
- `<run_dir>/artifacts/manifest.json`（其中 `predictions_path` 非空）

---

## 4) Chap04（DEN：fuzzy/logic 等 best-effort 扩展）最小可跑证据

跑 full demo（SP2D + fusion + fuzzy + operator-attention + logic；best-effort）：

```bash
python main.py --config configs/demo/uxfd/20_smoke_tspn_uxfd_full.yaml --override trainer.num_epochs=1
```

检查产物：
- `<run_dir>/artifacts/manifest.json`
- （若启用 predictions）`<run_dir>/artifacts/predictions.npz`

工程口径核对（人工检查）：
- fuzzy/logic 装配入口：`src/model_factory/X_model/TSPN_UXFD.py`
- fuzzy/logic 模块：`src/model_factory/X_model/UXFD/fuzzy/`、`src/model_factory/X_model/UXFD/neurosymbolic/`

---

## 5) 可选：汇总 runs → CSV（便于审计）

```bash
python -m scripts.collect_uxfd_runs --input results --out_dir reports
ls -la reports/uxfd_runs.csv
```

---

## 6) 合并到 main 前的人类检查清单（PR Reviewer Checklist）

- 代码/配置口径：
  - UXFD 核心模型入口稳定：`model.type: X_model` + `model.name: TSPN_UXFD`
  - 算子/特征 key 不“发明”：以 `src/model_factory/X_model/UXFD/OPERATOR_CATALOG.md` 为准
- 产物契约：
  - 每次 run 都能写出 `config_snapshot.yaml` 与 `artifacts/manifest.json`（best-effort 不 crash）
- 复现门禁：
  - 本文档第 1–4 节的命令均可跑通（CPU 环境即可）
- Submodule 状态（如你计划一起合并 submodule gitlink）：
  - 先在各 submodule 内提交必要改动，再更新父仓库 gitlink；否则父仓库只会显示 “dirty submodule” 而无法 review 文件级 diff。
