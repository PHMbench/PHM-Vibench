# Test Plan: Merge `lq_merge_UXFD` → `main`

范围：本计划用于评估“当前工作分支（基于 `lq_merge_UXFD`）”是否已具备合并到 `main` 的条件，并给出逐步验证命令。

约束：
- 本轮 **不考虑 submodule 指针更新**（只合并主仓库文件的改动）。
  - 注意：当前 `lq_merge_UXFD` 分支历史中如果已经包含 submodule gitlink 的提交（例如 `paper/LQ_vibench_fix`），
    建议在提 PR 前用 **cherry-pick 到新分支** 或 **interactive rebase** 将这些提交剔除，否则 merge 时仍会带入指针变更。

---

## A. 分支差异（必须先看清楚）

1) 获取远端并确认分叉点：

```bash
git fetch --all --prune
git log --oneline origin/main..origin/lq_merge_UXFD
git log --oneline origin/lq_merge_UXFD..origin/main
git diff --name-only origin/main..origin/lq_merge_UXFD | head -n 200
```

预期（当前仓库状态示例）：
- `origin/main` 相对 `origin/lq_merge_UXFD` 的额外提交很少（例如只改了 `data/metadata.xlsx`）。
- `origin/lq_merge_UXFD` 相对 `origin/main` 的提交包含 UXFD/NSN demos、SSOT、tests、tooling 等。

2) 如果你希望在 PR 前先把 `main` 的更新引入当前分支（推荐，减少 CI 偏差）：

```bash
git checkout lq_merge_UXFD
git merge origin/main
```

---

## B. 合并范围确认（不包含 submodule 指针）

1) 确认本次要合并的主仓库文件（示例，按实际 `git diff` 为准）：
- `src/trainer_factory/Default_trainer.py`（可选依赖 logger 的健壮性修复）
- `.gitignore`（忽略本地 thesis workspace）
- `paper/UXFD_paper/VALIDATE_UXFD_CHAP02_CHAP05.md`（Chap02–05 人类验证指令）
- `src/changelog/CHANGELOG.md`（合并说明与验证命令）

2) 确认 staged changes 不包含 submodule 变更：

```bash
git diff --cached --submodule=diff
```

通过标准：
- 输出中不应出现类似 `160000 <sha> paper/UXFD_paper/<submodule>` 的变更。

---

## C. 环境准备（一次性）

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

备注：
- `wandb` / `swanlab` 不是硬依赖；未安装时应跳过对应 logger，不阻塞训练/测试。

---

## D. 逐步门禁（必须全部通过）

### D1) 配置/文档/单测

```bash
python -m scripts.validate_configs
python -m scripts.validate_docs
python -m pytest test/ -q
```

通过标准：全部退出码为 0。

### D2) UXFD demos（支撑 Chap02–05）

> 统一建议：CPU + 1 epoch + 固定 seed，确保可复现与稳定产物。

Chap02/Chap03：最小 UXFD 主干（算子层 + 特征层 + 分类器）：
```bash
python main.py --config configs/demo/uxfd/00_smoke_tspn_uxfd.yaml --override trainer.num_epochs=1 --override trainer.device=cpu --override trainer.gpus=1 --override environment.seed=0
```

Chap05：SP2D（STFT 分支 + fusion + predictions）：
```bash
python main.py --config configs/demo/uxfd/10_smoke_tspn_uxfd_sp2d.yaml --override trainer.num_epochs=1 --override trainer.device=cpu --override trainer.gpus=1 --override environment.seed=0
```

Chap04：full（SP2D + fuzzy + operator-attention + logic；best-effort）：
```bash
python main.py --config configs/demo/uxfd/20_smoke_tspn_uxfd_full.yaml --override trainer.num_epochs=1 --override trainer.device=cpu --override trainer.gpus=1 --override environment.seed=0
```

### D3) 产物检查（必须存在）

对每个 run_dir，检查：
- `<run_dir>/config_snapshot.yaml`
- `<run_dir>/artifacts/manifest.json`

额外（D2 中 sp2d/full demo）：
- `<run_dir>/artifacts/predictions.npz`

快速定位（示例）：
```bash
find results/demo/uxfd -name manifest.json | tail -n 5
find results/demo/uxfd -name predictions.npz | tail -n 5
```

---

## E. 合并前最终检查清单（Reviewer Checklist）

```bash
git status -sb
git diff origin/main..HEAD --stat
```

确认点：
- `src/changelog/CHANGELOG.md` 已更新，且 `Validation` 命令齐全。
- submodule 仅处于工作区 dirty（如有），但没有被 stage/commit 到主仓库变更中。
