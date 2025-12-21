# `model_collection` 对比模型整理进 PHM‑Vibench 的计划（v1）

来源目录：
- `/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/model_collection/`

目标：
- 把“对比基线模型”整理到主仓库 `src/model_factory/X_model/` 下，便于统一通过 vibench 配置跑表
- 只保留模型定义与 forward（训练/数据/脚本全部剥离）
- 避免与主仓库已有模型重复/冲突

---

## 1) 上游包含的模型（初步盘点）

根目录显式模型文件：
- `Resnet.py`（包含 ResNet + 一些可学习滤波层/变体）
- `Sincnet.py`（SincNet 变体）
- `WKN.py`（WKN / WKN_m 变体）
- `TFN.py`（也有子目录 `TFN/`：更完整的工程实现）
- `MCN.py`（也有子目录 `MCN/`：含多模型、训练器、数据）
- `MWA_CNN.py`（注意：主仓库已存在 `src/model_factory/X_model/MWA_CNN.py`，需要比对是否重复）
- `CI_GNN.py`, `Physics_informed_PDN.py`, `F_EQL.py`, `EELM.py`
- 解释相关：`base_explainable.py`, `GradCAM_XFD.py`

---

## 2) 主仓库落位建议（保持整洁）

### 2.1 Baseline 模型统一放到 `X_model/baselines/`

建议结构：
```
src/model_factory/X_model/baselines/
  __init__.py
  ResNet_MC.py              # 从 model_collection/Resnet.py 抽取/封装
  SincNet.py
  WKN.py
  TFN.py
  MCN.py
  EELM.py
  CI_GNN.py
  PhysicsInformedPDN.py
  FEQL.py
```

每个文件只做两件事：
1) 定义一个 vibench 可加载的 `Model` 类（保持 repo 现有 model_factory 约定）
2) forward 支持输入形状（尽量与 vibench 统一）：`(B, L, C)` 或由 adapter 转换

### 2.2 统一适配规则（必须写死）

- 统一输入：默认接受 `x: (B,L,C)`；若原模型用 `(B,C,L)`，在 wrapper 内部做一次 permute
- 统一输出：分类输出 logits `(B,num_classes)`；回归/多任务按 task 要求走统一 head（必要时写 adapter）
- 统一参数名：尽量采用 `in_channels/num_classes/seq_len` 等与 vibench 常用字段一致，避免 paper 风格乱入

---

## 3) 注册与配置

### 3.1 注册到 `src/model_factory/model_registry.csv`

新增模型时：
- `model.type` 推荐继续用 `X_model`
- `model.name` 用稳定字符串（避免与现有 `CNN/Transformer/...` 重名）
  - 例：`BASE_ResNet_MC`, `BASE_SincNet`, `BASE_WKN`, `BASE_TFN`, `BASE_MCN`

### 3.2 配置放在哪里（重要：paper configs 在 submodule）

- 基线模型的 vibench 配置文件也放在各自 paper submodule 内（用于跑对比表）：
  - `paper/UXFD_paper/<paper_id>/configs/vibench/baselines/<baseline>.yaml`
- 主仓库不维护“某篇 paper 的对比实验 configs”

---

## 4) 解释方法的继承（与 explain_factory 联动）

- `GradCAM_XFD.py` 建议移植到主仓库 `src/explain_factory/explainers/gradcam_xfd.py`
- 若某些 baseline 模型自带“可解释 hook”（例如中间 feature map），在 wrapper 里暴露统一 hook 名称，供
  explain_factory 使用（避免每个模型单独写解释脚本）

---

## 5) 最小验收（本科生可操作）

1) 对一个 baseline 写一个最小 config（放入某个 submodule）并跑 1 epoch：
```bash
python main.py --config paper/UXFD_paper/<paper_id>/configs/vibench/baselines/base_resnet.yaml --override trainer.num_epochs=1
```
2) 确认输出目录包含：
- `config_snapshot.yaml`
- `metrics.*`
- `artifacts/manifest.json`

