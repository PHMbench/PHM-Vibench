# Paper 5 蓝图：Fuzzy-XFD（顶刊口径 / 可复现 / 可验收）

**最后更新**：2025-12-14  
**目标档位**：顶刊/顶会（可解释+安全关键方向）  
**数据口径**：PHM-Vibench 多数据集（至少 CWRU + XJTU），并保留 THU_018_basic 作为统一基线对齐  

---

## 1) 一句话定位

用“规则可审计”的模糊推理把故障诊断从黑盒回到工程师可读的规则空间：在多数据集场景下强调**可解释性可靠性（稳定/忠实/稀疏）+ 安全兜底（高风险失败案例）**，而不仅是准确率。

---

## 2) 顶刊证据链（必须交付）

### 2.1 性能与稳定性
- 至少 2 数据集（CWRU、XJTU）in-domain
- 至少 3-seed mean±std 或 95%CI

### 2.2 规则级解释评估
按 `Paper/doc/12_14/codex/explainability_eval_protocol.md`：
- Faithfulness（Del@k，对规则/特征遮挡）
- Stability（扰动下激活规则一致性）
- Sparsity（激活规则数/覆盖率）

### 2.3 安全关键失败案例（必做）
- 2–3 个“高风险误判”样本：解释输出必须可追溯（规则/隶属度/证据字段）

---

## 3) 复现入口（建议固定）

### 3.1 统一基线（对齐口径）
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config_dir configs/unified_baseline/config_FuzzyLogic_v2.yaml
```

### 3.2 PHM-Vibench 多数据集
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config_dir configs/PHM_Vibench/config_FuzzyLogic.yaml
```

---

## 4) 交付物清单（写作/图表）

- Figure：规则/隶属度可视化（不同故障/不同数据集对照）
- Figure：faithfulness/stability/sparsity 指标对比（含误差条）
- Table 2：主性能（CWRU/XJTU，3-seed）
- Table 4：解释评估
- Case study：安全关键失败案例（解释证据链）

---

## 5) TODO（按可验收拆解）

### P0（本周）
- [ ] 锁定当前最佳配置的可复现命令（含seed与输出目录）
  - **验收**：输出 `run_meta.yaml` + `metrics.json`
- [ ] 产出 2–3 个安全关键失败案例（含解释证据字段）
  - **验收**：每例可复现、可截图入论文

### P1（两周）
- [ ] CWRU/XJTU 完成 3-seed 统计（mean±std/95%CI）
- [ ] 完成规则级解释评估：faithfulness + stability + sparsity

### P2（一个月）
- [ ] 跨数据集泛化（LODO或transfer）+ 解释一致性分析

