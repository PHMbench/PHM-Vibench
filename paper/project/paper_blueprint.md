# Paper 2 蓝图：Explainable FD Toolkit（顶刊口径 / 可复现 / 可验收）

**最后更新**：2025-12-14  
**目标档位**：顶刊/顶会（系统/基准/工具链方向）  
**数据口径**：PHM-Vibench 多数据集（至少 CWRU + XJTU）  

**创新契约真源**：`innovation_contract.md`

---

## 1) 一句话定位

把故障诊断的可解释性从“单篇论文里的零散可视化”升级为**统一API + 统一评估协议 + 一键benchmark复现**的基础设施，使不同模型/解释方法在PHM-Vibench多数据集上可公平对比、可复现、可发布。

---

## 2) 顶刊证据链（必须交付）

### 2.1 统一接口与协议（可引用）
- `SignalData / ModelPlugin / ExplainabilityMethod` 接口稳定
- 输出格式统一：`metrics.json`、表格、图表、`run_meta.yaml`
- 协议绑定：`Paper/doc/12_14/codex/explainability_eval_protocol.md`

### 2.2 Benchmark与对比（可复现）
- 至少 5 个模型 × ≥2 类解释方法（intrinsic + post-hoc）
- 至少 2 个数据集（CWRU、XJTU）
- 至少 3-seed 或等价统计口径

### 2.3 工业demo（可复现）
- 至少 2 个端到端 demo（脚本 + 英文图表 + 报告）

### 2.4 竞争对比（必须）
- Captum/SHAP/LIME：速度、稳定性、忠实度、工程友好度对比表

---

## 3) 复现入口（建议固定）

```bash
# 工具包独立benchmark
python paper/UXFD_paper/Explainable_FD_Toolkit/scripts/run_benchmark_standalone.py

# 统一基线解释评估（对齐主仓库模型）
python paper/UXFD_paper/Explainable_FD_Toolkit/scripts/run_unified_explain_eval.py
```

输出建议收口到：
- `paper/UXFD_paper/Explainable_FD_Toolkit/benchmark_results/`
- `paper/UXFD_paper/Explainable_FD_Toolkit/results/`

---

## 4) 交付物清单（写作/图表）

- Figure 1：Toolkit总体架构（接口与流水线）
- Figure 2：评估协议与指标定义（faithfulness/stability/…）
- Figure 3：多模型×多方法对比（性能+解释性+效率）
- Table 2：主结果（性能）
- Table 4：解释评估（协议指标）
- Table X：与 Captum/SHAP/LIME 的对比表

---

## 5) TODO（按可验收拆解）

### P0（本周）
- [ ] 固定“一键复现入口”与输出目录结构（写入README）
  - **验收**：单命令生成 JSON/CSV/Markdown + 关键图表
- [ ] 把当前README中“2024路线图”标记为历史，新增“2025路线图”
  - **验收**：路线图条目绑定交付物与验收标准

### P1（两周）
- [ ] 补齐剩余模型适配：FuzzyLogic + OperatorAttention（对齐主仓库与7篇Paper）
  - **验收**：最小集成测试脚本通过 + 生成统一报告
- [ ] 完成 Captum/SHAP/LIME 对比实验与报告
  - **验收**：对比表可直接入论文 + 可复现脚本
- [ ] 完成 2 个工业demo（含英文图表）
  - **验收**：脚本可跑通 + 输出报告

### P2（一个月）
- [ ] 发布候选 v1.0：README/安装/示例/License/贡献指南
  - **验收**：新环境可安装并跑通最小demo
