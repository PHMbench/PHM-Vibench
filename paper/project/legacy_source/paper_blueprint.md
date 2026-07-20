# Paper 7 蓝图：Operator-Level Attention（顶刊口径 / 理论主导）

**最后更新**：2025-12-14  
**目标档位**：顶刊/顶会（理论/信号处理方法轨道）  
**数据口径**：合成信号验证为主证据链；PHM-Vibench（CWRU/XJTU）工业数据为可运行性补充  

---

## 1) 一句话定位

把注意力从“特征维度”提升到“信号处理算子维度”：给出严格的算子空间与物理约束注意力理论，并用可控合成信号实验验证“算子选择符合物理预期”的解释可靠性。

---

## 2) 顶刊证据链（必须交付）

### 2.1 合成信号验证（必须，主证据链）
- 至少 8 类信号：单频/双频/瞬态/噪声/调制/脉冲等
- 输出：权重热图 + 物理一致性评分报告（目标>0.9，阈值可解释）

### 2.2 数学证明（必须）
- 定理1（通用逼近/表示能力）
- 定理2（物理约束下的稳定性/一致性）

### 2.3 工业数据（可运行性补充）
- PHM-Vibench：CWRU/XJTU（可选）或 THU_018_basic（对齐统一基线）
- 强调“可解释性分析”，性能不是核心卖点（如需性能，必须给出口径与对照）

---

## 3) 复现入口（建议固定）

### 3.1 合成信号验证（主入口）
```bash
python Paper/TII_operator_attention/code/synthetic_verification.py --verbose
```

### 3.2 工业数据概念验证（可选）
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config_dir configs/unified_baseline/config_OperatorAttention_optimized.yaml
```

---

## 4) 与统一协议的绑定

- 可解释评估协议：`Paper/doc/12_14/codex/explainability_eval_protocol.md`
- 结果表模板：`Paper/doc/12_14/codex/results_tables_template.md`

---

## 5) TODO（按可验收拆解）

### P0（本周）
- [ ] 跑完8类合成信号验证并生成论文级图表（热图/一致性柱状图/雷达图）
  - **验收**：`results/` 下出现可引用的报告与图（含元数据与配置快照）
- [ ] 补齐定理证明的可投稿附录结构
  - **验收**：LaTeX或Markdown可直接并入论文

### P1（两周）
- [ ] 工业数据补充实验（L1=0 vs 1e-6 等配置矩阵）
  - **验收**：给出权重分布与解释稳定性对照表
- [ ] 与MoE/融合方法做机制对比讨论（算子级 vs 路径级/跨模态）
  - **验收**：形成可写入Discussion的对照结论

