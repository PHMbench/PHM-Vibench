那我就直接按你现在的目标来设计一版「dev_lq 分支里的 NNSPN：**路由用神经网络 + 自带可解释分析套件**」的优化方案，重点放在**可解释性**，顺便参考了一圈最新的 MoE / routing / 可解释 TSC / 可解释故障诊断的思路。

---

## 0. 核心目标先捋一下

你现在这个需求可以拆成两个子问题：

1. **路由用神经网络来学：**
   NNSPN 里每个样本（或通道 / 时间片）要通过一个 *Router 网络* 来选择/加权一组「专家算子」（频带滤波、包络、STFT patch、HSE 模块之类）。

2. **路由过程是可解释的：**

   * 能说清楚「某个样本是因为 *哪些物理特征*，被路由到了 *哪些专家*」；
   * 能给出**类间的路径签名**（比如：外圈故障主要走高频冲击 + 解调通路，内圈故障走中高频调制通路等）；
   * 最好还能对单样本给到局部解释（类似 SHAP/Grad-CAM 那种）。

下面所有设计你都可以映射到 `Unified_X_fault_diagnosis` 里 NNSPN 的 operator graph 上。

---

## 1. 文献里「可解释路由 / MoE」的几个模式

### 1.1 Routing Networks & MoE：把「路由器 + 函数块」分开

* **Routing Networks**：显式把网络拆成 *router + function blocks*，router 根据输入递归选择下一个 block，典型是 Rosenbaum 等的 Routing Networks，用强化学习训练路由器。([IBM Research][1])
* **Mixture of Experts (MoE)** / sparsely-gated MoE：输入只激活少量 experts，通过 softmax / top-k gating 做稀疏路由，代表性工作是 Shazeer 的 sparsely-gated MoE。([维基百科][2])

这类结构天然适合你 NNSPN：**专家就是各种可解释算子（物理滤波 / 结构化 block），路由器就是 NNSPN 的神经网络路由**。

### 1.2 「本身就可解释」的 MoE 变种

近两年 MoE + 可解释性的工作很多，几个可以直接借鉴思路：

* **IME（Interpretable Mixture of Experts）**：gating 把样本分配给 **线性 / Logistic 回归这种可解释 expert**，每个样本只用一个 expert，整个模型等价于「先分簇，再用简单模型」。([DeepAI][3])
* **MoE-X**：提出「本身可解释的 MoE」，通过 **稀疏激活 + 稀疏路由**，使每个 expert 只捕获少量语义因素，路由器还会偏向选择激活更稀疏的 expert，以提升可解释性。([Proceedings of Machine Learning Research][4])
* **FairMOE / I2MoE**：分成「可解释 expert + 黑盒 expert」，用 assignment module 决定什么时候走可解释支路、什么时候走强表达支路，并显式建模公平性、模态交互等。([SpringerLink][5])

抽象出来就是两点：

1. **专家本身要可解释**（线性 / KAN / 物理算子 / shapelet 等）；
2. **路由器有约束**（稀疏、单 expert、或「优先选择可解释 expert」）。

### 1.3 时间序列方向：InterpGN & shapelet / event-based 可解释 TSC

时间序列里也有一条明确路线：

* **InterpGN（Interpretability Gated Networks）**：用 gating 选择「解释型 expert（shapelet 规则）」或「深度 模型 expert」，路由器根据**解释型模型的置信度**决定是否需要更复杂模型。([ICLR 会议记录][6])
* **MAPIC、Z-Time 等可解释 TSC**：

  * MAPIC 用 Matrix Profile 找 motifs/discords 作为 shapelet，再用浅层决策树做分类，路径就是「出现/不出现某一段波形」。([PMC][7])
  * Z-Time 把多维时间序列抽象成**事件区间**，特征是事件的时序关系，本身就很容易解释。([SpringerLink][8])

这给 NNSPN 一个很自然的方向：**路由器只看「可解释的时序特征」来做路径选择**，比如 shapelet 匹配度、各频带能量、冲击指标等。

### 1.4 故障诊断领域的 XAI 实践

在机械故障诊断上，已经有不少「XAI + 故障诊断」工作：

* 用 SHAP 分析振动特征对故障分类贡献度，找出如偏度、波形因子等关键统计量。([MDPI][9])
* 「Explainable fault diagnosis」框架，用 CNN + Grad-CAM 看时频图上哪些区域支撑决策。([pure.hud.ac.uk][10])
* KAN-based 可解释故障诊断：浅层 KAN + 符号化激活，直接给出可读公式，同时做 feature attribution。([PubMed][11])
* 更多综述（XAI + rotating machinery）也强调：**特征重要性 + 通道/频带/时间位置的可视化**是主线。([Emergent Mind][12])

这些基本就是你想要的 NNSPN 可解释分析模块要做的事。

---

## 2. 把 NNSPN 的路由改成「神经网络 + 物理专家」的结构

假设你在 `dev_lq` 里，NNSPN 已经有一堆 operator/expert（比如不同频带滤波、不同尺度 HSE、STFT-patch 提取等），我们把路由显式写成：

### 2.1 Router 网络形式

1. 先对输入 (x) 做一个**可解释的特征提取** (z(x))：

   * 多通道振动/声音的统计量：RMS、峰值因子、偏度、峭度、形状因子等；
   * 物理频带能量：围绕 BPFI/BPFO/BSF、1×/2× 转频、网频旁带等的带通能量；
   * 若已有 HSE/TON 里的嵌入向量，可以再叠一层线性变换压缩。

2. Router 用一个小型 MLP / NAM：

   $$
   \pi(x) = \operatorname{softmax}\left(\frac{g_\theta(z(x))}{\tau}\right), \quad
   \pi(x) \in \mathbb{R}^M
   $$

   其中 (M) 是专家数，(\tau) 是温度。

3. 对于需要「接近离散路由」的，可以用 **Gumbel-Softmax** 或仅取 top-k：

   * 训练期：Gumbel-Softmax + straight-through；
   * 推理期：直接取 (\arg\max_k \pi_k(x)) 或 top-2 experts。

### 2.2 专家层的组合方式

假设第 (k) 个专家是 (f_k(\cdot))，NNSPN 的输出写成：

$$
h_k(x) = f_k(x), \quad
f(x) = \sum_{k=1}^M \pi_k(x), h_k(x)
$$

这样一来，**每个 (\pi_k(x), h_k(x)) 自然可以视为该 expert 对最终 logit 的贡献**（非常利于可解释性分析）。

如果 NNSPN 是多层多跳路由（类似 Routing Networks 的递归选择），可以在每一层都加一个 router：

$$
h^{(l+1)}(x) = \sum_{k=1}^{M_l} \pi^{(l)}_k(x), f^{(l)}_k\left(h^{(l)}(x)\right)
$$

同时记录每层的 (\pi^{(l)}(x))。

### 2.3 训练时加上「稀疏 + 可解释」正则

总 loss 可以设计成：

$$
\mathcal{L} =
\mathcal{L}_{\text{cls}}

* \lambda_{\text{sparse}} |\pi(x)|_1
* \lambda_{\text{ent}} H(\bar{\pi})
* \lambda_{\text{distill}} \mathcal{L}_{\text{KD}}
  $$

- (\mathcal{L}_{\text{cls}})：故障分类交叉熵；
- (|\pi(x)|_1)：鼓励**单样本路由稀疏**（每个样本尽量只走极少专家），呼应 MoE-X 的「稀疏激活更易解释」观点。([Proceedings of Machine Learning Research][4])
- (H(\bar{\pi}))：(\bar{\pi} = \mathbb{E}_x[\pi(x)])，加一个负熵或 load-balancing 约束，避免所有样本挤到同一个 expert（参考 MoE 里的 capacity / entropy 正则）。([维基百科][2])
- (\mathcal{L}_{\text{KD}})：和**一个可解释 teacher** 的知识蒸馏损失（下面说）。

---

## 3. 提升「路由本身」可解释性的 3 层设计

### 3.1 Router 自身是「可解释模型」

这部分可以走两条线：

1. **基于可解释特征 + 简单模型：**

   * Router 不用深 MLP，而是：

     * 线性 / logistic 模型（每个路由输出对应一组权重）；
     * **Neural Additive Model (NAM)**：每个特征一个子网络 (f_j(z_j))，总路由 logit 是 (\sum_j f_j(z_j))，可以画曲线看「某个物理特征改变对路由概率的影响」。([MDPI][9])
   * 好处：对每个路由 logit，你可以直接画「特征贡献条形图」，文本解释会特别清晰。

2. **Distill 到决策树 / 神经决策森林：**

   * 用 Router 的输入 (z(x)) 和输出的 top-1 路由标签训练一个浅层决策树 / TreeEnsemble，当做路由的 surrogate；
   * 参考 Neural Oblivious Decision Ensembles 等 work，把树结构和 NN 结合，也可以保留一定连续性。([DeepAI][3])
   * 最终可以给出类似：

     > 若「2×转频能量 > 阈值」且「峭度 > 阈值」则优先走高频冲击 expert。

### 3.2 专家定义要物理可解释

NNSPN 的 experts 尽量做成**有明确物理语义的 operator**，比如：

* Band-pass / Wavelet branch：对应某个典型故障频带；
* Envelope Demodulation branch：冲击性故障；
* 时频 patch expert：特定频带 + 时间窗；
* HSE 型 expert：固定的「频带 + 统计特征」组合。

这样路由解释可以写成非常「论文友好」的句子：

> 该样本主要通过了「低频不平衡 + 高频冲击解调」两条路径，对应转子不平衡叠加轴承外圈缺陷的物理机理。

这也呼应了可解释故障诊断里「让模型结构对齐物理机理」的建议。([pure.hud.ac.uk][10])

### 3.3 「优先选择可解释 expert」的路由策略

可以直接借鉴 InterpGN 和 IME 的思路：([ICLR 会议记录][6])

* 把 NNSPN 的 experts 分成两类：

  * **E_int**：完全可解释专家（物理算子 + 浅线性头 / KAN）；
  * **E_deep**：更强但较黑盒的深度 expert。

* Router 目标：

  * 大部分样本走 E_int；
  * 只有当「可解释 expert 置信度低」时才放到 E_deep。

可以实现成一个两步 gating：

1. 先用可解释 expert 预测并给出置信度 (p_{\text{int}}(y|x))；
2. Router 根据 (p_{\text{int}}) 决定「留在可解释支路」还是「交给深度支路」，类似：

   $$
   s(x) = \sigma(w^\top \phi(p_{\text{int}}(x), z(x))) \in [0,1]
   $$

   * (s(x) \approx 1)：使用可解释 expert；
   * (s(x) \approx 0)：使用深度 expert；
   * 在 dev_lq 里可以简单些：用一个阈值规则「最大类概率 > 0.9 则不用深度 expert」。

---

## 4. 一套「NNSPN 可解释分析方法」的完整 pipeline

你后面写子刊的时候，完全可以单独立一个节叫：

> XXX: **Routing-aware Explainability for NNSPN**

我给你拆成四层分析，从「全局」到「单样本」。

### 4.1 全局：类–路径签名矩阵

1. 对训练集/测试集每个样本，保存 router 输出 (\pi(x))；

2. 对每个类别 (c)，计算平均路由向量：

   $$
   \mu_c = \mathbb{E}_{x|y=c}[\pi(x)] \in \mathbb{R}^M
   $$

3. 把 ({\mu_c}_{c=1}^C) 组成一个 (C \times M) 的矩阵，画热力图：

   * 行：故障类型（正常、内圈、外圈、滚动体、不对中、齿轮断齿…）；
   * 列：专家（低频不平衡、倍频共振、高频冲击、解调、啸叫频段……）；
   * 颜色：该类样本在该 expert 上的平均路由概率。

这和 MoE-KAN/IME 里分析 expert 对不同模式分工的做法类似。([MDPI][13])

**用途：**

* 可以直接看「不同故障类型在物理算子上的典型路径」；
* 可以计算类间路径 KL 距离 / mutual information，当做一个**解释性 metric**。

### 4.2 局部：单样本的路径贡献分解

对单个样本 (x)，你已经有：

$$
f(x) = \sum_{k=1}^M \pi_k(x), h_k(x)
$$

可以定义每个 expert 对某个类 logit 的贡献：

$$
\Delta_k(x) = \pi_k(x), h_k(x)
$$

然后：

1. 做一个「路径贡献条形图」：Top-K experts + 对应的贡献大小；
2. 对每个 expert (f_k)，你又可以进一步可视化它基于的时频区域（比如 STFT patch / 频带滤波后的包络）；
3. 文本解释模板可以写成：

> 该样本被判为「Outer race fault」主要由以下路径贡献：
> 1）高频冲击 + 包络解调 expert（贡献 0.47），突出在 4×BPF_O 附近平带；
> 2）中频调制 expert（贡献 0.31），对应齿轮啸叫频段调制。

这本质上就是把「MoE 的加权和」显式拆开给人看，和 IME / MoE-X 中逐 expert 分解输出类似。([Proceedings of Machine Learning Research][4])

### 4.3 特征层：Router 的 SHAP / IG 分析

借鉴现有故障诊断里用 SHAP 分析特征贡献的做法：([MDPI][9])

1. 对 Router 的输入特征 (z(x))（物理统计量 + 频带能量等）做 SHAP / Integrated Gradients：

   * 目标输出可以是：

     * 某个 expert 的路由概率 (\pi_k(x))，或
     * 「是否走可解释 expert」的门控 (s(x))。

2. 得到 feature importance 后，可以讲：

   * 「高频能量 + 峭度」对冲击 expert 的路由影响最大；
   * 「1×/2× 转频能量差异」对不平衡 expert 选择影响最大。

3. 可以再把 SHAP 排名前几的特征映射回具体传感器/频段/时间窗，形成**物理解释闭环**。

### 4.4 Surrogate：用浅层 KAN/树作为「路由解释器」

结合 KAN & 可解释 MoE 的思路：([MDPI][14])

* 用 Router 的输入 (z(x)) 和 top-1 expert 标签训练一个浅层 KAN 或决策树；

* KAN 的优势是可以给出**符号表达式**，例如：

  $$
  \pi_{\text{shock}}(x) \approx \sigma(1.2,\log(\text{kurtosis}) + 0.8,E_{\text{HF}} - 3.1)
  $$

* 你可以把这个当做「简化版路由规则」，写进论文的可解释性分析章节。

---

## 5. dev_lq 分支里可以直接按这个来改的 TODO 清单

最后给你落到 repo 级别的 checklist（假设当前 NNSPN 还有个旧版 rule-based / heuristic 路由）：

### 5.1 模型结构改动

1. **新增 Router 模块**（比如 `models/nnspn/router.py`）：

   * 输入：`z(x)`（在 `forward` 里从原始时序 / HSE embedding 提取）；
   * 结构：

     * v1：两层 MLP + softmax 输出 (\pi(x))；
     * v2：NAM / 线性模型 + 可视化 hooks（以后再升级）。

2. **NNSPN forward 改造**：

   * 在 `forward` 中：

     * 先算 `z = feature_extractor(x)`；
     * `pi = router(z)`；
     * 对 experts 的输出做 `sum_k pi[:,k] * h_k(x)`；
   * 把 `pi` 存到 `self.last_routing` 里，方便外面调用 `explain()`。

3. **配置文件支持**：

   * 在 `configs/...yaml` 增加：

     * `router_type: {mlp, nam, tree-distill}`
     * `num_experts: M`
     * `route_mode: {soft, gumbel, topk}`
     * `interpret_features: [rms, kurtosis, bpfi_band_energy, ...]`

### 5.2 训练逻辑 & 正则

1. 在 loss 里加上前面那几项正则：

   * `lambda_sparse`、`lambda_ent`、`lambda_distill` 可在 config 里调。
2. 如果你做「InterpGN 风格」：

   * 先训练一个 shapelet/物理特征决策树 model，当做 teacher；
   * 训练 Router 时加一个 KD loss：

     * `KD(router_logits, teacher_logits)`。([ICLR 会议记录][6])

### 5.3 可解释性接口

写一个类似：

```python
class NNSPN(nn.Module):
    ...
    def explain(self, x, y_true=None, topk=3):
        """
        返回:
        - routing_probs: [B, M]
        - expert_contrib: [B, M]  (对目标 logit 的贡献)
        - class_path_signature: 可选，按 batch 统计
        """
```

并提供：

* 对单样本：返回 Top-K expert 名字 + 贡献；
* 对一个 batch：返回平均 (\mu_c)（按 label 聚类）；
* 可选：调用外部的 SHAP/IG 函数，对 Router 输入特征出一张「特征贡献表」。

### 5.4 分析脚本 / Notebook

在 `analysis/` 里搞几个脚本：

1. `plot_class_path_signature.py`：

   * 载入训练好的 NNSPN；
   * 对每个类计算 (\mu_c)；
   * 绘制热力图 + 保存为 pdf。

2. `sample_explain_demo.py`：

   * 随机抽几条样本，打印：

     * 预测类别；
     * Top-K experts & 贡献；
     * 若有时频图，叠加可视化。

3. `router_shap_analysis.py`：

   * 对 Router 做 SHAP/IG；
   * 输出一个「物理特征重要性表」，后面可以直接塞到子刊里。

---

如果你愿意，下一步我可以直接帮你**按这个设计写一份「NNSPN 可解释路由模块」的伪代码 / PyTorch 草稿**，你丢到 `dev_lq` 分支里改就行。

[1]: https://research.ibm.com/publications/routing-networks-adaptive-selection-of-non-linear-functions-for-multi-task-learning?utm_source=chatgpt.com "Routing networks: Adaptive selection of non-linear functions for multi-task learning for ICLR 2018 - IBM Research"
[2]: https://en.wikipedia.org/wiki/Mixture_of_experts?utm_source=chatgpt.com "Mixture of experts"
[3]: https://deepai.org/publication/interpretable-mixture-of-experts-for-structured-data?utm_source=chatgpt.com "Interpretable Mixture of Experts for Structured Data | DeepAI"
[4]: https://proceedings.mlr.press/v267/yang25ag.html?utm_source=chatgpt.com "Mixture of Experts Made Intrinsically Interpretable"
[5]: https://link.springer.com/article/10.1007/s10994-024-06583-2?utm_source=chatgpt.com "FairMOE: counterfactually-fair mixture of experts with levels of interpretability | Machine Learning"
[6]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/7ca55dae829a2305716f34658cd33a97-Abstract-Conference.html?utm_source=chatgpt.com "Shedding Light on Time Series Classification using Interpretability Gated Networks"
[7]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8564499/?utm_source=chatgpt.com "Matrix Profile-Based Interpretable Time Series Classifier - PMC"
[8]: https://link.springer.com/article/10.1007/s10618-023-00969-x?utm_source=chatgpt.com "Z-Time: efficient and effective interpretable multivariate time series classification | Data Mining and Knowledge Discovery"
[9]: https://www.mdpi.com/2076-3417/13/4/2038?utm_source=chatgpt.com "Explainable AI for Machine Fault Diagnosis: Understanding Features’ Contribution in Machine Learning Models for Industrial Condition Monitoring"
[10]: https://pure.hud.ac.uk/en/publications/an-explainable-intelligence-fault-diagnosis-framework-for-rotatin/?utm_source=chatgpt.com "An explainable intelligence fault diagnosis framework for rotating machinery - University of Huddersfield Research Portal"
[11]: https://pubmed.ncbi.nlm.nih.gov/40282637/?utm_source=chatgpt.com "Explainable Fault Classification and Severity Diagnosis in Rotating Machinery Using Kolmogorov-Arnold Networks - PubMed"
[12]: https://www.emergentmind.com/articles/2102.11848?utm_source=chatgpt.com "Explainable AI for Unsupervised Fault Diagnosis"
[13]: https://www.mdpi.com/2079-9292/13/20/4116?utm_source=chatgpt.com "Interpretable Mixture of Experts for Decomposition Network on Server Performance Metrics Forecasting"
[14]: https://www.mdpi.com/1099-4300/27/4/403?utm_source=chatgpt.com "Explainable Fault Classification and Severity Diagnosis in Rotating Machinery Using Kolmogorov–Arnold Networks"
