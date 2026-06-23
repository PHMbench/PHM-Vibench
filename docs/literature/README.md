# 2025+ PHM Literature Map

This directory tracks recent PHM work that can inform future PHM-Vibench
baselines, tasks, losses, and paper comparisons. The source of truth is
`phm_2025_plus.csv`; this README exposes the references requested by the goal.

UXFD paper-specific TOP-source citation and reproduction status is tracked in
`paper/UXFD_paper/goal/08_recent_work_citation_readme.md`. This broad PHM
inventory is not sufficient for UXFD core related work, baseline, novelty, or
SOTA positioning.

Validation:

```bash
python -m scripts.phm_literature_matrix --min-count 50
python -m pytest -q test/test_phm_literature_matrix.py
```

Current inventory: 58 works from 2025 or later.

Support status is intentionally conservative:

- `represented`: a broad method family is already represented by an existing
  PHM-Vibench surface, but the exact paper method is not claimed as reproduced.
- `candidate-baseline`: useful candidate for future implementation/comparison.
- `literature-only`: reference for context, paper writing, or future design.
- `dependency-blocked`: requires a dependency not currently available.
- `unsupported`: outside current runtime scope.

## References

| ID | Year | Reference | Task | Method | Status |
|---|---:|---|---|---|---|
| PHM2025-001 | 2025 | Liu et al., [Enhanced Mamba model with multi-head attention mechanism and learnable scaling parameters for remaining useful life prediction](https://www.nature.com/articles/s41598-025-91815-1), Scientific Reports. | `rul` | `mamba` | `candidate-baseline` |
| PHM2025-002 | 2025 | Qiao et al., [A Comparative Study of Deep Learning Model Based Equipment Fault Diagnosis and Prognosis](https://papers.phmsociety.org/index.php/ijphm/article/view/4254), International Journal of Prognostics and Health Management. | `fault_diagnosis` | `benchmark_review` | `literature-only` |
| PHM2025-003 | 2025 | Wen et al., [A generalized diffusion model for remaining useful life prediction with uncertainty](https://link.springer.com/article/10.1007/s40747-024-01773-w), Complex & Intelligent Systems. | `rul` | `diffusion` | `candidate-baseline` |
| PHM2025-004 | 2025 | Niu et al., [Hybrid Gaussian process regression with temporal feature extraction for partially interpretable remaining useful life interval prediction in Aeroengine prognostics](https://www.nature.com/articles/s41598-025-88703-z), Scientific Reports. | `rul` | `gaussian_process` | `literature-only` |
| PHM2025-005 | 2025 | Chen and Fang, [Remaining useful life prediction of lithium-ion batteries via spatial attention TLSTM and dilated CNN with evolutionary optimization](https://www.nature.com/articles/s41598-025-17610-0), Scientific Reports. | `rul` | `rnn` | `represented` |
| PHM2025-006 | 2025 | Cheng et al., [An adaptive dual distillation framework for efficient remaining useful life prediction](https://link.springer.com/article/10.1007/s40747-025-01886-w), Complex & Intelligent Systems. | `rul` | `distillation` | `candidate-baseline` |
| PHM2025-007 | 2025 | Cui et al., [Leveraging Pre-Trained GPT Models for Equipment Remaining Useful Life Prognostics](https://www.mdpi.com/2079-9292/14/7/1265), Electronics. | `rul` | `llm` | `literature-only` |
| PHM2025-008 | 2025 | Yang et al., [Deep multiscale feature fusion network with dual attention for rolling bearing remaining useful life prediction](https://www.nature.com/articles/s41598-025-97380-x), Scientific Reports. | `rul` | `attention` | `represented` |
| PHM2025-009 | 2025 | Dersin et al., [Analytical Health Indices: Towards Reliability-Informed Deep Learning for PHM](https://www.papers.phmsociety.org/index.php/ijphm/article/view/4262), International Journal of Prognostics and Health Management. | `health_indicator` | `physics_informed` | `literature-only` |
| PHM2025-010 | 2025 | Liu et al., [Multi-Condition Remaining Useful Life Prediction Based on Mixture of Encoders](https://www.mdpi.com/1099-4300/27/1/79), Entropy. | `rul` | `transformer` | `candidate-baseline` |
| PHM2025-011 | 2025 | Cha et al., [Large Language Model-Based Autonomous Agent for Prognostics and Health Management](https://www.mdpi.com/2075-1702/13/9/831), Machines. | `phm_agent` | `llm` | `literature-only` |
| PHM2025-012 | 2025 | Dinten and Zorrilla, [Using Time Series Foundation Models for Few-Shot Remaining Useful Life Prediction of Aircraft Engines](https://www.sciencedirect.com/org/science/article/pii/S1526149225002085), Computer Modeling in Engineering & Sciences. | `rul` | `foundation_model` | `candidate-baseline` |
| PHM2025-013 | 2025 | Ren et al., [PHM-GPT: A Large Language Model for Prognostics and Health Management](https://www.sciencedirect.com/science/article/pii/S2095809925006745), Engineering. | `phm_agent` | `llm` | `literature-only` |
| PHM2025-014 | 2025 | Solis-Martin and Galan-Paez, [CONELPABO: composite networks learning via parallel Bayesian optimization to predict remaining useful life in predictive maintenance](https://link.springer.com/article/10.1007/s00521-025-10995-z), Neural Computing and Applications. | `rul` | `nas` | `candidate-baseline` |
| PHM2025-015 | 2025 | Tefera et al., [Constraint-Guided Learning of Data-driven Health Indicator Models](https://papers.phmsociety.org/index.php/ijphm/article/view/4268), International Journal of Prognostics and Health Management. | `health_indicator` | `physics_informed` | `literature-only` |
| PHM2025-016 | 2025 | Solis-Martin et al., [difLIME: Enhancing Explainability with a Diffusion-Based LIME Algorithm for Predictive Maintenance](https://papers.phmsociety.org/index.php/ijphm/article/view/4166), International Journal of Prognostics and Health Management. | `explainability` | `diffusion` | `candidate-baseline` |
| PHM2025-017 | 2026 | Zhang et al., [Bearing fault diagnosis based on multi-branch enhanced GhostNet with adaptive focal loss](https://www.nature.com/articles/s41598-026-49801-8), Scientific Reports. | `fault_diagnosis` | `cnn` | `represented` |
| PHM2025-018 | 2025 | Pang and Li, [Bearing fault detection with lightweight feature extraction mechanism based on smoothed dilated convolution](https://www.nature.com/articles/s41598-025-31960-9), Scientific Reports. | `fault_diagnosis` | `cnn` | `represented` |
| PHM2025-019 | 2025 | Jaiswal et al., [Fault analysis on deep groove ball bearing using ResNet50 and AlexNet50 algorithms](https://www.nature.com/articles/s41598-025-97410-8), Scientific Reports. | `fault_diagnosis` | `cnn` | `represented` |
| PHM2025-020 | 2025 | Chen et al., [Multi-fault diagnosis and damage assessment of rolling bearings based on IDBO-VMD and CNN-BiLSTM](https://www.nature.com/articles/s41598-025-17177-w), Scientific Reports. | `fault_diagnosis` | `signal_processing` | `candidate-baseline` |
| PHM2025-021 | 2025 | Zhao et al., [Multi scale convolutional neural network combining BiLSTM and attention mechanism for bearing fault diagnosis under multiple working conditions](https://www.nature.com/articles/s41598-025-96137-w), Scientific Reports. | `fault_diagnosis` | `attention` | `represented` |
| PHM2025-022 | 2025 | Li et al., [Rolling bearing fault diagnosis in noisy environments using Channel-Time parallel attention networks](https://www.nature.com/articles/s41598-025-22683-y), Scientific Reports. | `fault_diagnosis` | `attention` | `represented` |
| PHM2025-023 | 2026 | Wu et al., [A fault diagnosis method for complex systems based on hierarchical belief rule base with One-vs-Rest strategy](https://www.nature.com/articles/s41598-026-46214-5), Scientific Reports. | `fault_diagnosis` | `rule_based` | `literature-only` |
| PHM2025-024 | 2025 | Anwarsha and Babu, [Fault detection of taper roller bearings using tunable Q-factor wavelet transform and fault classification using long-short-term memory network](https://www.nature.com/articles/s41598-025-93514-3), Scientific Reports. | `fault_diagnosis` | `signal_processing` | `candidate-baseline` |
| PHM2025-025 | 2025 | Guan et al., [Power transformer fault diagnosis method based on multi source signal fusion and fast spectral correlation](https://www.nature.com/articles/s41598-025-91428-8), Scientific Reports. | `fault_diagnosis` | `signal_processing` | `candidate-baseline` |
| PHM2025-026 | 2025 | Bian et al., [Rolling bearing fault diagnosis under small sample conditions based on WDCNN-BiLSTM Siamese network](https://www.nature.com/articles/s41598-025-12370-3), Scientific Reports. | `few_shot` | `few_shot` | `represented` |
| PHM2025-027 | 2025 | Dong et al., [Bearing fault diagnosis method based on WSST and ISSA-MCNN-BIGRU](https://www.nature.com/articles/s41598-025-25469-4), Scientific Reports. | `fault_diagnosis` | `signal_processing` | `candidate-baseline` |
| PHM2025-028 | 2025 | Petrosian et al., [Lightweight bearing fault diagnosis via decoupled distillation and low rank adaptation](https://www.nature.com/articles/s41598-025-06734-y), Scientific Reports. | `fault_diagnosis` | `distillation` | `candidate-baseline` |
| PHM2025-029 | 2025 | Chen et al., [A cycle-aware and physics-informed framework for battery remaining useful life prediction](https://www.nature.com/articles/s41598-025-28505-5), Scientific Reports. | `rul` | `physics_informed` | `candidate-baseline` |
| PHM2025-030 | 2026 | Han and Mo, [Prediction of remaining useful life for electronic equipment based on online PINN](https://www.nature.com/articles/s41598-025-32497-7), Scientific Reports. | `rul` | `physics_informed` | `candidate-baseline` |
| PHM2025-031 | 2025 | Sun et al., [Remaining useful life prediction of lithium batteries based on jump connection multi-scale CNN](https://www.nature.com/articles/s41598-025-08619-6), Scientific Reports. | `rul` | `cnn` | `represented` |
| PHM2025-032 | 2026 | Shi et al., [Enhanced remaining useful life prediction of lithium-ion battery based on a dual attention hybrid data-driven method](https://www.nature.com/articles/s41598-025-30849-x), Scientific Reports. | `rul` | `attention` | `represented` |
| PHM2025-033 | 2026 | Shen et al., [A novel dual-dimensional contrastive self-supervised learning-based framework for rolling bearing remaining useful life prediction](https://www.nature.com/articles/s41598-026-38417-7), Scientific Reports. | `rul` | `contrastive` | `represented` |
| PHM2025-034 | 2025 | Wang et al., [Nonlinear degradation modeling and remaining useful life prediction for electric drive system with multiple failure modes](https://www.nature.com/articles/s41598-025-22866-7), Scientific Reports. | `rul` | `uncertainty` | `literature-only` |
| PHM2025-035 | 2025 | Li et al., [Degradation modeling and remaining useful life prediction for electronic device under multiple stress influences](https://www.nature.com/articles/s41598-025-03786-y), Scientific Reports. | `rul` | `uncertainty` | `literature-only` |
| PHM2025-036 | 2025 | Ibrahim et al., [Hybrid optimized remaining useful life prediction framework for lithium-ion batteries with limited data samples](https://www.nature.com/articles/s41598-025-26743-1), Scientific Reports. | `rul` | `optimization` | `candidate-baseline` |
| PHM2025-037 | 2025 | Fan et al., [A hybrid approach for lithium-ion battery remaining useful life prediction using signal decomposition and machine learning](https://www.nature.com/articles/s41598-025-92262-8), Scientific Reports. | `rul` | `signal_processing` | `candidate-baseline` |
| PHM2025-038 | 2025 | Wang et al., [Rolling bearing remaining useful life prediction using deep learning based on high-quality representation](https://www.nature.com/articles/s41598-025-93165-4), Scientific Reports. | `rul` | `representation_learning` | `candidate-baseline` |
| PHM2025-039 | 2025 | Cao et al., [Attention-Gaussian-LSTM-Wiener based remaining useful life prediction method](https://link.springer.com/article/10.1007/s43684-025-00105-0), Autonomous Intelligent Systems. | `rul` | `rnn` | `candidate-baseline` |
| PHM2025-040 | 2025 | Mao et al., [Efficient Architecture Search for Remaining Useful Life Prediction Using Rainflow Counting Features](https://link.springer.com/article/10.1007/s11424-025-3210-z), Journal of Systems Science and Complexity. | `rul` | `nas` | `candidate-baseline` |
| PHM2025-041 | 2025 | Guo et al., [Few-shot cross-domain fault diagnosis via adversarial meta-learning](https://www.nature.com/articles/s41598-025-25854-z), Scientific Reports. | `few_shot` | `domain_adaptation` | `represented` |
| PHM2025-042 | 2025 | Liu et al., [Bearing fault diagnosis based on cross image multi-attention mechanism](https://www.nature.com/articles/s41598-025-07562-w), Scientific Reports. | `fault_diagnosis` | `attention` | `represented` |
| PHM2025-043 | 2025 | Liu et al., [A novel temporal classification prototype network for few-shot bearing fault detection](https://www.nature.com/articles/s41598-025-98963-4), Scientific Reports. | `few_shot` | `few_shot` | `represented` |
| PHM2025-044 | 2025 | Sun et al., [A bearing fault diagnosis method for hydrodynamic transmissions integrating few-shot learning and transfer learning](https://www.nature.com/articles/s41598-025-04543-x), Scientific Reports. | `few_shot` | `domain_adaptation` | `represented` |
| PHM2025-045 | 2025 | Wang et al., [Auto-embedding transformer under multi-source information fusion for few-shot fault diagnosis](https://www.nature.com/articles/s41598-025-10124-9), Scientific Reports. | `few_shot` | `transformer` | `candidate-baseline` |
| PHM2025-046 | 2025 | Li et al., [An intelligent fault diagnosis model for bearings with adaptive hyperparameter tuning in multi-condition and limited sample scenarios](https://www.nature.com/articles/s41598-025-92838-4), Scientific Reports. | `fault_diagnosis` | `optimization` | `candidate-baseline` |
| PHM2025-047 | 2025 | Tang and Chen, [TFDFNet: a dual-branch fault diagnosis model for bearings under noisy and complex industrial environments](https://www.nature.com/articles/s41598-025-19258-2), Scientific Reports. | `fault_diagnosis` | `multimodal_fusion` | `candidate-baseline` |
| PHM2025-048 | 2025 | Zhang et al., [A hybrid approach combining deep learning and signal processing for bearing fault diagnosis under imbalanced samples and multiple operating conditions](https://www.nature.com/articles/s41598-025-98138-1), Scientific Reports. | `fault_diagnosis` | `domain_adaptation` | `candidate-baseline` |
| PHM2025-049 | 2025 | Lei et al., [A fault diagnosis method for rolling bearings in open-set domain adaptation with adversarial learning](https://www.nature.com/articles/s41598-025-88353-1), Scientific Reports. | `domain_generalization` | `domain_adaptation` | `candidate-baseline` |
| PHM2025-050 | 2025 | Liu et al., [Enhancing unsupervised bearing fault diagnosis through structured prediction in latent subspace](https://www.nature.com/articles/s41598-025-26013-0), Scientific Reports. | `anomaly_detection` | `unsupervised` | `candidate-baseline` |
| PHM2025-051 | 2025 | Parmar et al., [Advanced deep learning approach for the fault severity classification of rolling-element bearings](https://www.nature.com/articles/s41598-025-16895-5), Scientific Reports. | `fault_diagnosis` | `signal_processing` | `candidate-baseline` |
| PHM2025-052 | 2025 | Yang et al., [Discriminative fault diagnosis transfer learning network under joint mechanism](https://www.nature.com/articles/s41598-025-93996-1), Scientific Reports. | `domain_generalization` | `domain_adaptation` | `candidate-baseline` |
| PHM2025-053 | 2025 | Chen et al., [Domain Generalization for Bearing Fault Diagnosis via Meta-Learning with Gradient Alignment and Data Augmentation](https://www.mdpi.com/2075-1702/13/10/960), Machines. | `domain_generalization` | `domain_generalization` | `candidate-baseline` |
| PHM2025-054 | 2025 | Bai et al., [Unsupervised multiple-target domain adaptation for bearing fault diagnosis](https://www.sciencedirect.com/science/article/pii/S0952197625010644), Engineering Applications of Artificial Intelligence. | `domain_generalization` | `domain_adaptation` | `candidate-baseline` |
| PHM2025-055 | 2025 | Yan et al., [Numerical Simulation Data-Aided Domain-Adaptive Generalization Method for Fault Diagnosis](https://www.mdpi.com/1424-8220/25/11/3482), Sensors. | `domain_generalization` | `domain_generalization` | `candidate-baseline` |
| PHM2025-056 | 2025 | Huo et al., [A novel domain generalization network integrating invariance and cohesiveness for rolling bearing fault diagnosis](https://journals.sagepub.com/doi/abs/10.1177/14759217251384919), Structural Health Monitoring. | `domain_generalization` | `domain_generalization` | `candidate-baseline` |
| PHM2025-057 | 2025 | Zhi et al., [An unsupervised transfer learning bearing fault diagnosis method based on multi-channel calibrated Transformer with shiftable window](https://journals.sagepub.com/doi/abs/10.1177/14759217251324671), Structural Health Monitoring. | `domain_generalization` | `transformer` | `candidate-baseline` |
| PHM2025-058 | 2025 | Huang et al., [A dual-perspective joint domain generalization network for bearing fault diagnosis under unseen working conditions](https://www.sciencedirect.com/science/article/pii/S1474034625003404), Advanced Engineering Informatics. | `domain_generalization` | `domain_generalization` | `candidate-baseline` |
