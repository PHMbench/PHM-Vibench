# 2026-09-21: B01 S1 candidate audit

After B−1 and B00, the next model-catalogue action audits S1 supervised-classification candidates without adding runtime code.

TimesNet classification from THUML Time-Series-Library at `4e938a1767106324dd753b2a44832bf870a0252e` is the preferred first S1 candidate because it adds frequency-selected periodic 2-D variation modeling beyond the current TCN/ResNet and conventional Transformer families. It remains AUDITED rather than QUEUED: current-dev fidelity, candidate-specific resource evidence and an implementation WIP slot are still required.

ModernTCN, TSLANet and iTransformer are not rejected. ModernTCN currently overlaps the convolutional role; TSLANet has a higher fidelity/dependency burden; iTransformer's official main repository is forecasting-only, so the exact classification variant/source must be resolved.

No model, Task, config, test, workflow or runtime registry is added in this documentation-only audit.
