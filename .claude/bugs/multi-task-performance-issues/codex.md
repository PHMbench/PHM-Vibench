Multi‑Task Performance Issues — Codex Report

Summary
- Scope: Single‑epoch multi‑task run (classification, anomaly, signal prediction, RUL) on RTX 3090 using ISFM M_01 with 4 heads.
- Outcome: Classification is strong and stable; anomaly detection collapses under imbalance (very low AUROC); RUL metrics are invalid due to label quality/scale; signal prediction underperforms a simple baseline.
- Priority fixes: Correct labels/targets and shapes per task, rebalance/disable noisy heads, align monitoring, and harden metric computation.

Run Context
- Data: cache.h5 built and used; 8,373 train/val and 8,373 test ID‑window datasets.
- Config: task.enabled_tasks = [classification, anomaly_detection, signal_prediction, rul_prediction]. Model heads match enabled tasks.
- Checkpoint: 1 epoch; path exists; saved best on val_loss.

Key Observations
- Classification: Acc/F1/Precision/Recall ≈ 0.923 → healthy.
- Anomaly: Acc ≈ 0.929 but AUROC ≈ 0.020, F1 ≈ 0.067 → degenerate under severe imbalance/labeling proxy.
- RUL: MAE ≈ 0.527, MAPE ≈ 1.0, R2 ≈ −2641.96 → targets likely invalid/mismatched scale; many defaults.
- Signal: MAE ≈ 0.953, MSE ≈ 1.452, R2 ≈ −0.45 → worse than mean baseline; likely head/target misalignment.
- Warnings: torchmetrics “No positive/negative samples” (anomaly imbalance); Tensor Cores tip; checkpoint dir pre‑exists; monitor/filename inconsistency potential.

Probable Root Causes (by task)
- Anomaly Detection
  - Proxy labeling: current rule anomaly = (label > 0) may not reflect true dataset anomaly flags.
  - Class imbalance: Same‑system batching frequently yields single‑class batches; BCE/metrics degrade; AUROC ~ 0.5 expected for random, 0.02 suggests inversion or score/label mismatch.
- RUL Prediction
  - Missing labels: default fallback used (e.g., 1000.0) contaminates targets; R2 collapses.
  - Scale mismatch: head scales by `rul_max_value` while targets may be raw/unscaled.
  - Batch label extraction: each sample needs its own RUL value (per‑ID), not a batch‑level scalar.
- Signal Prediction
  - Decoder capacity vs target length: large output space with limited supervision; objective not well aligned.
  - Potential target misalignment: pred_len/out_dim vs input `(L,C)` requires consistent slicing/reshape for loss/metrics.
- Metrics/Logging
  - Regression metrics (torchmetrics) flatten with `view(-1)`; non‑contiguous tensors can error without `.contiguous()`/`reshape` (observed earlier).
  - Checkpoint filename uses `{val_loss:.4f}` while experiments may intend `monitor: val_total_loss`.

Fast Triage (next run in <30 min)
1) Isolate classification to validate pipeline
   - YAML: `task.enabled_tasks: ['classification']`
   - Expect classification Acc/F1 ≈ 0.92+; verify checkpoint/monitor alignment.
2) Reintroduce anomaly with proper setup
   - Labels: switch to dataset’s anomaly flag if available (not `y > 0`).
   - Loss: use BCEWithLogits with `pos_weight` estimated from training ratio.
   - Metrics: track AUROC (probabilities), and determine threshold on val set for F1.
3) Temporarily disable or downweight RUL & signal
   - YAML: drop tasks or set task_weights small (e.g., 0.1) until targets verified.
4) Unify monitor/filename
   - Log and monitor the same key (`val_loss` or `val_total_loss`), and align `ModelCheckpoint.filename` with it.

Robust Fix Plan (1–2 dev sessions)
- Anomaly
  - Data: use correct anomaly labels from metadata; oversample or set `pos_weight` for BCE; consider batch composition to ensure both classes.
  - Metrics: primary AUROC on val; derive threshold for F1/precision/recall.
- RUL
  - Filter: exclude samples with missing/invalid RUL; remove default 1000 fallback.
  - Scale: standardize RUL targets (z‑score/min‑max) and invert on logging; ensure head output scale matches.
  - Batch: build per‑sample RUL targets (file‑ID aware).
- Signal
  - Decoder: replace with low‑rank/conv decoder or reduce `hidden_dim/max_len/max_out`; start with realistic `pred_len` (e.g., 96) and `out_dim`.
  - Targets: slice to match prediction `(B, pred_len, out_dim)` before loss/metrics.
- Metrics stability
  - Before torchmetrics: `preds = preds.contiguous(); targets = targets.contiguous()`; optionally `reshape(-1)` for regression metrics.
- Trainer
  - Set `torch.set_float32_matmul_precision('medium')` at startup for 3090 throughput.
  - Ensure `{monitor}` exists in logs; update `filename` template accordingly.

Validation Checklist
- Classification‑only run passes with high Acc/F1.
- Anomaly AUROC > 0.5 on val after label fix/pos_weight; chosen threshold yields reasonable F1.
- RUL: R2 near > 0 (or MAE decreasing) on a subset with valid labels after scaling fix.
- Signal: R2 improves over baseline with aligned `pred_len`/`out_dim`.
- No torchmetrics shape/contiguity errors during regression metrics.

Code Pointers
- Multi‑task task (labels/metrics/loss)
  - `src/task_factory/task/In_distribution/multi_task_phm.py:200` (classification metrics)
  - `src/task_factory/task/In_distribution/multi_task_phm.py:232` (anomaly metrics/AUROC)
  - `src/task_factory/task/In_distribution/multi_task_phm.py:246` (regression metrics: add contiguous/reshape)
  - `src/task_factory/task/In_distribution/multi_task_phm.py:308` (batch‑wise RUL target build)
  - `src/task_factory/task/In_distribution/multi_task_phm.py:402` (signal/RUL loss handling; add target slicing for signal)
- Metrics registry
  - `src/task_factory/Components/metrics.py:26` (MeanSquaredError, etc.; regression metrics do not need num_classes)
- Trainer checkpoint naming
  - `src/trainer_factory/Default_trainer.py:84` (filename uses `{val_loss:.4f}`)

Notes & Tips
- Same‑system sampler: batches can still have mixed `file_id`s. Build per‑sample targets for RUL; do not assume a single metadata row.
- When enabling signal head, ensure model’s `enabled_tasks` mirrors `task.enabled_tasks` so proper head is instantiated.
- For HPC: if h5py SWMR read fails, reopen without SWMR as a fallback.

Appendix — Current Test Metrics (excerpt)
- Classification: acc=f1=precision=recall ≈ 0.923
- Anomaly: acc≈0.929, AUROC≈0.020, F1≈0.067, precision≈0.078, recall≈0.063
- RUL: MAE≈0.527, MAPE≈0.998, MSE≈0.358, R2≈−2641.96
- Signal: MAE≈0.953, MSE≈1.452, R2≈−0.45
