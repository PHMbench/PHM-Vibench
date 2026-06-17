# 13. Final Repo Target for a Comprehensive Paper

## User command flow

```bash
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only

python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --dry-run

python main.py --config configs/paper/phm_generative/cfm_train_grid_seed0.yaml \
  --override environment.seed=0 \
  --override task.target_system_id=[1] \
  --override trainer.device=cuda

python main.py --config configs/paper/phm_generative/cfm_train_grid_seed0.yaml \
  --override task.generative.mode=sample \
  --override task.generative.checkpoint_path=<resolved.ckpt>

python main.py --config configs/paper/phm_generative/cfm_train_grid_seed0.yaml \
  --override task.generative.mode=eval \
  --override task.generative.generated_path=<resolved.samples.pt>

python -m scripts.paperpack_generative --run_dir <eval_run_dir> --stage_ledger <stage_ledger.json>

python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --from-runs results/paper/phm_generative/six_dataset_submission_v1/runs \
  --output results/paper/phm_generative/six_dataset_submission_v1/effect

python -m scripts.generative_submission_draft \
  --summary results/paper/phm_generative/six_dataset_submission_v1/effect/benchmark_effect_summary.csv \
  --manifest results/paper/phm_generative/six_dataset_submission_v1/effect/benchmark_effect_manifest.json \
  --output specs/002-phm-genbench-frontier/paper/PAPER_DRAFT.md
```

## Final paper artifact tree

```text
results/paper/phm_generative/six_dataset_submission_v1/
  run_plan.csv
  run_status_ledger.csv
  runs/
    <dataset>/<method>/seed_<seed>/
      train/
      sample/
      eval/
      stage_ledger.json
      paperpack/
  effect/
    benchmark_effect_summary.csv
    benchmark_effect_manifest.json
  paper/
    PAPER_DRAFT.md
    evidence_gaps.md
    submission_readiness.md
```

## Submission readiness definition

A draft can be marked `SUBMISSION_READY` only when:

```text
- configured dataset count >= 6
- observed configured dataset count >= 6
- every reported row is benchmark-valid
- quality and utility evidence exist for each ready dataset
- metric source paths are present
- manifest source paths are present
- no required primary metric is silently missing
```

## Final paper table groups

```text
Table 1: dataset/protocol summary
Table 2: method/backbone/loss summary
Table 3: generation quality
Table 4: downstream utility
Table 5: leakage/memorization audit
Table 6: efficiency and NFE
Table 7: ablation summary
Appendix: missing metric audit
Appendix: per-dataset/condition metrics
Appendix: run manifest completeness
```
