# P09 — HSE-Prompt GFS experiment configs

Paper-bridge configs for **P09-HSE-Prompt-GFS**. These are additive configs composed from
`configs/base/*` using registry-style component IDs (`E_*/B_*/H_*` or the registered
`HSE_prompt` embedding for the ISFM_Prompt family). They are the PHM-Vibench_fix-side runnable
targets that map 1:1 to the experimental arms in this paper's
`paper/experiments/experiment_plan.md` and `config_bridge.yaml`.

P09 is the **GFS fork** of the HSE-Prompt line (the CDDG fork is P08). The central task here is
`task.type: GFS` (generalized few-shot: base + novel classes, scored by the harmonic mean), not
CDDG. The method arm, baselines, k-shot grid, ablations, backbone sweep, and noise sweep are all
run under the GFS protocol.

## Source provenance (READ-ONLY)

The legacy manuscript snapshot (`legacy/source_snapshot/JACS_v2.tex`) is CDDG-flavored per its
own `readme.md` ("CDDG / next is GFS"). P09 re-frames every legacy number under GFS. The configs
here derive (not copy) from:

- demo `configs/demo/04_cross_system_fewshot/cross_system_tspn.yaml`  (GFS task surface)
- demo `configs/demo/03_fewshot/cwru_protonet.yaml`                   (episodic FS baselines)
- demo `configs/demo/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml` (two-stage pretrain)
- demo `configs/demo/06_pretrain_cddg/pretrain_hse_cddg.yaml`         (HSE contrastive view)
- the P08 sibling `configs/experiments/p08/*.yaml`                     (CDDG fork reference)

## Canonical P09 protocol (locked here for the bridge)

- **Datasets (5, LOSO pool):** `[1, 13, 6, 12, 19]` = CWRU, Ottawa-19, THU, JNU, HUST24.
  Same pool as P08. **P1 BLOCKER:** manuscript prose mentions "Ottawa-23" / "XJTU" in places;
  `Dataset_id 5 = Ottawa-23`, `13 = Ottawa-19`, `12 = JNU`. There is no XJTU in this pool.
  Reconcile at PROTOCOL_LOCK (P09-G040).
- **Per-dataset `model.input_dim`** (from P08 run scripts, same metadata): `1->2, 13->2, 6->1, 12->1, 19->3`.
- **Seed policy:** 5 seeds `[42, 123, 456, 789, 999]` (matches P08; supports the manuscript's
  5-seed statistical-rigor claim and the paired-t-test claim). Default `seed: 42`; other 4 via
  `--override environment.seed=<s>`.
- **GFS task surface:** `task.type: "GFS"`, `task.name: "classification"`, with sampler knobs
  `num_episodes / num_support / num_query / num_labels`. k-shot = `task.num_support` swept in
  `{1, 5, 10, 20}`.
- **Pipeline:** single-stage GFS arms use `Pipeline_01_default`; pretrain arms use
  `Pipeline_02_pretrain_fewshot` with a `stages:` list (pretrain -> GFS finetune).

## Smoke-friendly defaults

Every config here defaults to **1 epoch** so `python main.py --config <yaml>` is a fast smoke test.
Real training budgets are in `paper/experiments/experiment_plan.md` and applied at launch via
`--override trainer.num_epochs=<N>` (and `stages[k].trainer.num_epochs=<N>` for two-stage arms).

## Leakage flags (must be resolved at P09-G040 / G050 before ANY claim promotion)

1. **GFS base/novel label mapping (P0, P09-G040).** The legacy draft uses CDDG transfer pairs
   ("CWRU->XJTU"). For GFS, per target system s* we must define which labels are *base* vs
   *novel* on s*. **This mapping is NOT encoded** in `src/data_factory/dataset_task/GFS/` or in
   any config knob. The maintained GFS sampler does not currently expose base/novel flags.
   Until P09-G040 produces an accepted base/novel split, **every** GFS accuracy claim is `blocked`.
2. **LOSO target leakage (P0, P09-G050).** The pretrain pool currently includes s*; the few-shot
   arms draw support/query from s*. The headline harmonic-mean claims require a verified
   leave-one-system-out split with s* held out from pretraining.
3. **Normalization leakage.** `data.normalization: standardization` is fit on the full dataset;
   must be refit on the source pool only.
4. **Model-selection-on-target.** `early_stopping` monitors `val_loss`; val must come from the
   source pool, not the target.
5. **GFS sampler support/query disjointness.** The maintained GFS sampler
   (`src/data_factory/dataset_task/GFS/Classification_dataset.py`) is a thin wrapper; whether
   support and query are disjoint per episode, and whether base/novel classes are correctly
   partitioned, is UNVERIFIED.
6. **Noise injector consumer (E7).** `grep` of `src/data_factory` and `src/task_factory` confirms
   NEITHER consumes `data.noise_injection` / `task.noise_robustness`. E7 is
   `needs_new_component` until a noise transform is registered.

## Usage

```bash
# Smoke (1 epoch, default seed 42)
python main.py --config configs/experiments/p09/p09_exp3_hse_prompt.yaml

# Real launch with seed sweep (method arm, representative)
for s in 42 123 456 789 999; do
  python main.py --config configs/experiments/p09/p09_exp3_hse_prompt.yaml \
    --override environment.seed=$s \
    --override stages[0].trainer.num_epochs=100 \
    --override stages[1].trainer.num_epochs=50
done
```

See `paper/experiments/experiment_plan.md` and `paper/experiments/config_bridge.yaml` in the
destination repo for the full per-arm plan.
