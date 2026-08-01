# P01 Paired 1D/2D Experiment Configurations

These additive configurations bind paper P01 to the maintained `main.py`
entrypoint. The scientific source of truth is
`paper/experiments/config_bridge.yaml`; this directory provides executable
dataset and arm bases.

## Active files

- `p01_shared_private_cwru.yaml` and `p01_shared_private_xjtu.yaml`: full
  shared-private method.
- `p01_generic_attention_cwru.yaml` and
  `p01_generic_attention_xjtu.yaml`: predeclared primary comparator and base
  for the other `P01Baselines` variants.
- `p01_shared_only_cwru.yaml` and `p01_shared_only_xjtu.yaml`: direct
  private-branch removal.
- `no_local_override.yaml`: explicit empty machine-local override required by
  evidence commands.
- `configs/base/model/p01_shared_private.yaml` and
  `configs/base/model/p01_baseline_family.yaml`: registered model bases.

Older P01 configs for DLinear, ResNet1D, CDDG, few-shot, pretraining, and the
superseded tri-level placeholder remain historical/supporting files. They are
not members of protocol `P01-G040-v1` and cannot support C1–C3.

## Inspection

```bash
conda run -n LQ_signal python -m scripts.config_inspect \
  --config configs/experiments/p01/p01_shared_private_cwru.yaml \
  --dump targets --format yaml
```

## Post-lock smoke shape

```bash
conda run -n LQ_signal python main.py \
  --config configs/experiments/p01/p01_shared_private_cwru.yaml \
  --local_config configs/experiments/p01/no_local_override.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

This shape is an implementation smoke only. Evidence commands additionally
require the approved protocol, a unique arm/dataset/fold/seed/attempt output
path, the frozen split manifest, and exactly one allowed physical GPU.

## Frozen boundary

- Training seeds: `[42, 123, 456, 789, 1024]`.
- Split, pairing, and analysis seed: `20260801`.
- CWRU: four `File`-stratified outer folds. This is source-record-disjoint, not
  bearing-identity-disjoint.
- XJTU: five `FileParent` outer folds, stratified by operating-condition
  `Domain_id`, with binary normal/fault labels.
- Standardization is deterministic per window and has no corpus-level fitted
  state.
- The negative control deranges only training windows within class and group;
  validation and test stay paired.
- Physical GPU index 2 and multi-GPU runs are forbidden.

No smoke, configuration inspection, or unit test is experimental evidence.
