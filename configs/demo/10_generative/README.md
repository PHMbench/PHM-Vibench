# Conditional Flow Matching smoke demo

`dummy_generative_cfm.yaml` is the maintained Pipeline 06 CFM runtime smoke
for the repository-shipped Dummy_Data fixture. The supported combination is
exactly:

```text
Pipeline_06_generative
+ generative_model/phm_cfm_mlp1d
+ generative/conditional_flow_matching
+ Euler ODE sampler
+ fault_label and domain_id conditions
+ repository Dummy_Data
```

It completed the separate train, sample, and eval stages on CPU and on one
NVIDIA GeForce RTX 4090 with seed 0. `sanity_ok` means the artifact and factory
chain ran successfully; scientific validity remains `exploratory`.

Start the train stage with:

```bash
python main.py \
  --config configs/demo/10_generative/dummy_generative_cfm.yaml \
  --override trainer.num_epochs=1 \
  --override trainer.device=cpu \
  --override trainer.gpus=1 \
  --override data.num_workers=0
```

Sampling is a separate invocation and requires the exact checkpoint plus the
train-only normalization path and hash written to `stage_ledger.json`:

```bash
python main.py \
  --config configs/demo/10_generative/dummy_generative_cfm.yaml \
  --override task.generative.mode=sample \
  --override task.generative.checkpoint_path=<checkpoint.ckpt> \
  --override task.generative.normalization_path=<normalization_params.json> \
  --override task.generative.normalization_sha256=<sha256>
```

Evaluation is also separate and consumes the generated sample and synthetic
manifest paths recorded by the sample stage:

```bash
python main.py \
  --config configs/demo/10_generative/dummy_generative_cfm.yaml \
  --override task.generative.mode=eval \
  --override task.generative.generated_path=<samples.pt> \
  --override task.generative.synthetic_manifest_path=<synthetic_data_manifest.json>
```

Known limits:

- This is a one-epoch functional smoke, not a benchmark-performance result.
- Direct model conditions are only `fault_label` and `domain_id`.
- The smoke generates one condition, so downstream classifier utility is
  present but `not_computable`; its reason is recorded in the evidence manifest.
- Support does not extend to arbitrary data, backbones, samplers, GPUs,
  multi-GPU training, or paper configurations.
