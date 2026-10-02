# Conditional Flow Matching candidate smoke

`dummy_generative_cfm.yaml` uses repository Dummy_Data, the canonical
`Pipeline_06_Generative_Modeling` entrypoint, CFM velocity loss, Euler sampling,
and direct `fault_label`/`domain_id` conditions. Training, sampling, and evaluation
remain separate public invocations.

## Check the complete CPU path

After installing PHMFactory, run from the repository root:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest \
  test/generative/test_euler_request_contract.py \
  test/generative/test_cfm_demo_e_chain.py -q
```

The end-to-end test copies the two bundled CSV signals and metadata into a temporary
data directory, then calls `python main.py --config ...` separately for train, sample,
and eval. It checks checkpoint restoration, finite float32 samples with the declared
shape and condition lengths, complete structured metric results, and absence of new
repository H5 caches. It does not replace training with a mock or download PHM data.

The existing CFM workflow runs this regression alongside component and dispatch tests.
Its CPU PyTorch and torchvision wheels use the same official CPU index.

## Run stages manually

```bash
CUDA_VISIBLE_DEVICES="" python main.py \
  --config configs/demo/10_generative/dummy_generative_cfm.yaml \
  --override trainer.num_epochs=1 \
  --override trainer.device=cpu \
  --override trainer.devices=1 \
  --override data.num_workers=0
```

For `task.generative.mode=sample`, provide the training checkpoint,
`normalization_path`, `normalization_sha256`, and `protocol_path` from the existing
stage result. For `mode=eval`, provide `generated_path` and `synthetic_manifest_path`
from sampling. `test_cfm_demo_e_chain.py` is the executable three-stage example.
These legacy artifact fields remain compatibility inputs; their hashes do not prove
signal quality, train-only preprocessing, or downstream utility.

## Sampling contract and evidence boundary

`sample_euler_ode` requires a positive integer step count and finite increasing time
bounds. It rejects fractional, boolean, and string counts before calling the model;
truncating a fractional count would integrate the wrong interval. Positive integer
calls keep the existing Euler computation and shape/device/dtype checks.

The registry remains `needs_smoke` until current-head execution and review justify an
explicit update. All current claims stay software-only and exploratory. A successful
single-class Dummy run is not TSTR, TRTS, physical-unit generation, condition fidelity,
or PHM benchmark evidence. Metrics that cannot be computed retain their reasons.

No arbitrary-dataset, additional model-family, or accelerator support is implied.
