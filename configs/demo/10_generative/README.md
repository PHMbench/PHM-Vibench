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

The existing CFM workflow runs this regression alongside component and dispatch tests
for relevant pull requests and pushes to `dev`. Its CPU PyTorch and torchvision wheels
use the same official CPU index.

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
bounds. The Task and public sample stage preserve the requested count for validation:
fractional, boolean, and string counts fail before sampler model evaluation, instead
of being converted to integers. Truncating a fractional count would integrate the wrong
interval. Positive integer calls keep the existing Euler computation and
shape/device/dtype checks.

## Method alignment

| Path | Current implementation and reference boundary |
| --- | --- |
| Vanilla CFM | Linear noise-to-data path `x_t=(1-t)z+t*x_1`, velocity MSE against `x_1-z`, and Euler sampling; see the [Flow Matching Guide](https://arxiv.org/abs/2412.06264). |
| Population-aware CFM experiment | Adds MMD on per-window channel-correlation features with a shared batch time. This adapts the population-dependency idea of [PaD-TS (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/34038) to PHM CFM; it does not reproduce PaD-TS's full DDPM method. |
| Few-step reference | [iMF](https://arxiv.org/abs/2512.02012) is not integrated into this CFM path. Setting Euler to one step does not implement the MeanFlow training objective or establish its quality. |

The population variant remains under `configs/experiments/generative/`. Its configured
`batch_size >= 2` does not guarantee that each `Dataset_id` batch has two windows: a
single-window tail batch still fails the population MMD contract. No tail samples are
silently discarded to avoid this failure.

The smoke metrics have narrower meanings than a scientific benchmark: condition
consistency compares label/domain frequencies, not waveform condition fidelity, and
nearest-centroid utility is not an independently held-out diagnostic benefit. The
normalization artifact summarizes processed training windows; it is not an inverse
transform to original physical units.

The registry remains `needs_smoke` until current-head execution and review justify an
explicit update. All current claims stay software-only and exploratory. A successful
single-class Dummy run is not TSTR, TRTS, physical-unit generation, condition fidelity,
or PHM SOTA evidence. Metrics that cannot be computed retain their reasons.

No arbitrary-dataset, additional model-family, or accelerator support is implied.

## Factory interface

Pipeline 06 reuses the existing Data, Model, Task and Trainer factories. Its public
entrypoint is `python main.py --config <yaml>`; `task.generative.mode` selects one
of `train`, `sample` or `eval`. No extra runner or registry is needed.

| Owner | Interface consumed by Pipeline 06 |
| --- | --- |
| Data Factory | `build_data(args_data, args_task)`, `get_metadata()` and `get_dataloader(split)`; retain the existing reader and physical-record identities. |
| Model Factory | `Model(args_model, metadata)` from the configured `model.type`/`model.name`, or explicit `model.module`. The current CFM model maps `(x_t, t, condition)` to a velocity of shape `[N,C,L]`. |
| Task Factory | A registered Lightning Task selected by `task.type`/`task.name`; it owns the loss, optimizers, validation loss and the model-specific forward call. |
| Sampling Task | `sample(condition, *, num_samples, length, channels, num_steps, device)` returns finite float32 samples `[N,C,L]`. The Task exposes `method_id`, `loss_id`, `sampler_id` and `sampler_metadata()`. |
| Trainer Factory | The existing Trainer fits the Task and selects/restores a checkpoint. GAN optimizer scheduling belongs in its Lightning Task. |

`condition` has `fault_label` and `domain_id`, both integer tensors of length `N`.
The current Task obtains them from batch fields or `Label`/`Domain_id` metadata
using `file_id`. A domain identifier is not automatically an RPM, load or severity
measurement. The present encoder does not consume continuous physical conditions.
Use `explicit` sampling conditions for observed label/domain pairs; a Cartesian
`grid` can request combinations absent from training metadata.

Model forward signatures can differ between method families. A GAN generator, VAE
decoder and DDPM denoiser should keep their own Task objectives rather than masquerade
as CFM velocity models. The common boundary is the Task sampling result and the
existing Trainer lifecycle. Set `num_steps` according to the selected sampler and
record its actual network evaluations; one Euler step does not implement MeanFlow.

For a caller-owned method, this is a **configuration fragment**, requiring the two
named modules to be installed in the caller's environment:

```yaml
pipeline: Pipeline_06_Generative_Modeling
model:
  module: example_extension.model
task:
  type: generative
  name: example_generator
  factory_module: example_extension.factory
  generative:
    mode: train
```

The Factory module registers the Task with `register_task("generative",
"example_generator")`. It also maps that name through the existing
`register_dataset_adapter` interface to a compatible existing dataset adapter.
Pipeline 06 imports this module before building data in every stage. A failed
import is an error. `task.module` instead belongs to the separate scientific
`research` execution form and is not a Lightning Task selector. See
[runtime ownership](../../../docs/RUNTIME_BOUNDARY.md).

## First comparison candidates

Conditional GAN, Gaussian VAE, DDPM and the existing CFM now have built-in Model/Task
candidates. Their exact configuration names, objectives, sampling contracts,
historical implementation sources and bounded software evidence are maintained
in the [generative candidate guide](../../experiments/generative/README.md).
CFM is reused; DDPM adapts an existing historical capability to the current
runtime. No Factory layer is recreated. The external hook above remains available
for caller-owned methods. Dataset populations, splits, budgets and results belong
in the caller's paper repository.

TimeGAN and TimeVAE are separate adaptations of their published methods, not names
for an arbitrary GAN or VAE. Rectified Flow, MeanFlow, Drifting and physics-informed
methods remain outside this first comparison until their current-runtime Task
semantics are adapted and checked. Forecasting checkpoints do not become unconditional
generators by changing a configuration label.

The following YAML is a **planning matrix inside this documentation**, not a second
configuration schema and not input accepted by `main.py`. A null config means that
no runnable configuration is supplied.

```yaml
comparison_candidates:
  - dataset: bundled_Dummy_Data
    task: generation_software_smoke
    methods: [conditional_flow_matching]
    config: configs/demo/10_generative/dummy_generative_cfm.yaml
    modes: [train, sample, eval]
    evidence_scope: software_only
  - dataset: RM_001_CWRU
    task: conditional_generation
    methods: [conditional_gan, conditional_vae, conditional_ddpm, conditional_flow_matching]
    config: null
    status: TODO_ADAPT_AND_QUALIFY_DATA
  - dataset: RM_001_CWRU
    task: augmentation_on_independent_real_test_records
    methods: [real_only, real_plus_gan, real_plus_vae, real_plus_ddpm, real_plus_cfm]
    config: null
    status: TODO_HELD_OUT_UTILITY
  - dataset: [RM_027_PU, HUST_selected_reader]
    task: independent_dataset_replication
    methods: [conditional_gan, conditional_vae, conditional_ddpm, conditional_flow_matching]
    config: null
    status: TODO_METADATA_AND_PROTOCOL
```

For the real-data rows, first identify independent physical records or specimens in
the supplied metadata, then use the existing `data.split.strategy=grouped_metadata`
with its explicit group key and split manifest. Do not invent record IDs, bearing
IDs, RPM labels or severity supervision. Every compared method must see the same
training records, channels, window length, preprocessing and available conditions.
Fix the training and tuning budgets, checkpoint criterion, sampling population and
evaluation seeds before testing; equal epoch counts alone do not establish equal
compute. Report parameter counts, optimizer updates, wall time, hardware, sampling
latency and NFE rather than claiming fairness from a method name.

The current eval stage requires `task.generative.eval_split=train`; it cannot yet
measure held-out fidelity or independent real-test augmentation utility. Its
condition-frequency score does not test waveform controllability. Real-data rows
above therefore remain pending even if the CFM train/sample stages run. The Dummy
normalization and metric choices must not be copied into a scientific protocol
without examining their effect on physical amplitudes and the evaluation population.
