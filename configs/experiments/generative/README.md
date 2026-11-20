# Generative Factory candidates

These configurations exercise ordinary conditional GAN, Gaussian VAE, DDPM and
Conditional Flow Matching (CFM) through the existing public runtime. They are
software fixtures; real-data populations, physical groups, tuning budgets and
paper results remain owned by the downstream study.

## Existing implementations and migration scope

CFM was already implemented on `dev@f105bdb`; this change reuses its model,
velocity loss and Euler sampler. Its shared batch/condition and Lightning methods
are factored into `GenerativeTask` for the four current consumers.

DDPM also existed before this change. The historical
[`Feature_factory-update@ed1bc7d`](https://github.com/PHMbench/PHM-Vibench/tree/ed1bc7d7968cff6a1e0f0cf1c2d1c476054d4cdb/src/task_factory/Components/generative)
and `cleanup/repo-slim-2026-07-05@7b97deb` snapshots contain `ddpm_epsilon`,
epsilon loss, a scheduler, a reverse sampler and YAML examples. They were not
integrated into the current `dev` runtime. The existing
[migration contract](../../../docs/PIPELINE_06_GENERATIVE_MIGRATION.md)
requires a file-level audit of those snapshots before migration.

The current DDPM candidate adapts that existing method capability to the current
Task/condition interface and reuses the current CFM backbone. The historical
sampler selected sparse timesteps with `linspace` but applied an adjacent-step
transition at each selected index; its fixture requested 8 steps from a 100-step
training schedule. The current Task instead requires the complete configured
ancestral schedule. It also records actual NFE and uses the explicitly declared
fixed posterior variance. The variance choice alone is not a defect in the older
implementation. Only `conditional_ddpm` is maintained in this runtime; the old
`ddpm_epsilon` snapshot is not registered as a second implementation.

GAN/VAE Model and Task implementations were not found in the checked `dev`,
`main`, `Feature_factory-update` and `cleanup/repo-slim-2026-07-05` snapshots.
Their entries below add ordinary conditional candidates. This inventory is scoped
to those snapshots, not a claim that no other branch or local checkout contains
an implementation. Source presence, current-runtime integration and PHM benchmark
validity are separate states.

## Executable software matrix

All four use the same bundled Dummy signals, two channels, length 128, batch size
2, four requested windows, per-window standardization and one CPU epoch. Sampling
requests the two observed `(fault_label, domain_id)` pairs `(0,0)` and `(1,1)`.
A Cartesian grid would also request unobserved cross-pairs.

| Config | `model.name` | `task.name` | Sampling |
| --- | --- | --- | --- |
| [GAN](dummy_generative_gan.yaml) | `phm_gan_mlp1d` | `conditional_gan` | Standard Gaussian latent, one generator call |
| [VAE](dummy_generative_vae.yaml) | `phm_vae_mlp1d` | `conditional_vae` | Standard Gaussian latent, one decoder call, Gaussian observation noise |
| [DDPM](dummy_generative_ddpm.yaml) | `phm_ddpm_mlp1d` | `conditional_ddpm` | Full 8-step ancestral fixture |
| [CFM](dummy_generative_cfm.yaml) | `phm_cfm_mlp1d` | `conditional_flow_matching` | 8-step explicit Euler |

Every row uses `model.type=generative_model` and `task.type=generative`.
GAN and VAE use fixed-window MLPs. DDPM reuses the stateless conditional convolutional
backbone of CFM as an epsilon predictor; its diffusion objective and sampler are
implemented in its own Task. These small architectures are implementation
candidates, not reproductions of TimeGAN, TimeVAE or the DDPM image U-Net.

One epoch is a bounded execution check, not an equal-compute research budget.
GAN executes one discriminator and one generator update per batch, so Lightning's
`global_step` advances twice. Model parameter counts, data passes, optimizer
updates and tuning costs must be accounted for separately in a scientific comparison.

## Existing Factory contract

The entrypoint remains `python main.py --config <yaml>`. The public resolver
composes the five existing configuration blocks. Model and Task selection use the
existing [Model Factory](../../../src/model_factory/README.md) and
[Task Factory](../../../src/task_factory/README.md); each new Task is explicitly
mapped to the existing `Default_dataset` adapter.

- Model exports `Model(args_model, metadata)` and owns trainable networks.
- Task owns its objective, optimizer, validation loss and sampler. There is no
  model-level `compute_loss()` protocol.
- The shared `GenerativeTask` contains the previous CFM batch/condition handling
  and Lightning lifecycle. It does not select an objective or introduce a registry.
- Trainer Factory owns fitting and checkpoint selection/restoration. Pipeline 06
  still orchestrates the separate `train`, `sample` and `eval` invocations.

The shared sampling boundary is:

```python
sample(condition, *, num_samples, length, channels, num_steps, device)
```

`condition` contains integer `fault_label` and `domain_id` tensors of length `N`.
The CPU fixtures return finite float32 `[N,C,L]`. Each Task declares `method_id`,
`loss_id`, `sampler_id` and `sampler_metadata()`; the latter records `nfe` for the
selected sampling method. Network forward signatures remain method-specific.
Continuous RPM/load controls are not part of this condition encoder.

## Objectives and sampler choices

| Method | Task objective | Explicit settings and selection criterion |
| --- | --- | --- |
| GAN | Non-saturating generator BCE; real/fake discriminator BCE | Separate G/D parameters and AdamW optimizers; one D and one G update per batch. `val_loss` is generator BCE against the current discriminator, not a distribution-quality estimator. |
| VAE | Gaussian observation NLL + `kl_weight` times analytic Gaussian KL | `observation_std=1`, `kl_weight=1`; both terms sum within each window and average over the batch. Sampling includes the same observation noise as the likelihood. |
| DDPM | Uniform discrete-timestep epsilon MSE | Linear `beta_start` to `beta_end`; fixed posterior variance; last reverse step adds no noise; no image-range clipping. Sampling requires `num_steps == diffusion_steps`. |
| CFM | Existing linear-path velocity MSE | Existing `t_eps=0.001`; Gaussian source; unchanged Euler integration. |

For VAE, weight 1 gives the usual single-sample negative ELBO; other weights select
a beta-weighted objective. GAN and VAE require `num_steps=1`, rather than silently
ignoring a requested multi-step sampler.

DDPM's method default is 1000 steps with betas from `0.0001` to `0.02`. The fixture
uses eight steps to check execution. Its cumulative terminal alpha is about
`0.9222`, so the forward endpoint is not close to a standard Gaussian. That short
schedule is not suitable for judging DDPM quality. The actual cumulative alpha and
variance choice are recorded with sampling metadata.

Checkpoint restoration is verified with the same training configuration.
`strict` checks state-dictionary keys and tensor shapes; it does not establish
that arbitrary configurations are semantically interchangeable. Preserve the
window channels and length, condition encoding and VAE observation likelihood
when reusing a checkpoint. Cross-configuration restoration is not validated.
DDPM additionally rejects checkpoint schedule buffers that disagree with the
configured diffusion steps or beta schedule, before loading network weights.

## Run the public stages

Use the existing end-to-end test to execute all four experiment configurations
and the original CFM demo. It copies bundled inputs to temporary directories,
invokes `main.py` separately for train/sample/eval, restores the selected checkpoint,
checks method identity, conditions and NFE, and verifies that no H5 cache is added
to the checkout:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 python -m pytest \
  test/generative/test_cfm_demo_e_chain.py -q
```

For a manual run, prepare a new caller-owned directory and keep outputs there:

```bash
AIGC_SMOKE_DIR=$(mktemp -d)
mkdir -p "$AIGC_SMOKE_DIR/data/raw/Dummy_Data"
cp data/metadata_dummy.csv "$AIGC_SMOKE_DIR/data/"
cp data/raw/Dummy_Data/dummy1.csv data/raw/Dummy_Data/dummy2.csv \
  "$AIGC_SMOKE_DIR/data/raw/Dummy_Data/"

python main.py --config configs/experiments/generative/dummy_generative_vae.yaml \
  --override data.data_dir="$AIGC_SMOKE_DIR/data" \
  --override environment.output_dir="$AIGC_SMOKE_DIR/vae"
```

The experiment YAML defaults use a caller directory adjacent to the checkout;
prepare that location or pass explicit paths as above. Do not point a paper run
at the upstream repository's data cache or results directories.

Then select `task.generative.mode=sample` using the existing stage result's
`checkpoint_path`, `normalization_path`, `normalization_sha256` and `protocol_path`.
For `mode=eval`, supply the returned `generated_path` and `synthetic_manifest_path`.
The test is the complete executable example. These existing artifact fields are
preserved; no alternate runner, configuration loader or result manager is added.

## Source and implementation scope

The components implement the cited equations under the repository's
[Apache-2.0 license](../../../LICENSE); no external model code is vendored.
The repository implementation history is recorded above, including existing CFM
and historical DDPM. [references.bib](references.bib) records the method references.

| Family | Primary source | Implementation scope |
| --- | --- | --- |
| GAN | [Goodfellow et al., 2014](https://arxiv.org/abs/1406.2661), [Mirza and Osindero, 2014](https://arxiv.org/abs/1411.1784) | Non-saturating loss and conditioning of both networks |
| VAE | [Kingma and Welling](https://arxiv.org/abs/1312.6114) | Reparameterization, Gaussian KL and observation likelihood; label/domain conditioning added here |
| DDPM | [Ho et al., 2020](https://arxiv.org/abs/2006.11239) | Eq. (4), (7), (14), Algorithm 2; continuous-waveform backbone adaptation |
| CFM | [Lipman et al.](https://arxiv.org/abs/2210.02747) | Existing independent Gaussian-to-data linear path and velocity regression |

TimeGAN, TimeVAE, Rectified Flow, MeanFlow and Drifting remain `TODO-ADAPT` for this
current runtime. Historical snapshots contain several flow-family implementations;
the status means that their current-runtime adaptation and method-specific
validation are incomplete, not that their source never existed. One Euler step
does not implement a one-step training objective. Physics-informed paper methods
are supplied by their downstream owner.

## What remains before PHM comparison

The current Pipeline 06 evaluator requires a training-reference population.
Condition-frequency distance does not test waveform controllability, and the
nearest-centroid smoke utility does not measure independent real-test augmentation.
These limitations remain explicit; this change does not promote a scientific result.
The legacy `paper_smoke_metric_eligible` manifest field records whether all smoke
metrics were computable. It can be true for a two-class fixture while
`paper_smoke_ready` and `benchmark_valid` remain false.

A downstream comparison must provide original waveforms, source-verified physical
unit IDs, a fixed train/validation/test assignment, train-only preprocessing,
method and tuning budgets, held-out generation metrics and an independent
real-test augmentation evaluator. Existing `data.split.strategy=grouped_metadata`
can enforce a supplied group key; a filename alone is not evidence of a distinct
bearing. The current CWRU, PU and HUST metadata inventory does not supply that
missing scientific protocol.

Configurations remain unpromoted under `configs/experiments/`. CPU execution alone
does not satisfy the existing CPU/GPU and review requirements for promotion to a
maintained demo or a `baseline_valid` experiment.
