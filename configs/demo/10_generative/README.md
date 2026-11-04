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

## Multi-class simulation and validation references

Reuse the explicit simulated data prepared by `scripts/prepare_simulated_phm.py`.
The separate configuration does not change the bundled single-class smoke:

```bash
python main.py --config configs/experiments/phmfactory/simulated_cfm.yaml
```

Use the same three-stage procedure above with this configuration. It trains for two
epochs on 162 normalized windows and samples 54 windows across the nine requested
class/condition pairs. `task.generative.eval_split=val` uses the complete validation
loader: 54 non-overlapping validation windows in this configuration, regardless of
how many synthetic samples are requested. The returned stage records the reference
split and both population counts. `eval_split=train` retains its existing bounded
training-reference smoke. Final `test` evaluation requires the separately declared
group-disjoint configuration below; it is rejected for the legacy window split.

These are held-out windows from the same simulated records and conditions, not
independent real assets. Validation is also used for checkpoint selection and is not
an untouched final test. The simulation uses per-window standardized signals; it does
not establish physical-unit synthesis. Distances are relative to the declared reference;
a nearest-neighbor distance to validation signals is not a training-copying test.
The nearest-centroid metric remains a proxy, not a matched augmentation experiment.

A completed seed-17 local run produced finite [54,2,128] samples and eight computable
smoke metrics. Relabelling those fixed samples cyclically, while retaining signals
and class/domain counts, kept condition-frequency distance at zero and nearest-centroid
accuracy at 1/3. This null control does not support a condition-dependent diagnostic
benefit. Runtime execution and generation efficacy remain separate claims.

### Exact population and count contract

The simulation config states the six setting declarations at its header. Its label/domain
conditions are legitimate generation queries, while validation loss selects the checkpoint.
It establishes a development-reference execution path, not the final-evaluation isolation
required by the full F10/I12 contract. Relabelling generated samples is an association
negative control; it is not a test that evaluator-only labels cannot influence fitting.
No family-wide L2 or post-merge capability is inferred from this one run.

For `grid` or `explicit` sampling, the expanded condition count must equal
`task.generative.num_samples`. A request for 54 samples with a nine-row condition grid
previously sampled nine silently; it now fails before calling the sampler. Set both
`num_samples=9` and `condition_grid.samples_per_condition=1` to request the smaller
three-class/three-domain grid deliberately. Existing matching requests are unchanged.

A separate sampling/evaluation check reused the previously trained seed-17 checkpoint,
requested those nine samples, and evaluated with `data.batch_size=1`. The evaluation
used 54/54 validation windows over 54 batches, confirming that neither the generated
count nor the legacy 32-batch training-reference limit truncates validation. No model
was retrained or selected for this check, and the eight required smoke metrics were
computable. The original 54-sample result and its negative control remain unchanged.


## Group-disjoint final evaluation (explicit simulation)

The separate configuration `configs/experiments/phmfactory/simulated_cfm_holdout.yaml`
uses the existing read-only grouped split, not a second loader. For each class,
simulated unit 0 supplies training, unit 1 supplies validation, and unit 2 supplies
final evaluation. Each partition contains nine records and 72 windows; their simulated
asset IDs and raw records are disjoint. All three conditions occur in every partition.
The earlier validation-window configuration and its results are unchanged.

```bash
python -m scripts.prepare_simulated_phm --output data/simulated_cfm_holdout --cfm-split
python main.py --config configs/experiments/phmfactory/simulated_cfm_holdout.yaml
```

Use the returned checkpoint, normalization and protocol paths with the same public
sample/eval commands described above, selecting this configuration for every stage.
It requests nine class/domain conditions and evaluates all 72 test windows. Evaluation
may set `--override data.batch_size=1`; this changes evaluation batching, not its
population or fitted model. Paths in `data.split.manifest_path` and `data.data_dir`
must both be overridden when relocating the prepared input.

`--cfm-split` writes the format already consumed by `splitting.py`, with its existing
metadata binding. The public Schema now admits `manifest_mode=read_only` and
`partition_map`; it does not construct another split. CFM model cardinalities and
metadata-derived sampling conditions use training rows, while validation labels remain
available to validation loss. An explicit generation grid is a declared query, not a
request for the evaluator's labels. `eval_split=test` requires grouped metadata;
renaming the overlapping legacy test loader cannot establish a held-out population.

The seed-17, two-epoch configuration was executed with two distinct final-test
interventions: cyclically permute test labels, or replace only test waveforms with
seeded noise. Train and validation data and group membership were held fixed. Selected
parameters/buffers, Adam and scheduler state, selection scores, training normalization,
Python/NumPy/Torch RNG states after training and sampling, and generated tensors agreed
exactly. Each evaluation included 72/72 windows against nine samples; NumPy recomputation
of time-statistics and spectral distance agreed at `atol=1e-6, rtol=1e-5`.

This is configuration-specific simulation execution and isolation evidence, not
physical fidelity, real-equipment transfer or downstream augmentation benefit.
The centroid metric remains a proxy. There is no learned filter or downstream
augmentation classifier in this configuration. Registry promotion remains separate.
