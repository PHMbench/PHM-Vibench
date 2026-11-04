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
training-reference smoke; `test` remains unsupported rather than falling back.

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
