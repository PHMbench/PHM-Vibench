# Generative, adaptive and online learning: mechanism and integration map

Reviewed against dev `baee6bcf641fd7ef92e909f4fadf7e43571d7580`, 2026-10-14.
This is a bounded primary-source review, not an exhaustive SOTA ranking or a claim that
all 187 model candidates were audited. [The existing policy](MODEL_INTEGRATION_POLICY.md)
remains authoritative. Learning algorithms are not new backbone identities.

Metadata lives in one [BibTeX file](references.bib). Each row below links its original
paper and gives an original technical synopsis rather than reproducing the abstract.
Preprint entries retain their first-submission year; later conference versions are named
where verified. The RUL study has a 2027 issue assignment but was already online in July
2026. Full text/official code access limitations are stated rather than filled with guesses.

## Current-source findings and selected work

| Object | Actual implementation / protocol boundary | Decision in this batch |
|---|---|---|
| Native CFM | `src/model_factory/generative_model/phm_cfm_mlp1d.py`, `Components/generative/flow_matching.py`, Pipeline 06. The loss uses independent noise/data linear interpolation and velocity MSE; epsilon excludes the time endpoints. This is not verified OT-CFM, TSFlow or IMM. | EXISTING; retain exploratory `needs_smoke / smoke_only`; no current industrial or downstream utility promotion. |
| Frozen source / Tent | B01/B02 runtimes and pinned Tent oracle exist. No TTA Task/CLI binding existed at this base. | EXISTING; selected change is a small Config binding reusing these runtimes, not a new method. |
| TimesNet | Existing classification port and installed-wheel tests. | EXISTING; E3 absent. Do not automatically enable Tent on a BN-free backbone. |
| SAR | Official source was read on the earlier B03 line; branch contains source-transport preparation only, not an integrated algorithm. | REFERENCE_ONLY for this batch; do not continue the superseded transport or claim an implementation. |
| Other rows below | Source mechanisms are documented; exact variant, revision, dependency budget and code/weights licenses are not all established. | REFERENCE_ONLY, not queued and not rejected scientifically. |
| VAE/GAN/DDPM/normalizing flow families | Not admitted as executable generative implementations by this audit. Catalogue entries are not execution evidence. | No model-by-model code copying or new registry. |
| Supervised online / replay / SFDA | Label-event lifecycle, separate populations and complete resume are not implemented by this Config slice. | Fail closed, rather than alias to Tent. |

`REFERENCE_ONLY` is a reading-list decision, not a new lifecycle/disposition enum. An eventual
candidate audit must select the existing `RESEARCH_ONLY` or `EXTERNAL_ONLY` disposition and
pass G1-G7 before implementation. The original 187-row catalogue is unchanged. No synthetic
priority score or arbitrary method-count target is introduced.

## Primary literature and technical synopses

### Generative objectives and time-series synthesis

| Reference key / method | Mechanism, cost driver and scientific boundary |
|---|---|
| [`kingma2013vae`](https://arxiv.org/abs/1312.6114) — VAE | Reparameterized variational inference provides an ELBO-based latent-variable baseline. A PHM decoder, observation likelihood and train-only normalization remain design choices; a generic autoencoder is not automatically this method. |
| [`goodfellow2014gan`](https://arxiv.org/abs/1406.2661) — GAN | A discriminator and generator are trained adversarially. Compare generator quality and mode coverage, not discriminator loss alone; GAN training adds a second optimized network. |
| [`dinh2016realnvp`](https://arxiv.org/abs/1605.08803) — Real NVP | Invertible coupling transformations enable exact density evaluation. A density score is not a calibrated fault probability, and channel/time transformations must preserve the declared observation semantics. |
| [`ho2020ddpm`](https://arxiv.org/abs/2006.11239) — DDPM | Denoising training and a multistep reverse process form a distinct strong generative baseline. Freeze noise schedules and sampling evaluations; repeated denoiser calls belong in inference cost. |
| [`lipman2022fm`](https://arxiv.org/abs/2210.02747) — Flow Matching | Regress the vector field of prescribed conditional probability paths without simulating the training ODE. Sampling still requires a declared solver, time interval and evaluation count. |
| [`tong2023cfm`](https://arxiv.org/abs/2302.00482) — Generalized CFM / OT-CFM | Conditional flow matching permits non-Gaussian source distributions; minibatch optimal transport changes the coupling. Native independent noise/data pairing must not be renamed OT-CFM. |
| [`yoon2019timegan`](https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks) — TimeGAN | Combines an embedding representation with adversarial and temporal supervised objectives. Its multiple training components and sequence losses must be reproduced, not replaced by a plain GAN bearing the same name. |
| [`yuan2024diffusionts`](https://arxiv.org/abs/2403.01742) — Diffusion-TS | A decomposed temporal representation and reconstruction/Fourier objectives distinguish it from noise-prediction DDPM. Periodicity priors may be useful for vibration but require physical-frequency and downstream checks. |
| [`kollovieh2024tsflow`](https://arxiv.org/abs/2410.03024) — TSFlow | Gaussian-process priors and conditional prior sampling encode temporal structure for generation/forecasting. Forecasting requires history-only conditioning and a separate forecasting Task; it is not a classification toggle. |
| [`zhou2025imm`](https://proceedings.mlr.press/v267/zhou25c.html) — Inductive Moment Matching | Single-stage moment-matching self-distillation targets one/few-step generation. Image-domain quality results are not PHM utility; first establish ordinary CFM and sampling-cost controls. |

### TTA, CTTA and source-free adaptation

| Reference key / method | Mechanism, cost driver and scientific boundary |
|---|---|
| [`wang2021tent`](https://openreview.net/forum?id=uXl3bZLkr3c) — Tent | Entropy minimization updates normalization affine parameters on unlabeled target batches. Current-batch normalization is already an adaptation effect before the first parameter update; use a normalization-only control for attribution. |
| [`niu2023sar`](https://openreview.net/forum?id=g2YraF75Tj) — SAR | Reliable entropy selection and a sharpness-aware second evaluation differ from Tent. Empty selections, recovery thresholds and both SAM/base-optimizer state must be explicit; two gradient evaluations are not one Tent update. |
| [`niu2022eata`](https://proceedings.mlr.press/v162/niu22a.html) — EATA | Selects reliable nonredundant samples and regularizes important parameters using Fisher information. Freeze the Fisher-estimation population and source-artifact policy; do not flatten all variants into checkpoint-only adaptation. |
| [`wang2022cotta`](https://arxiv.org/abs/2203.13591) — CoTTA | Weight/augmentation-averaged teacher predictions and stochastic restoration address error accumulation. Teacher, augmentation, restoration and RNG states add costs and restoration responsibilities absent from Tent. |
| [`gong2022note`](https://arxiv.org/abs/2208.05117) — NOTE | Instance-aware normalization and prediction-balanced reservoir sampling address correlated streams. The adaptation buffer is distinct from the evaluation population; report bytes, retention and update frequency. |
| [`liang2020shot`](https://proceedings.mlr.press/v119/liang20a.html) — SHOT | Freezes the source classifier while adapting representations using information maximization and self-supervised pseudo-labeling. Offline adaptation and independent evaluation populations must remain separate; it is not the B02 online estimator. |
| [`ragab2024emapu`](https://arxiv.org/abs/2406.02635) — E-MAPU / temporal imputation | Temporal masking/imputation and evidential calibration address time-series SFDA. A source-trained imputer and evidential model are method dependencies; this later work must not be conflated with the original KDD 2023 MAPU paper. |
| [`murphy2025neo`](https://arxiv.org/abs/2510.05635) — NEO (2026 conference version) | Re-centers target embeddings without gradient optimization. This is a low-compute mechanism comparator, not automatically an x-only black-box adapter: the required feature/head access needs a separate contract. |
| [`lim2026asr`](https://arxiv.org/abs/2603.03796) — Adaptive and Selective Reset | Dynamically chooses reset timing and locations, with importance-aware regularization. Evaluate lost versus retained adaptation on returning conditions rather than claiming that periodic full resets guarantee safety. |

### Online learning, replay and drift

| Reference key / method | Mechanism, cost driver and scientific boundary |
|---|---|
| [`aljundi2019mir`](https://arxiv.org/abs/1908.04742) — MIR | Retrieves replay samples whose predictions are most damaged by a proposed update. Candidate-scoring/virtual-update computation must be charged; supervised retrieval is not label-free TTA. |
| [`buzzega2020der`](https://arxiv.org/abs/2004.07211) — DER / DER++ | Replays stored historical logits, with supervised replay in DER++. Count examples, labels and stored logits in memory; delayed labels need an event lifecycle rather than a TTA metadata loophole. |
| [`seo2024budget`](https://arxiv.org/abs/2410.15143) — Budgeted OCL (ICLR 2025) | Compares computation and total storage rather than epochs or sample-count-only buffers. Adaptive layer freezing and frequency-based retrieval are candidates only after total-resource accounting and label timing are executable. |
| [`bifet2007adwin`](https://epubs.siam.org/doi/10.1137/1.9781611972771.42) — ADWIN | Adaptive windowing detects changes in a chosen scalar stream. Monitoring labeled prediction errors needs released labels; unlabeled-feature drift is a different monitored quantity and cannot prove diagnostic degradation. |

### PHM-specific studies and applicability limits

| Reference key / method | Mechanism, cost driver and scientific boundary |
|---|---|
| [`yue2026dgldm`](https://www.sciencedirect.com/science/article/pii/S1474034626000972) — DG-LDM | Publisher metadata establishes an imbalanced-fault data-augmentation application. Full text and official code were not accessible in this batch; detailed losses, split validity and reproduction are deliberately unverified. |
| [`yu2027rultta`](https://doi.org/10.17531/ein/226057) — RUL-TTA comparative study | The publisher reports direction-dependent adaptation and a normalization bottleneck on C-MAPSS/N-CMAPSS. These are study-specific reported findings, not a general theorem or PHMFactory result. Online publication: 2026-07-29; assigned issue: 2027. |

### Existing backbone provenance

| Reference key / method | Mechanism, cost driver and scientific boundary |
|---|---|
| [`wu2022timesnet`](https://arxiv.org/abs/2210.02186) — TimesNet | Period-selected two-dimensional temporal variation supplies a distinct backbone bias. The integrated variant is fixed-window classification; other tasks and arbitrary batch-size invariance are not inherited from the paper title. |

## Official implementation and licensing audit boundary

| Method | Official source located | Revision / license conclusion |
|---|---|---|
| Tent | [DequanWang/tent](https://github.com/DequanWang/tent) | Existing fixture pins `e9e926a668d85244c66a6d5c006efbd2b82e83e8`, MIT. BN1d is a declared PHMFactory extension, not verbatim upstream. No external weights are redistributed. |
| TimesNet | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) | Existing fixture pins `4e938a1767106324dd753b2a44832bf870a0252e`, MIT; classification scope only. |
| SAR | [mr-eggplant/SAR](https://github.com/mr-eggplant/SAR) | Earlier read pins `20f6e24b17525f34503510afccedc0629b67b7c4`, BSD-3-Clause. Upstream ignores its supplied reset threshold in favor of a literal 0.2; empty selection and base-SGD state need their own tests. No production integration. |
| CFM reference family | [TorchCFM](https://github.com/atong01/conditional-flow-matching) | Paper-linked external oracle candidate; no exact revision/license/fidelity approval in this batch. Native CFM is not asserted identical. |
| TimeGAN | [jsyoon0823/TimeGAN](https://github.com/jsyoon0823/TimeGAN) | Official project found; dependency and license/revision admission remains open. |
| TSFlow | [marcelkollovieh/TSFlow](https://github.com/marcelkollovieh/TSFlow) | Official project found; GP prior, forecasting Task, license/revision and sampling budget need separate qualification. |
| EATA / NOTE / SHOT | [EATA](https://github.com/mr-eggplant/EATA), [NOTE](https://github.com/TaesikGong/NOTE), [SHOT](https://github.com/tim-learn/SHOT) | Official projects located. This batch does not freeze their licenses/revisions or source-artifact policy; no admission from a repository link alone. |
| MIR / DER | [MIR](https://github.com/optimass/Maximally_Interfered_Retrieval), [Mammoth](https://github.com/aimagelab/mammoth) | Official projects located. Borrow algorithm operators only after a labeled online protocol exists; do not import their parallel trainers. |
| ASR | [YonseiML/asr](https://github.com/YonseiML/asr) | Paper links the project. Selective recovery requires an explicit reset/state test; no code/weight license admission yet. |
| DG-LDM / RUL study | Publisher records above | No legal complete executable oracle was established here. Detailed reproduction remains blocked, not silently substituted. |

For unlisted code or weights licenses the value is **unverified**, not MIT/Apache by
assumption. No external source or pretrained weights were copied in this change.

## Evolution and the non-duplicate next choices

Generative comparisons should separate objective (ELBO, adversarial, denoising, velocity,
exact likelihood) from backbone and conditioning. A short ODE step count alone does not
make a method IMM or establish faster matched-quality generation. The current CFM needs
current-source end-to-end qualification and a train-only generation population before a
second expensive generative family. DDPM is then the clearest non-duplicate objective;
TimeGAN or Diffusion-TS require demonstrating additional temporal information beyond it.

For adaptation, first compare frozen source, a declared normalization-only control and
Tent on the same checkpoint, observations, order and batch partition. EATA changes source
artifacts and selection; SAR changes update geometry; CoTTA changes teacher/restoration;
NOTE changes buffering/normalization; NEO changes the feature interface. They are not
interchangeable model names. The RUL study's reported direction dependence reinforces
the need to measure harmful adaptation rather than assume image-benchmark gains transfer.

For online learning, replay needs explicit label release, causal evaluation and a
memory policy. MIR selection overhead and DER's logits count toward the total budget.
ADWIN on observed errors is supervised monitoring; ADWIN on a physical feature is not an
error oracle. Concept change, observable covariate shift and fault progression must not
be conflated.

## Minimal next empirical gate

Freeze a legal source checkpoint, source label ontology, sample-level grouping, target
row/time order and train-only preprocessing. Freeze tuning on source validation or a
separate development shift. Evaluate all methods on the same population and report
population Macro-F1/accuracy plus paired harm, condition-wise results and measured
prediction/update cost. A-to-B-to-A return tests and checkpoint/cursor/RNG/evaluator
continuation come before safety or non-forgetting claims. For generation, train the
generator and downstream learner without test access; compare real-only, real-plus-
generated and matched simple augmentation with the same grouping and tuning budget.

Current tests establish software/protocol behavior on constructed observations. They do
not establish a PHM effect, literature-score reproduction, full model-catalogue coverage
or publishable SOTA performance. See the [actual Config contract](../adaptation/protocol.md).
