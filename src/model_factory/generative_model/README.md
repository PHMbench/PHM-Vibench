# Generative Model Factory

Generative models live under `src/model_factory/generative_model/`. This README
is the canonical model-factory guide for PHM generative work.

Models are imported by:

```yaml
model:
  type: generative_model
  name: phm_cfm_mlp1d
```

V0 models predict velocity for Conditional Flow Matching and use `[N, C, L]` signal tensors.
Model conditions are `fault_label` and `domain_id` only. Operating fields such as `load` and
`rpm` stay in the domain map for audit, evaluation, and reporting.

Factory-selectable generative backbones:

- `phm_cfm_mlp1d`: compact Conv1D velocity/epsilon/score model.
- `phm_unet1d`: conditional UNet1D backbone.
- `phm_dit1d`: tiny DiT-style 1D transformer backbone.
- `mamba1d_backbone`: stateless SSM/Mamba-style placeholder. It has no
  mandatory compiled dependency; `use_true_mamba=true` requires optional
  `mamba_ssm` and fails explicitly when unavailable.

## Condition Injection Contract

The default conditioning path should use time, fault, and domain embeddings with
FiLM or AdaLN-style feature modulation. Do not use raw
`torch.cat([x, condition], dim=1)` as the default condition mechanism.

```text
t_emb = sinusoidal_time_embedding(t)
fault_emb = Embedding(fault_label)
domain_emb = Embedding(domain_id)
c_emb = fuse(t_emb, fault_emb, domain_emb)
h' = gamma(c_emb) * Norm(h) + beta(c_emb)
```

Condition keys stay narrow in V0:

- `fault_label`
- `domain_id`

`load` and `rpm` remain domain-map metadata, not direct model conditions.

## Frontier Family Map

Frontier methods must enter through existing factories and remain exploratory
until evidence gates promote them.

Maturity labels:

- `docs-only`: design/reference material only
- `research-only`: research note or skeleton, not benchmark baseline
- `smoke-runtime`: minimal runtime path for local smoke validation
- `exploratory-runtime`: runnable but not benchmark-valid by default
- `benchmark-candidate`: eligible for evidence-gated paper comparison
- `benchmark-valid`: complete manifest/protocol/leakage/metric evidence exists

| Family | Status | Model role | Loss target | Runtime note |
| --- | --- | --- | --- | --- |
| Conditional Flow Matching | smoke-runtime | velocity model | `x1 - z` | V0 baseline |
| Rectified Flow | exploratory-runtime | velocity model | `x1 - z` | reuses stateless Euler ODE sampling |
| DDPM | exploratory-runtime | epsilon model | `epsilon` | requires scheduler and reverse sampler evidence |
| Score SDE | research-only | score model | score | needs reviewed predictor-corrector or ODE sampler |
| Mamba/SSM | research-only | sequence backbone | not a loss | stateless per denoising call |
| MeanFlow | research-only | one-step velocity family | method-specific average velocity | demo-only until separate promotion |
| Drifting Models | research-only | drift-field family | method-specific drift target | demo-only until separate promotion |

Runtime and promotion gates:

| Family | Prediction type | Compatible sampler | Future runtime location | Required tests before promotion | Maturity gate |
| --- | --- | --- | --- | --- | --- |
| Flow Matching | velocity | Euler ODE | `src/model_factory/generative_model/`, `src/task_factory/Components/generative/losses/` | loss shape, sampler finite guard, CPU smoke config | benchmark-candidate after manifest/leakage evidence |
| Rectified Flow | velocity | Euler ODE | same factory/component paths | loss target, sampler reuse, CPU smoke config | exploratory-runtime until evidence promotion |
| DDPM | epsilon | DDPM reverse sampler | same factory/component paths plus scheduler | epsilon target, scheduler finite guard, sampler smoke config | exploratory-runtime until evidence promotion |
| Score SDE | score | reviewed SDE/ODE sampler | same factory/component paths plus sampler | score target, drift/diffusion finite guard, sampler tests | research-only until sampler/protocol review |
| Mamba/SSM | backbone output head | delegated to flow/diffusion sampler | `src/model_factory/generative_model/` | stateless call test, optional dependency guard | research-only until paired method passes |
| MeanFlow | average velocity | one-step sampler | research-only until separate goal | identity/loss review, JVP/numerical tests | research-only |
| Drifting Models | drift field | one-step sampler | research-only until separate goal | drift target, neighbor/leakage tests | research-only |

Reference metadata defaults for every frontier family:

```yaml
reference_only: true
copy_code_allowed: false
implementation_language: unknown_if_not_verified
license_status: unknown_if_not_verified
paper_reference: required_before_promotion
code_reference: code_uncertain_unless_verified
```

Reference map:

| Family | Paper reference | Code reference | Language | License status |
| --- | --- | --- | --- | --- |
| Flow Matching | Lipman et al., "Flow Matching for Generative Modeling", arXiv:2210.02747 | `code_uncertain` unless verified for the selected implementation | unknown unless verified | unknown unless verified |
| Rectified Flow | Liu, Gong, and Liu, "Flow Straight and Fast", arXiv:2209.03003 | `https://github.com/gnobitab/RectifiedFlow` is marked official by its README; still reference-only here | Python | verify license before copying |
| DDPM | Ho, Jain, and Abbeel, "Denoising Diffusion Probabilistic Models", arXiv:2006.11239 | `https://github.com/hojonathanho/diffusion`; reference-only here | Python | verify license before copying |
| Score SDE | Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations", arXiv:2011.13456 | `code_uncertain` unless a selected implementation is license-reviewed | unknown unless verified | unknown unless verified |
| Mamba/SSM | Gu and Dao, "Mamba", arXiv:2312.00752 | `https://github.com/state-spaces/mamba`; optional dependency only | Python/CUDA | verify license before copying |
| MeanFlow | Geng et al., "Mean Flows for One-step Generative Modeling", arXiv:2505.13447 | `code_uncertain`; research-only | unknown unless verified | unknown unless verified |
| Drifting Models | Deng et al., "Generative Modeling via Drifting", arXiv:2602.04770 | project page lists code, but PHM-Vibench treats it as reference-only until license review | unknown unless verified | unknown unless verified |

All entries are `reference_only: true` and `copy_code_allowed: false` for this
repository unless a later goal performs explicit license and integration review.

Mamba/SSM is a backbone, not a generative loss. During flow or diffusion
sampling, each call at probability-flow time `t` must be stateless. Do not carry
hidden cache across ODE/SDE denoising steps. Sampler time is generative
probability-flow time, not physical sequence time.

## Experimental One-Step Methods

Exploratory one-step task targets include:

- `meanflow`
- `drifting_flow`
- `transition_flow_matching`
- `ot_nfm`

They are not benchmark-valid methods by default. They require:

- `task.generative.experimental: true`
- `task.generative.validity_status: exploratory`
- `task.generative.num_steps: 1`

Promotion requires method-specific evidence before any of these tasks can emit
benchmark-valid synthetic data.

## External Code Policy

External repositories are reference-only. Do not copy or vendor external code
unless a later goal explicitly performs license review. If official code cannot
be verified, mark it `code_uncertain` rather than inventing an official source.
The default for any reference is `reference_only: true` and
`copy_code_allowed: false`.
