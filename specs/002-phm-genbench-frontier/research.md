# Research: PHM-GenBench Frontier

## Decision: Use Flow Matching As The Primary Efficiency Axis

**Rationale**: The repo already contains CFM and Euler ODE sampling. FlowTS
shows time-series generation can use Rectified Flow with efficient straight-line
transport and conditional adaptation. TimeFlow adds stochastic-aware flow
matching for sequence randomness. These align with PHM vibration signals where
fault/domain conditioning and NFE reporting matter.

**Alternatives considered**: Diffusion-only roadmap; docs-only frontier tracking.
Both are insufficient for the user's core-fast integration target.

## Decision: Keep Diffusion-TS/DDPM As A Required Baseline

**Rationale**: Diffusion-TS is a strong general time-series generation reference,
and PHM-specific 2026 fault diagnosis papers continue to use conditional
diffusion variants. A PHM benchmark needs a diffusion baseline to compare
quality/utility/efficiency against faster flow methods.

**Alternatives considered**: Skip diffusion because flow is faster. Rejected
because paper reviewers expect diffusion baselines.

## Decision: Treat Mamba And DiT As Backbones, Not Protocols

**Rationale**: Mamba provides linear-time sequence modeling and DiT provides a
scalable transformer diffusion backbone. In this repo they should be model
backbones under `src/model_factory/generative_model/`, not new losses or
pipeline variants.

**Alternatives considered**: Separate Mamba pipeline. Rejected because it would
violate factory-first runtime boundaries.

## Decision: Integrate One-Step Frontier Families As Experimental Core

**Rationale**: The user selected Core Fast. MeanFlow/iMF, Drifting, Transition
Flow Matching, and OT-NFM are important frontier directions for one-step or
few-step generation. They should be integrated behind `experimental=true` and
default `validity_status=exploratory`.

**Alternatives considered**: Demo-only. Safer, but does not satisfy the selected
integration strategy.

## Decision: Paper Claims Depend On Evidence, Not Model Novelty

**Rationale**: A top PHM application paper must demonstrate reproducibility,
leakage resistance, physical/spectral fidelity, and downstream utility. The
pipeline must therefore prioritize manifest, normalization, condition coverage,
missing metric reasons, and paperpack aggregation before broad model claims.

**Alternatives considered**: Implement many methods first. Rejected because it
creates unverifiable results.

## Sources

- FlowTS: `https://arxiv.org/abs/2411.07506`
- TimeFlow: `https://arxiv.org/abs/2511.07968`
- Diffusion-TS: `https://arxiv.org/abs/2403.01742`
- Mean Flows: `https://arxiv.org/abs/2505.13447`
- Improved Mean Flows: `https://arxiv.org/abs/2512.02012`
- Generative Modeling via Drifting: `https://arxiv.org/abs/2602.04770`
- Transition Flow Matching: `https://arxiv.org/abs/2603.15689`
- ODE-free Neural Flow Matching: `https://arxiv.org/abs/2604.06413`
- Mamba: `https://arxiv.org/abs/2312.00752`
- DiT: `https://arxiv.org/abs/2212.09748`
- Diffusion Transformer-to-Mamba Distillation: `https://arxiv.org/abs/2506.18999`
- Physics-constrained Flow Matching: `https://arxiv.org/abs/2506.08604`
- ACS-DM PHM diffusion reference: `https://journals.sagepub.com/doi/10.1177/10775463251414180`
