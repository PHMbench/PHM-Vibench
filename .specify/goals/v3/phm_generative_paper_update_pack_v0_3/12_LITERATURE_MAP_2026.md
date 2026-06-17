# 12. Literature Map for PHM Generative Paper

## Classic foundations

### DDPM

Role: canonical diffusion baseline.  
Use in repo: `ddpm_epsilon` task and `DDPMEpsilonPredictionLoss`.

Key claim:
DDPM epsilon prediction is a stable, well-known baseline, but vibration
generation needs PHM-specific time/frequency evaluation.

### Score SDE

Role: theory and exploratory continuous-time score baseline.  
Use in repo: `score_sde` must stay exploratory until sampler and scaling are
validated.

### Flow Matching

Role: main continuous-time generative baseline.  
Use in repo: `conditional_flow_matching`.

Key claim:
Flow Matching trains vector fields by regression against conditional paths and
does not require simulating ODEs during training.

### Rectified Flow

Role: main straight-path velocity baseline.  
Use in repo: `rectified_flow`.

Key claim:
Rectified Flow learns ODE transport along straight paths by nonlinear least
squares and supports coarse/few-step sampling.

## Time-series and PHM-specific work

### Diffusion-TS

Role: time-series diffusion reference.  
Important for repo:
- transformer / temporal representation can be useful,
- Fourier loss is relevant for time-series generation,
- conditional generation can be extended.

### TSDM for vibration signal generation

Role: vibration-specific DDPM reference.  
Important for repo:
- raw vibration generation should not be judged like images,
- frequency preservation is essential,
- U-Net variants can improve DDPM for vibration signals,
- small-sample fault diagnosis augmentation is a key utility task.

### Diff-MTS

Role: industrial multivariate time-series diffusion reference.  
Important for repo:
- C-MAPSS / FEMTO-like industrial datasets matter,
- diversity/fidelity/utility should all be reported,
- conditional consistency matters.

### DiM-TS / Mamba time-series generation

Role: long-sequence backbone reference.  
Important for repo:
- Mamba/SSM claims require real selective-state-space implementation,
- channel correlation and temporal periodicity are key metrics.

## 2025+ frontier

### MeanFlow

Role: one-step flow frontier.  
Repo policy:
Keep as research-only until method-specific loss and tests exist.

### Drifting / Sinkhorn Drifting

Role: non-ODE one-step generative frontier.  
Repo policy:
Keep as research-only.  Current velocity-contract placeholder is not a full
drifting implementation.

## Repo implication

Paper structure should separate:

```text
Core baselines:
  CFM, Rectified Flow, DDPM

Backbone ablations:
  MLP1D, UNet1D, DiT1D, SSM-style adapter

Exploratory frontier appendix:
  Score SDE, MeanFlow, Drifting, Transition Flow, OT-NFM
```
