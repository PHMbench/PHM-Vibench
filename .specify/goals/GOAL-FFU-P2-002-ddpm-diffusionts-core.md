# GOAL-FFU-P2-002: DDPM / Diffusion-TS Core

## Objective

Promote DDPM / Diffusion-TS-style generation as a benchmark baseline.

## Required Behavior

- Add epsilon or reconstruction objective configuration.
- Add scheduler/sampler config.
- Support transformer or UNet1D backbone where available.
- Preserve manifest and metric contracts.

## Acceptance Criteria

- CPU smoke config passes preflight.
- One training step and sample path work on dummy data.
- NFE and sampler metadata are recorded.

## Validation Commands

```bash
python -m pytest test/generative/test_ddpm_diffusionts.py -q
```
