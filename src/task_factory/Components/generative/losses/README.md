# Generative Losses

Losses for PHM generative tasks live in this module. Loss definitions belong
under TaskFactory Components, not ModelFactory.

## Future Runtime Paths

Future loss implementations belong here:

```text
src/task_factory/Components/generative/losses/flow_matching.py
src/task_factory/Components/generative/losses/rectified_flow.py
src/task_factory/Components/generative/losses/ddpm.py
src/task_factory/Components/generative/losses/score_sde.py
```

## Conditional Flow Matching

V0 baseline: Conditional Flow Matching for raw PHM signals `[N, C, L]`.

```math
z \sim \mathcal{N}(0,I), \qquad t \sim \mathcal{U}(0,1)
```

```math
x_t = (1-t)z + tx_1
```

```math
u_t = x_1 - z
```

```math
\mathcal{L}_{CFM}
=
\mathbb{E}
\left[
\left\|
v_\theta(x_t,t,c) - (x_1-z)
\right\|_2^2
\right]
```

Shape contract:

```text
x1/z/xt/pred_velocity: [N, C, L]
t: [N] or [N, 1, 1]
fault_label: [N]
domain_id: [N]
loss: scalar
```

## Future Loss Families

| Family | Prediction | Target | Status |
| --- | --- | --- | --- |
| CFM | velocity | `x1 - z` | V0 baseline |
| Rectified Flow | velocity | `x1 - z` | exploratory |
| DDPM | epsilon | `epsilon` | exploratory |
| Score SDE | score | score | research-only |
| Mamba/SSM | backbone | not a loss | backbone-only |
| MeanFlow | average velocity | research target | research-only/demo-only |
| Drifting Models | drift field | research target | research-only/demo-only |

FFT, STFT, Hilbert envelope, envelope peak, and band-energy metrics are
eval-only in V0. Do not add FFT loss to generative training in V0.
