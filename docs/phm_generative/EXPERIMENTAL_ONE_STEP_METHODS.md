# Experimental One-Step Generative Methods

The following PHM generative tasks are integrated as exploratory, one-step
factory targets:

- `meanflow`
- `drifting_flow`
- `transition_flow_matching`
- `ot_nfm`

They currently reuse the stateless velocity-field contract and one-step Euler
sampling path. They are not benchmark-valid methods. The schema requires:

- `task.generative.experimental: true`
- `task.generative.validity_status: exploratory`
- `task.generative.num_steps: 1`

A later promotion goal must supply method-specific evidence before any of these
tasks can emit benchmark-valid synthetic data.
