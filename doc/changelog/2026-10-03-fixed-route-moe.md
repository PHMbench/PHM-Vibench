# Fixed-route MoE controls

The model factory now constructs `MoE/M_05_FixedRouteMoE` for explicitly prepared
physical-view packs. It owns the compact six-arm expert model and single-expert
clean-reference replacement, keeping the clean routing mass fixed. The paper trainer
reuses this implementation instead of defining a second model and intervention.

Inputs are views `[N,K,D]`, raw features `[N,D]`, and bounded compatibility `[N,K]`.
The model fragment is `configs/experiments/p04/fixed_route_model.yaml`; it is not a
complete general Pipeline training configuration. Split/probe validation and source-only
selection remain with the calling experiment. Historical M_04/G050 is unchanged.

Validation: `python -m pytest -q test/test_fixed_route_moe.py` exercises actual factory
construction, all six arms, explicit single-slot replacement parity, gradient finiteness,
and invalid axes/routes. These artificial fixtures do not establish physical validity,
baseline competitiveness, or a real-data result.

Explicit-width factory construction also defers the class-inference utility import,
so pure model tests do not initialize Lightning. Metadata-based class inference keeps
its existing path. A fresh-process regression checks this import boundary.
