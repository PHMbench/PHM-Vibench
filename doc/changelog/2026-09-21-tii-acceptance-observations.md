# TII one-model acceptance observations

The canonical `scripts/tii_one_model.py` gains an explicit `--acceptance` mode: one shared S model, two to five admitted sources, seed0, exactly20 native optimizer updates. Training-only mode is unchanged. No runtime, data loader or model implementation is duplicated in a paper repository.

The native Task observes per-source gradients from existing loss graphs and records actual joint-step parameter deltas. Per-source VJPs add observation cost but no forward/update. Joint deltas are not attributed to a single source.

Full source-validation windows are declared before fit, exported through the selected Task, checked on strict checkpoint reload, and compared with a frozen increment-masking diagnostic. The existing evaluator owns population checks and group NLL; the plotting module reads CSV only. Source validation is not an independent test or target transfer effect.

Recovery of complete predictions is offline: no H5, model or GPU access. Missing predictions require fixed-checkpoint export, never fit. Reuse cannot succeed merely because run.json or a plot README says complete. Failed output/logs remain available. Native checkpoint-score agreement is required for final acceptance status.

Focused tests cover this distinction and the real native CPU lifecycle on explicitly generated temporary tensor fixtures. Such regression inputs do not qualify any industrial source. The retained industrial qualification and historical evidence remain unchanged; no local industrial YAML, trained industrial weight, or new paper score is supplied here.
