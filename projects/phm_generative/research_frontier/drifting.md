# Drifting Models

Status: research-only.

Drifting Models are tracked as a frontier reference for drift-field training.
The approach depends on drift targets, stop-gradient choices, kernels, and
nearest-neighbor behavior that need separate review for PHM signals.

PHM risks include leakage through near-duplicate windows, kernel sensitivity
across domains, and overfitting to repeated operating conditions. No runtime
code, benchmark-valid outputs, or mandatory external dependencies are added.

Promotion requires a separate research review plus mature V0/V1 protocol,
manifest, leakage, and metric checks.
