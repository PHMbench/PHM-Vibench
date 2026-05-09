# MeanFlow

Status: research-only.

MeanFlow is tracked as a one-step generation research direction. The core idea
is to learn an average velocity field that can map noise to data with fewer
integration steps than standard flow models.

Implementation risks include JVP cost, numerical stability, and unclear PHM
conditioning behavior for fault/domain shifts. PHM adaptation questions include
whether average velocity preserves transient fault signatures and whether
single-step generation hides leakage or duplication artifacts.

No MeanFlow runtime code, benchmark-valid tables, or mandatory external
dependencies are added. Promotion requires stable V0/V1 protocol, manifest,
leakage, and metrics evidence.
