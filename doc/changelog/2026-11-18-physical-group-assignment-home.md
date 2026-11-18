# Physical-group assignment moved to the shared splitting module

`assign_physical_groups` now lives in `src/data_factory/splitting.py` next to
`resolve_data_splits`, making the sorted-identifier-then-PCG64 convention the
single home for deterministic group isolation. The previous definition in
`tii_sampling` is replaced by a re-export, so every existing import keeps
working and there is exactly one implementation. Behavior is unchanged; the
moved docstring records how explicit-count role inventories complement the
fraction-based grouped-metadata strategy.

`test/test_physical_group_assignment.py` pins the contract: identical seed
reproduces identical roles, record order does not change any group's role,
the RNG consumption order is exactly sorted-unique-then-PCG64, and inexact or
duplicate role coverage fails closed.
