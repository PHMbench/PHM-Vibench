# Explicit normalization for grouped residual readouts

`X_model/TSPN_fusion` accepts optional `model.residual_group_count` for
`head_type: operator_residual`. Each group contribution is divided by the square
root of this count. The positive integer must cover every active feature group,
including reference features when enabled. A smaller configured branch family can
therefore retain the normalization of its declared full family.

Omitting the value preserves active-group normalization, initialization and state
keys. Linear and joint-MLP heads reject an explicit residual count. Restore a model
with its saved configuration, including this value; no checkpoint key or parameter
shape changes. The returned branch contributions use the same normalization as the
predicted logits.

The existing `test/test_tspn_fusion_v2.py` contains focused checks for default
equivalence, subset contributions, class log-odds reconstruction and strict state
restoration with saved configuration. These checks establish software behavior;
they do not establish industrial performance.
