# P01 matched nonlinear readout

`TSPN_fusion` accepts `head_type: linear` (unchanged default) or `head_type: mlp` with one explicit `head_hidden_dim` (default 16). For the nonlinear frozen-feature control use `branches: []` and `use_reference_features: true`. This adds a single ReLU hidden layer; the reference and input normalization remain unchanged. No operator, task, reader or general factory is replaced.

The linear head retains its parameter keys and Frobenius cap M. The MLP caps each layer at sqrt(M), so the product of operator norms remains at most M. Old linear checkpoints load strictly with their original configuration. An MLP checkpoint must carry the selected head type and width; mismatched reconstruction fails strictly.

The control is matched on frozen input features and the source training/selection budget, not necessarily on parameter count. Its existence does not establish equal optimization difficulty or diagnostic superiority. The P01 project owns paired comparisons, real-data preflight and tests through the installed Model Factory. This is a research-branch addition, not promotion of P01 into stable main/dev.
