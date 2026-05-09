# Rectified Flow

Status: exploratory runtime baseline.

Rectified Flow uses a straight interpolation from noise `z` to signal `x1` and
predicts the velocity target `x1 - z`. It keeps the V0 `[N, C, L]` signal
contract and the condition keys `fault_label` and `domain_id`.

Sampler reuse: the V0 Euler ODE sampler is reused because the model predicts a
stateless velocity field. Synthetic outputs remain exploratory unless the
protocol manifest and leakage checks mark them benchmark-valid.
