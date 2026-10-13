# Finite TSPN gradients for constant features — 2026-10-13

A native CPU training smoke exposed nonfinite gradients in TSPN `CustomBatchNorm`: computing population standard deviation as `var.sqrt()` differentiates the square root at zero for constant coordinates.

Training now computes the same standard deviation as the centered vector norm divided by the square root of batch size. Its zero-norm derivative is finite. The positive-variance formula, epsilon placement, running mean/variance update and evaluation formula are preserved.

Five CPU regression cases passed for float32/float64 forward values, constant-coordinate backward behavior, running updates and evaluation. Native synthetic checkpoint/export tests also passed after this correction. This fix may change training trajectories that encountered constant features; it does not establish reproduction of historical results or industrial baseline quality.
