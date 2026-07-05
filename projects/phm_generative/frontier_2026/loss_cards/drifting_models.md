# Loss Card: Drifting Models

## Objective

$$x^{+}=\mathrm{sg}(x+V(x;\mathcal D_{real},\mathcal D_{fake})),\quad \mathcal L=\|x-x^{+}\|_2^2$$

## Contract

```text
inputs: method-specific tensors + condition
output: dict with scalar `loss` and named diagnostics
finite check: required
shape mismatch: fail fast
```

## Claim boundary

This loss card is a design contract. It is not evidence of a faithful reproduction until implementation and tests exist.
