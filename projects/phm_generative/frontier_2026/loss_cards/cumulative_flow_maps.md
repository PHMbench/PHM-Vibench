# Loss Card: Cumulative Flow Maps

## Objective

$$\mathcal L_{CFMmap}=\mathbb E\|\Phi_\theta(x_s,s,t,c)-x_t\|_2^2$$

## Contract

```text
inputs: method-specific tensors + condition
output: dict with scalar `loss` and named diagnostics
finite check: required
shape mismatch: fail fast
```

## Claim boundary

This loss card is a design contract. It is not evidence of a faithful reproduction until implementation and tests exist.
