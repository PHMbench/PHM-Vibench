# Loss Card: Euler Mean Flows

## Objective

$$\mathcal L_{EMF}=\mathbb E\|u_\theta(z_t,r,t,c)-\mathrm{sg}(u_{target})\|_2^2$$

## Contract

```text
inputs: method-specific tensors + condition
output: dict with scalar `loss` and named diagnostics
finite check: required
shape mismatch: fail fast
```

## Claim boundary

This loss card is a design contract. It is not evidence of a faithful reproduction until implementation and tests exist.
