# Loss Card: OT-NFM

## Objective

$$\mathcal L_{OT}=\mathbb E_{(z,x)\sim\gamma_{OT}}\|F_\theta(z,c)-x\|_2^2$$

## Contract

```text
inputs: method-specific tensors + condition
output: dict with scalar `loss` and named diagnostics
finite check: required
shape mismatch: fail fast
```

## Claim boundary

This loss card is a design contract. It is not evidence of a faithful reproduction until implementation and tests exist.
