# Loss Card: DiM-TS

## Objective

$$\mathcal L_{\epsilon}=\mathbb E\|\epsilon-\epsilon_\theta(x_t,t,c)\|_2^2$$

## Contract

```text
inputs: method-specific tensors + condition
output: dict with scalar `loss` and named diagnostics
finite check: required
shape mismatch: fail fast
```

## Claim boundary

This loss card is a design contract. It is not evidence of a faithful reproduction until implementation and tests exist.
