# Loss Card: Path-dependent Physical Field One-Step FM

## Objective

$$\mathcal L=\mathcal L_{VAE}+\lambda_{FM}\mathcal L_{FM}^{latent}+\lambda_{aux}\mathcal L_{aux}$$

## Contract

```text
inputs: method-specific tensors + condition
output: dict with scalar `loss` and named diagnostics
finite check: required
shape mismatch: fail fast
```

## Claim boundary

This loss card is a design contract. It is not evidence of a faithful reproduction until implementation and tests exist.
