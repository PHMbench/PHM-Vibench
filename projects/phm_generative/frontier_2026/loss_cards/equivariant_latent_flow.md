# Loss Card: Equivariant Latent Flow

## Objective

$$\mathcal L=\mathcal L_{recon}+\lambda_{eq}\mathcal L_{equiv}+\lambda_{FM}\mathcal L_{FM}^{latent}$$

## Contract

```text
inputs: method-specific tensors + condition
output: dict with scalar `loss` and named diagnostics
finite check: required
shape mismatch: fail fast
```

## Claim boundary

This loss card is a design contract. It is not evidence of a faithful reproduction until implementation and tests exist.
