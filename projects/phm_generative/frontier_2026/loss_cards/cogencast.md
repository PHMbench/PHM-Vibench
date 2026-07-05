# Loss Card: CoGenCast

## Objective

$$p(x_{future}|x_{context})=\int p_{FM}(x_{future}|h)\,p_{AR}(h|x_{context})\,dh$$

## Contract

```text
inputs: method-specific tensors + condition
output: dict with scalar `loss` and named diagnostics
finite check: required
shape mismatch: fail fast
```

## Claim boundary

This loss card is a design contract. It is not evidence of a faithful reproduction until implementation and tests exist.
