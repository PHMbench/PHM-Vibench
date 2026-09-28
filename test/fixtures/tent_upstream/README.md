# Fixed Tent fidelity reference

- Official repository: https://github.com/DequanWang/tent
- Revision: `e9e926a668d85244c66a6d5c006efbd2b82e83e8`
- Source: `tent.py`, copied byte-for-byte, including the original TorchScript entropy.
- License: [MIT](LICENSE), copyright Dequan Wang and Evan Shelhamer (2021).
- Paper: *Tent: Fully Test-Time Adaptation by Entropy Minimization*, ICLR 2021,
  https://arxiv.org/abs/2006.10726.

This is a test oracle, not a production dependency or a replacement implementation.
The configured Adam uses lr supplied by the experiment, betas=(0.9,0.999), eps=1e-8,
weight_decay=0 and one update; the official example uses lr=1e-3. No external source
weights are redistributed. All numerical fixtures construct explicit synthetic weights.

`test/test_tta_tent.py` compares returned pre-update logits, entropy, affine gradients,
updated model parameters, Adam moments and torch RNG over three successive batches.
BN2d uses the official parameterization unchanged. The BN1d extension is compared to
BN2d with a singleton spatial axis and identical weights. Both fixture paths use the
same contiguous BN input/output layout before dropout; this is a controlled test input,
not a production model/input conversion. Non-contiguous channels-last BN2d and BN1d
must not be assumed to assign identical dropout masks from the same RNG seed.

CPU FP32 tolerances: logits rtol=1e-6/atol=1e-7; gradient/model/optimizer tensor
rtol=atol=1e-7; loss rtol=1e-6. Fixed-fixture label and resume comparisons are exact.
These tolerances are not cross-device bitwise reproducibility or industrial accuracy
claims. The tests also run outside the checkout against the installed PHMFactory wheel.
