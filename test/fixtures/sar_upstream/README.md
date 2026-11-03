# Fixed SAR oracle

Source-equivalent files from `mr-eggplant/SAR` at
`20f6e24b17525f34503510afccedc0629b67b7c4` (BSD-3-Clause):

- `sar.py`
- `sam.py`
- `LICENSE`

They are test-only numerical oracles, not runtime dependencies. Repository-required trailing whitespace is normalized without changing executable content. B03 preserves the
upstream BN2d/LN/GN update and adds a separately tested BatchNorm1d operator extension.
The production implementation also makes the empty reliable-set edge explicit by
skipping the update; the upstream code takes a mean over an empty tensor and is not used
as an oracle for that edge case.
