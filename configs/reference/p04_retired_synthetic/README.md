# Retired P04 synthetic protocol

These byte-preserved `FULL`, `HOMO`, and `RAND` configurations belong to the
superseded `E-MINDEC` synthetic protocol. They remain available for historical
reproduction, but they are not active experiment configurations and must not be
used to support the current P04 claims.

The only active P04 G050 entry point is
`scripts/p04/run_g050_decisive.py`, which always loads
`configs/experiments/p04/g050_decisive.yaml`.

The historical preparation contract requires retired local synthetic artifacts.
Its tests are preserved at `test/historical/p04/run_preparer_contract.py` and
must be invoked explicitly only when those artifacts are available.
