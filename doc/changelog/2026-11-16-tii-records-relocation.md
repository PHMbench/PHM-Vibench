# TII J1 qualification records relocated to the paper repository

Experiment records are no longer stored in this repository. The 40 tracked
files under `reports/tii_local_j1/` (qualification CSVs, gate documents, the
native fixture and its run artifacts) moved verbatim to the paper repository
under `paper/experiments/records/tii_local_j1/`. `reports/` keeps only its
generic README and the unrelated UXFD run index.

`scripts/tii_one_model` resolves its qualification CSV from
`TII_QUALIFICATION_CSV` first, falling back to the removed default path. When
no qualification file is present, every no-config command still fails closed
with exit code 2 and a `qualification_records_relocated` blocker instead of a
traceback; CI asserts that contract. Config-driven runs keep reading
`data.qualification_file` and `data.record_inventory`, which should now point
at the paper-repository copies. `scripts/tii_qualify_data` requires an explicit
`--output-dir` so new qualification output never lands under `reports/`.

The portable M0 fixture config moved to `test/tii_m0_tensor_fixture.yaml`; it
is a test asset and no longer ships under `configs/`. The fixture README
example commands now write to scratch space outside this repository.
