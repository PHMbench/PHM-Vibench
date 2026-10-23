# Goal07 PU original-record observations

Added `python -m scripts.tii_pu_observations` for the fixed 2,559 PU originals
identified by native metadata and the existing record inventory. It checks the
unique `vibration_1` channel's XIndex, linked raster, vector lengths and stored
timestamps without reading H5, modifying inputs, training or qualifying units.
Sentinel Id47567 is first; remaining records follow metadata Id order.

The CLI requires `--data-root`, `--inventory` and `--output-dir`. It refuses to
overwrite its three outputs and retains per-record parse/mapping failures.
Stored-axis rate comparisons explicitly depend on treating X.Data as seconds;
exact timestamp nonuniformity does not establish a physical clock model.
Scientific qualification remains `UNRESOLVED`.

Local validation: `CUDA_VISIBLE_DEVICES='' conda run -n LQ_signal
--no-capture-output python -m pytest test/test_tii_pu_observations.py -q`:
**PASS, 18 tests**. These use synthetic MAT files and do not establish real-PU
coverage or acquisition eligibility. The real-data command and observations
belong to the separate Goal07 execution report.
