# phm-data-factory backend

**Status: Experimental.** This adapter is implemented and covered by a synthetic
contract test, but it is not part of the v0.2 release-supported demo matrix and
has no real-dataset performance claim.

PHM-Vibench can keep its existing `build_data()` pipeline while delegating
typed metadata and dense signal reads to the standalone provider.

```yaml
data:
  factory_name: phm_data
  phm_data_config: configs/data/cwru-iotdb.yaml
  dataset_name: CWRU
```

Initialize the exact provider revision and install it locally:

```bash
git submodule update --init packages/phm-data-factory
pip install -e 'packages/phm-data-factory[yaml,legacy]'
python -m scripts.validate_configs
python -m pytest -q test/test_phm_data_factory_backend.py
```

`phm_data_config` is mandatory; there is no silent HDF5 fallback. The provider
may select local HDF5 or IoTDB. PHM-Vibench continues to own dataset filtering,
splits, windowing, normalization, Dataset/DataLoader, task and trainer logic.

Older IoTDB imports with indexed/string-only metadata must be upgraded with
`phm-data-iotdb sync-metadata` before training.
