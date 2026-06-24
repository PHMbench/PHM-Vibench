# phm-data-factory

中文快速开始：[`docs/QUICKSTART_ZH.md`](docs/QUICKSTART_ZH.md)

Standalone, read-only-first data layer extracted from PHM-Vibench. It exposes
one API over Vibench `metadata.xlsx/CSV`, HDF5 signal caches, and Apache IoTDB,
plus bounded CLI/MCP tools for local agents.

## Boundary

`phm-data-factory` owns metadata indexing, signal random access, IoTDB import,
validation and Agent tools. PHM-Vibench keeps dataset splitting, PyTorch
Datasets/DataLoaders, task policy, models and trainers.

## Install

```bash
pip install -e .
pip install -e '.[excel,yaml,agent,iotdb]'
```

## Python

```python
from phm_data_factory import PHMDataRepository, AgentDataTools

repo = PHMDataRepository.from_local("data/metadata.xlsx", "data")
window = repo.get_signal_window(1, 0, 12000, channels=[0, 1], max_points=2048)
tools = AgentDataTools(repo)
print(tools.search_samples(task="fault_diagnosis", limit=10))
repo.close()
```

HDF5 keys may be `1`, `Id_1` or `sample_1`. Signals normally have `(L, C)`;
cache-only trailing singleton axes such as `(L, C, 1)` are removed by the
repository API.

## CLI

```bash
phm-data --metadata data/metadata.xlsx --signals data summary
phm-data --metadata data/metadata.xlsx --signals data window 1 \
  --start 0 --end 12000 --channels 0,1 --max-points 1024
```

All output is JSON. Agent-facing windows are bounded and report `step`; when
`step > 1`, the values are a preview rather than a model-training tensor.

## Local Agent / MCP

```bash
phm-data-mcp --config /absolute/path/phm-data.local.yaml
```

Client configuration:

```json
{
  "mcpServers": {
    "phm-data": {
      "command": "/absolute/path/to/venv/bin/phm-data-mcp",
      "args": ["--config", "/absolute/path/phm-data.local.yaml"]
    }
  }
}
```

Read-only tools: `repository_summary`, `list_datasets`, `search_samples`,
`get_sample_metadata`, `get_signal_window`, `get_signal_statistics`, and
`validate_sample`.

## IoTDB

```bash
cd docker/iotdb && docker compose up -d
phm-data-iotdb check
phm-data-iotdb import \
  --metadata /absolute/path/metadata.xlsx \
  --signals /absolute/path/data \
  --root root.vibench --report import-report.json
```

Tree layout:

```text
root.vibench.<dataset>.sample_<Id>.signal.ch_<channel>
root.vibench.<dataset>.sample_<Id>.meta.<field>
```

Timestamps are sample indices `0..L-1`; channels are aligned. Metadata is
mirrored at timestamp `0`, allowing an IoTDB-only Agent configuration.

## PHM-Vibench bridge

The `integration/phm_vibench/` directory contains the thin adapter and the
exact files intended for the PHM-Vibench repository. It adds only
`build_data_repository` and `build_agent_data_tools`; the existing
`build_data()` training entrypoint is unchanged.

## Test and build

```bash
pytest
python -m pip wheel --no-deps --no-build-isolation .
```
