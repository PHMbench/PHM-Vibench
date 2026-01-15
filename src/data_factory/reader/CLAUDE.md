# Dataset Readers - CLAUDE.md

This module provides architecture guidance for the dataset reader system in PHM-Vibench. For implementation details, see [@README.md].

## Architecture Overview

The reader system implements a uniform interface for loading 30+ industrial vibration datasets:

```
Raw Data Files (CSV, Excel, MAT, TXT, etc.)
     ↓
┌─────────────────────────────────────┐
│  BaseReader (Abstract Base)           │
│  - Common interface                    │
│  - Standardization methods            │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│  Dataset Reader (RM_XXX_*)           │
│  - Dataset-specific loading            │
│  - Metadata extraction                │
│  - Preprocessing                      │
└─────────────────────────────────────┘
     ↓
Standardized Output (dict with keys: data, labels, metadata)
```

## Design Principles

### 1. Uniform Interface
All readers inherit from `BaseReader`:
```python
class RM_XXX_Dataset(BaseReader):
    def load_data(self):
        # Dataset-specific implementation
        return {
            'data': np.array,
            'labels': np.array,
            'metadata': dict
        }
```

### 2. Metadata-Driven
Each reader provides metadata in a standard format:
- `Dataset_id`: Unique dataset identifier
- `Name`: Human-readable name
- `File`: Source file reference
- `Label`: Fault class or RUL value
- `Domain_id`: Operating condition
- `Sample_rate`, `Sample_lenth`, `Channel`: Signal properties

### 3. Registration System
Readers are registered in `__init__.py`:
```python
from .reader.RM_001_CWRU import RM_001_CWRU
```

## Available Datasets

### Classic Bearing Datasets
- `RM_001_CWRU`: Case Western Reserve University
- `RM_002_XJTU`: Xi'an Jiaotong University
- `RM_003_FEMTO`: FEMTO bearing degradation

### University/Lab Datasets
- `RM_006_THU`: Tsinghua University
- `RM_017_Ottawa19`: Ottawa University (2019)
- `RM_018_Ottawa23`: Ottawa University (2023)
- `RM_026_HUST23`: Huazhong University (2023)
- `RM_027_HIT23`: Harbin Institute (2023)

### And More...
Total 30+ dataset readers covering:
- Bearings, gears, pumps, fans
- Various fault types and operating conditions
- Different sampling rates and signal lengths

## Adding New Readers

### 1. Create Reader Class
```python
# In src/data_factory/reader/RM_XXX_NewDataset.py
from .BaseReader import BaseReader

class RM_XXX_NewDataset(BaseReader):
    def __init__(self, data_dir):
        super().__init__(data_dir)

    def load_data(self):
        # Implement loading logic
        return standardized_dict
```

### 2. Register in __init__.py
```python
# In src/data_factory/__init__.py
from .reader.RM_XXX_NewDataset import RM_XXX_NewDataset
```

### 3. Add to Metadata
Add dataset info to `data/metadata.xlsx` with proper Dataset_id.

## Related Documentation

- [@README.md] - Reader Usage and Dataset List
- [@../README.md] - Data Factory Overview
- [@../../configs/README.md] - Configuration System
