# Paper Submodule Guide

## Overview
The `paper/2025-10_foundation_model_0_metric` directory is a Git submodule that points to the private repository:
https://github.com/liq22/PHM-Vibench-Paper-2025-Metric

## Structure
- **Main Repository**: PHM-Vibench (codebase)
- **Submodule**: PHM-Vibench-Paper-2025-Metric (paper materials)

## Common Operations

### Clone with Submodule
```bash
git clone --recurse-submodules git@github.com:liq22/Vbench.git
```

### If Already Cloned Without Submodule
```bash
git submodule update --init --recursive
```

### Update Submodule to Latest
```bash
cd paper/2025-10_foundation_model_0_metric
git pull origin main
cd ../..
git add paper/2025-10_foundation_model_0_metric
git commit -m "Update paper submodule"
```

### Make Changes to Paper
```bash
cd paper/2025-10_foundation_model_0_metric
# Make your changes
git add .
git commit -m "Describe your changes"
git push origin main
```

### Check Submodule Status
```bash
git submodule status
```

## Benefits
- Clean separation between code and paper materials
- Independent versioning for paper
- Reduced main repository size
- Flexible access permissions

## Notes
- The paper repository is private and only accessible to you
- The main repository tracks a specific commit of the paper submodule
- Always commit submodule changes in the main repository after updating