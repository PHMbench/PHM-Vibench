# Bug Report

## Bug Summary
Foundation model training scripts fail with memory issues and extremely slow performance when executing multi-task experiments from script/Vibench_paper/foundation_model/ directory.

## Bug Details

### Expected Behavior
- All 4 foundation models (DLinear, FNO, PatchTST, TimesNet) should complete training within reasonable time (<12 hours each)
- All datasets in metadata_6_11.xlsx should be processed successfully  
- Training should complete without memory errors
- Results should be saved properly for all models

### Actual Behavior  
1. **DLinear**: Crashes with Out of Memory (OOM) error after processing only datasets 1, 2, 19
2. **FNO, PatchTST, TimesNet**: Training runs for >24 hours without completion
3. Multiple .err files generated in logs/9_10/ directory indicating system issues
4. Incomplete experimental results

### Steps to Reproduce
1. Navigate to PHM-Vibench project directory
2. Execute foundation model training scripts:
   ```bash
   sbatch script/Vibench_paper/foundation_model/run_dlinear.sbatch
   sbatch script/Vibench_paper/foundation_model/run_fno.sbatch  
   sbatch script/Vibench_paper/foundation_model/run_patchtst.sbatch
   sbatch script/Vibench_paper/foundation_model/run_timesnet.sbatch
   ```
3. Monitor job execution with `squeue`
4. Observe the issues described above

### Environment
- **Version**: PHM-Vibench current development version
- **Platform**: SLURM cluster with GPU nodes (V100, A100)
- **Configuration**: Multi-task foundation model training
- **Hardware**: V100 GPUs with 48GB RAM allocation
- **Data**: metadata_6_11.xlsx with multiple industrial datasets

## Impact Assessment

### Severity
- [x] High - Major functionality broken

### Affected Users
- Researchers running foundation model experiments
- Users attempting multi-task learning on industrial datasets
- Anyone using the foundation model paper reproduction scripts

### Affected Features
- Multi-task foundation model training pipeline
- Dataset processing for large-scale experiments
- Resource utilization and memory management
- Experimental reproducibility for paper results

## Additional Context

### Error Messages
```
# DLinear Error (from dlinear_multitask_dlinear_44217667.err):
slurmstepd: error: Detected 1 oom_kill event in StepId=44217667.batch. Some of the step tasks have been OOM Killed.

# Training Progress (DLinear reached Epoch 176 before crash):
Epoch 175: 100%|██████████| 66/66 [03:48<00:00,  0.29it/s, v_num=ftt7, classification_val_acc=0.979, ...]
Epoch 176:  18% [interrupted by OOM]
```

### Screenshots/Media
- Log files located in `/logs/9_10/` directory
- Multiple .err and .out files showing system resource issues
- Large log files (>50MB each) indicating extensive but incomplete processing

### Related Issues
- Memory configuration in SLURM batch scripts may be insufficient
- Batch sizes in YAML configurations may be too large
- Multi-task learning overhead not properly accounted for
- Dataset loading strategy may be inefficient for large-scale experiments

## Initial Analysis

### Suspected Root Cause
1. **Memory Misconfiguration**: 
   - Current memory allocation (48GB) insufficient for multi-task training
   - Batch sizes (384-512) too large for available GPU memory
   - Multiple concurrent tasks (classification, anomaly_detection, signal_prediction, rul_prediction) causing memory overflow

2. **Inefficient Resource Utilization**:
   - High num_workers (16-32) causing CPU/memory contention
   - Large model dimensions not optimized for available hardware
   - Lack of gradient accumulation strategy for large effective batch sizes

### Affected Components
- `/script/Vibench_paper/foundation_model/multitask_B_04_Dlinear.yaml`
- `/script/Vibench_paper/foundation_model/multitask_B_08_PatchTST.yaml`
- `/script/Vibench_paper/foundation_model/multitask_B_06_TimesNet.yaml`
- `/script/Vibench_paper/foundation_model/multitask_B_09_FNO.yaml`
- `/script/Vibench_paper/foundation_model/run_*.sbatch` (all SLURM scripts)
- Multi-task training pipeline in `/src/task_factory/task/In_distribution/multi_task_phm.py`