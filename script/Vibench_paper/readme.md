
🚀 Grace HPC实验运行指南

  第一步: 登录和环境设置

  ssh ql334@grace.hpc.yale.edu
  cd /vast/palmer/home.grace/ql334/LQ/PHM-Vibench/

  # 加载环境
  module load miniconda
  conda activate P

  # 创建必要目录
  mkdir -p logs results

  第二步: 快速测试

  # 运行1个epoch快速验证
  sbatch script/Vibench_paper/foundation_model/run_test.sbatch

  # 监控状态
  squeue -u $USER
  tail -f logs/test_*.out

  第三步: 提交4个完整实验

  # 并行提交所有4个模型训练
  sbatch script/Vibench_paper/foundation_model/run_dlinear.sbatch    # V100, 24h
  sbatch script/Vibench_paper/foundation_model/run_fno.sbatch        # A5000, 24h  
  sbatch script/Vibench_paper/foundation_model/run_timesnet.sbatch   # A100, 36h
  sbatch script/Vibench_paper/foundation_model/run_patchtst.sbatch   # A100, 36h

  # 查看作业队列
  squeue -u $USER

  第四步: 实时监控

  # 监控各模型训练进度
  tail -f logs/dlinear_*.out
  tail -f logs/fno_*.out
  tail -f logs/timesnet_*.out
  tail -f logs/patchtst_*.out

  第五步: 查看结果

  # 训练完成后检查结果
  ls -la results/multitask_B_04_Dlinear/
  ls -la results/multitask_B_06_TimesNet/
  ls -la results/multitask_B_08_PatchTST/
  ls -la results/multitask_B_09_FNO/

  资源配置摘要

  | 模型       | GPU   | 内存  | 时间  | 预计完成 |
  |----------|-------|-----|-----|------|
  | DLinear  | V100  | 48G | 24h | 最快   |
  | FNO      | A5000 | 48G | 24h | 中等   |
  | TimesNet | A100  | 64G | 36h | 较慢   |
  | PatchTST | A100  | 64G | 36h | 较慢   |

  故障排查命令:
  scancel <JOB_ID>           # 取消作业
  cat logs/*_<JOB_ID>.err    # 查看错误日志
  scontrol show job <JOB_ID>  # 查看作业详情

  所有配置已针对Grace HPC优化，可以直接按步骤运行4个多任务基础模型实验！