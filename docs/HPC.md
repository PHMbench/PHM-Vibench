下面是一份可直接落地的《YCRC（Grace & Bouchet）GPU 项目最佳实践：从代码到提交》手册。结构涵盖：目录组织→环境→调试→提交→队列与优先级→数组作业（dSQ）→参数传递→依赖编排→监控优化→常见故障→“什么时候提交更容易上机”。
1) 项目目录基线（可直接套用）
your-project/
├─ src/                     # 源码（Python/R/CUDA等）
│  └─ train.py
├─ configs/                 # 训练/推理超参、数据路径（YAML/JSON）
│  └─ default.yaml
├─ envs/                    # 环境锁定文件（conda .yaml / requirements.txt）
│  ├─ env.yaml
│  └─ requirements.txt
├─ slurm/                   # 提交脚本与模板
│  ├─ grace_gpu_a100.sbatch
│  ├─ bouchet_gpu_h200.sbatch
│  └─ dsq_jobs.txt          # dSQ 作业清单（每行一条命令）
├─ scripts/                 # 数据预处理、评测脚本
│  ├─ preprocess.sh
│  └─ eval.sh
├─ logs/                    # 统一收集 slurm 日志（建议 slurm-%A_%a.out）
├─ notebooks/               # OOD/Jupyter 调试用
└─ README.md                # 用法、依赖、分区说明与常见问题
Slurm 输出名可使用 %j/%A/%a 模式变量，便于按 jobid/数组下标归档，例如 #SBATCH -o logs/slurm-%A_%a.out。此用法见 YCRC “Run Jobs with Slurm” → Common Job Request Options（包含输出命名、--time/--mem/--cpus-per-task/--gpus 等常用选项）。
2) 环境管理（Conda / 模块 / 容器）
Conda（推荐）：通过 miniconda 模块创建/激活环境；不要把 Python/R 模块与 Conda 环境混用；创建/安装包必须在计算节点上进行（salloc 申请交互节点再 module load miniconda）。示例与注意事项详见官方 Conda 指南。
Jupyter(OOD)：首次选择 ycrc_default 会自动构建；自建环境需包含 jupyter，并在构建后执行 ycrc_conda_env.sh update 让环境出现在 OOD 下拉列表。
模块（modules）：系统已预装常用软件；避免与 Conda 同时加载 Python/R 模块，以免冲突。
容器（Apptainer）：用于封装依赖/复现环境，支持在计算节点运行，官方提供容器开发指南。
3) 开发→调试→提交：最短闭环
交互调试（短时）
申请交互会话（默认进 devel 分区），例如：
 salloc -t 2:00:00 --mem=8G。
进入后 module load miniconda && conda activate <env>，小规模跑通核心流程。
锁定参数
将路径/超参写入 configs/default.yaml；避免在脚本里硬编码。
准备批处理脚本（sbatch）
参考下文模板，根据目标集群/分区/卡型调整 --partition 与 --gpus。
常用选项（--time/--mem/--cpus-per-task/--gpus/--output）详见 “Run Jobs with Slurm”。
4) GPU 分区与模板
4.1 Grace（公共 gpu 分区；需显式申请 GPU）
必须用 --gpus 申请 GPU，例如 --gpus=a5000:2；未申请到 GPU 不要占用 GPU 节点。
Grace · A100 单卡示例
#!/bin/bash
#SBATCH -J exp_grace_a100
#SBATCH -p gpu
#SBATCH --gpus=a100:1
#SBATCH -t 08:00:00
#SBATCH -c 4
#SBATCH --mem=30G
#SBATCH -o logs/slurm-%j.out

module reset
module load miniconda
conda activate YOUR_ENV

python -u src/train.py --config configs/default.yaml
说明：--gpus 为 YCRC 官方推荐申请方式；其它通用选项见“Common Job Request Options”。
4.2 Bouchet（H200/RTX 5000 Ada）
Bouchet 配有 80× NVIDIA H200 与 48× RTX 5000 Ada；面向 AI 工作负载。
Bouchet · H200 单卡示例
#!/bin/bash
#SBATCH -J exp_bch_h200
#SBATCH -p gpu_h200
#SBATCH --gpus=h200:1
#SBATCH -t 08:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -o logs/slurm-%j.out

module reset
module load miniconda
conda activate YOUR_ENV

python -u src/train.py --config configs/default.yaml
Bouchet · RTX 5000 Ada 单卡示例
#!/bin/bash
#SBATCH -J exp_bch_rtxada
#SBATCH -p gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH -t 08:00:00
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -o logs/slurm-%j.out

module reset && module load miniconda
conda activate YOUR_ENV
python -u src/train.py
提示：具体分区/默认上限以集群页为准；YCRC 也提供 PyTorch 模块与 GPU 指南（若选模块路径）。
4.3 可选：Priority & Scavenge
Priority Tier：快车道分区（如 priority_gpu），在队列中优先于标准层；适合必须尽快启动的作业。
Scavenge / scavenge_gpu：可被抢占的“捡漏”分区；Grace 亦有 scavenge_gpu。
5) dSQ 作业数组（高效批量提交）
当你有大量相似小任务，用 dSQ(Job Arrays) 比循环 sbatch 更好（回填/限速均更友好）。
步骤
写一个作业清单 slurm/dsq_jobs.txt：每行一条命令，例如
python -u src/train.py --config configs/default.yaml --seed 1
python -u src/train.py --config configs/default.yaml --seed 2
...
用 dSQ 生成并提交（常见 Slurm 选项可直接传入并转给 sbatch）：
dsq --job-file slurm/dsq_jobs.txt \
    -p gpu --gpus=a100:1 -t 4:00:00 -c 4 --mem=24G \
    --submit --batch-file slurm/dsq-%j.sbatch
YCRC 文档解释了 dSQ 的优势、写法与与 Slurm 集成方式；数组作业在“回填”与“Top-N 限制”统计上按一个作业计数，利于整体推进。
也可与 Nextflow/COMSOL 等结合使用。
6) 向作业传参（两种方式）
命令行参数 或 环境变量 注入，YCRC 给出 Python/R 示例与推荐做法；避免修改源码即可更换输入/超参。
7) 依赖编排（把多步流程串起来）
使用 Slurm 依赖机制：--dependency=afterok:<jobid> 等，将预处理→训练→评测串成流水线；官方给了依赖类型与示例。
8) 监控与优化（CPU/内存/GPU）
运行后用 seff <jobid> / sacct -j <jobid> 看效率；“Run Jobs with Slurm”明确给出命令与上下文。
Monitor CPU and Memory：文档教你 SSH 到作业所在节点查看瞬时占用；最大内存需等作业结束由记账生成。
jobstats（门户/命令行）：查看 CPU/Memory/GPU/GPU 内存的利用曲线与报告；还有“总体 Slurm 使用量（getusage）”。
资源申请策略：根据历史利用率下调到“刚好够”，尤其是 --time/--mem/--gpus/--cpus-per-task；更短墙时更利于回填启动。YCRC 在“Priority & Wait Time”中强调了数组作业与回填的交互。
9) 常见故障与限速
内存不足被 OOM：slurmstepd: error: Detected 1 oom-kill event(s) → 提高 --mem 或优化程序；详见“Common Job Failures”。
提交限速（避免刷队列）：集群对提交速率有限制（如文档提到200 jobs/hour 阈值）；使用 dSQ/依赖作业/Nextflow 的混合执行模式可避免超限。
10) 什么时候提交更容易上机（实践建议）
晚间（22:00–08:00 ET）与周末：公共 GPU 分区普遍更空；9 月学期中 Bouchet 课程使用多，白天更拥挤，夜晚/周末更友好（YCRC 明确课程用户可用公共分区；“更拥挤”的结论基于此政策与典型 HPC 负载规律的合理推断，具体以你实际 squeue/监控为准）。
必须马上跑：若你组开通 Priority Tier，优先投 priority_gpu。
实时判断：
 squeue --me 查看队列；sprio -j <jobid> 看优先级；配合门户状态与 getusage 评估组内 fairshare。关键命令与页面见 YCRC 的调度与监控文档。
附录 A：通用 CPU 作业模板
#!/bin/bash
#SBATCH -J exp_cpu
#SBATCH -p general
#SBATCH -t 04:00:00
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o logs/slurm-%j.out

module reset
module load miniconda
conda activate YOUR_ENV

python -u src/train.py --config configs/default.yaml --device cpu
常用参数详见“Run Jobs with Slurm”→Common Job Request Options。
附录 B：从 0 到 1 的最小提交流程（Checklist）
创建/激活环境（交互节点）：module load miniconda && conda create -n proj ... && conda activate proj；OOD 场景执行 ycrc_conda_env.sh update。
小规模调通（salloc）→ 保存超参到 configs/。
编写 sbatch（Grace/Bouchet 各一份）并设置日志命名。
批量任务 用 dSQ（数组作业）而不是循环 sbatch。
提交后监控：seff/sacct + jobstats + getusage；下一波作业据此缩短墙时/降低资源请求。
紧急任务：用 priority_gpu；能容忍中断的长作业考虑 scavenge/scavenge_gpu。根据您提供的节点配置信息，我整理了GPU性能排行表格：

## GPU性能排行（从高到低）

| 排名 | GPU型号 | 显存 | 性能特点 | 节点数量 | CPU/节点 | 内存/节点 | 适用场景 |
|------|---------|------|----------|----------|----------|-----------|----------|
| 1 | **A100-80G** | 80GB | 双精度，HBM2e高带宽内存 | 26+16=42节点 | 32-48核 | 984-2096GB | 大模型训练、科学计算 |
| 2 | **A100-40G** | 40GB | 双精度，HBM2内存 | 26节点 | 36核 | 361GB | 中等规模AI训练 |
| 3 | **A5000** | 24GB | 双精度，RTX架构 | 116节点 | 32核 | 206GB | 图形渲染、AI推理 |
| 4 | **V100** | 16GB | 双精度，Volta架构 | 46+26=72节点 | 24-36核 | 181-370GB | 传统深度学习 |
| 5 | **RTX 5000** | 16GB | 双精度，Turing架构 | 25节点 | 22核 | 181GB | 图形工作站任务 |
| 6 | **RTX 2080 Ti** | 11GB | 单精度，游戏卡 | 36节点 | 24核 | 181GB | 入门级AI训练 |

## 性能对比详细说明

### 顶级性能（A100系列）
- **A100-80G**: 最强算力，312 TFLOPS (FP16)，适合LLM训练
- **A100-40G**: 性价比之选，相同架构但显存减半

### 中高端（A5000/V100）
- **A5000**: 新一代专业卡，27.8 TFLOPS (FP32)，适合多任务
- **V100**: 经典选择，15.7 TFLOPS (FP32)，成熟稳定

### 入门级（RTX系列）
- **RTX 5000**: 专业图形卡，11.2 TFLOPS (FP32)
- **RTX 2080 Ti**: 消费级改装，13.4 TFLOPS (FP32)，仅单精度

## 选择建议

| 任务类型 | 推荐GPU | 理由 |
|---------|---------|------|
| **LLM/Transformer训练** | A100-80G > A100-40G | 大显存+高带宽必需 |
| **科学计算（双精度）** | A100 > V100 > A5000 | 需要FP64性能 |
| **CNN/传统DL** | A100-40G > V100 > A5000 | 性价比平衡 |
| **推理/部署** | A5000 > RTX 5000 | 功耗效率高 |
| **预算受限/原型验证** | RTX 2080 Ti | 最便宜，但限制多 |
| **混合精度训练** | A100系列 | Tensor Core性能最佳 |

## 特别提示
1. **A100-80G** 节点通常更难申请（资源稀缺）
2. **RTX 2080 Ti** 仅支持单精度，不适合科学计算
3. **V100** 虽然较老但节点数多，等待时间可能更短
4. 根据`Node Features`选择：需要大内存选`bigtmp`节点

建议根据实际需求（显存需求、精度要求、等待时间）选择合适的GPU，而非一味追求最高性能。





## ✅ **正确的提交方式**

### 方法1：使用 sbatch 命令提交（正确方式）
```bash
sbatch /vast/palmer/home.grace/ql334/LQ/PHM-Vibench/script/Vibench_paper/foundation_model/run.sbatch
sbatch /vast/palmer/home.grace/ql334/LQ/PHM-Vibench/script/Vibench_paper/foundation_model/run.sbatch
```

### 方法2：先查看文件内容
```bash
# 查看脚本内容，确认配置
cat /vast/palmer/home.grace/ql334/LQ/PHM-Vibench/script/Vibench_paper/foundation_model/run.sbatch

# 或用 less 查看长文件
less /vast/palmer/home.grace/ql334/LQ/PHM-Vibench/script/Vibench_paper/foundation_model/run.sbatch
```

### 方法3：检查并修复权限（如果需要）
```bash
# 查看当前权限
ls -la /vast/palmer/home.grace/ql334/LQ/PHM-Vibench/script/Vibench_paper/foundation_model/run.sbatch

# 如果需要添加执行权限（通常不需要）
chmod +x /vast/palmer/home.grace/ql334/LQ/PHM-Vibench/script/Vibench_paper/foundation_model/run.sbatch

# 但仍然要用 sbatch 提交
sbatch /vast/palmer/home.grace/ql334/LQ/PHM-Vibench/script/Vibench_paper/foundation_model/run.sbatch
```

## 📋 **完整工作流程**

```bash
# 1. 进入项目目录
cd /vast/palmer/home.grace/ql334/LQ/PHM-Vibench

# 2. 查看脚本内容（可选）
cat script/Vibench_paper/foundation_model/run.sbatch

# 3. 提交作业
sbatch script/Vibench_paper/foundation_model/run.sbatch

# 4. 查看作业状态
squeue --me

# 5. 如果需要取消作业
# scancel <jobid>
```

## 🔍 **常用 SLURM 命令**

| 命令 | 用途 |
|------|------|
| `sbatch script.sbatch` | 提交批处理作业 |
| `squeue --me` | 查看自己的作业队列 |
| `squeue -p gpu` | 查看GPU分区队列 |
| `scancel <jobid>` | 取消作业 |
| `sinfo -p gpu` | 查看GPU分区节点状态 |
| `sacct -j <jobid>` | 查看作业历史 |

## ⚠️ **注意事项**

1. **永远不要**直接执行 `.sbatch` 文件（`./run.sbatch`）
2. **永远使用** `sbatch` 命令提交
3. **确保在login节点**提交作业（您现在的位置是正确的）

## 🎯 **快速开始**

立即运行：
```bash
sbatch script/Vibench_paper/foundation_model/run.sbatch
```

提交后会返回类似：
```
Submitted batch job 12345678
```

然后查看状态：
```bash
squeue --me
```

如果脚本有问题或需要修改，请先用 `cat` 查看内容，我可以帮您调整参数。




