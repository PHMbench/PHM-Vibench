# B−1：资源基线与验收范围

更新日期：2026-09-18。本页区分已有 CI、本轮本地测量和未测维度，不把不同机器的数据拼成通用上限。

## 1. 已有 CI：读取产物，不重复训练

- 源码：#266 `c31c888ec85c18c3db2f2cd2af89c5ec3e840867`，尚未合入 dev。
- [run 35315838858](https://github.com/PHMbench/PHM-Vibench/actions/runs/35315838858)，job 105507178658。
- 本轮重新解析原 JUnit：321 项，0 failure/error/skip；suite time 119.890 s。
- 306 项非公共用例累计 5.499 s；15 项公共 CLI Dummy 累计 112.556 s，单项范围 7.300–7.918 s。
- 最大非公共单项 3.423 s；最大测试文件组累计 3.640 s。
- job 墙钟 205 s；安装步骤 75 s；pytest 步骤 121 s。各口径分别使用，不相加成训练时间。

按文件统计保留于 [RESOURCE_OBSERVATIONS.csv](RESOURCE_OBSERVATIONS.csv)。这些是既有执行，不是本轮重新跑过 321 个测试，也不自动构成全部模型的 E2/E3。

## 2. 本轮本地补测：仅现有简单基线源码切片

从 dev@b4c3209c 读取未改动的
[src/model_factory/Baseline/GlobalAverageLinear.py](../../src/model_factory/Baseline/GlobalAverageLinear.py)。
在三个新 Python 进程中分别测量；不导入整个 PHMFactory，不安装包，不运行 Trainer、optimizer、THU 或新模型。

环境：Linux x86_64，Python 3.13.5，PyTorch 2.10.0+cpu；一线程；float32 `[4,128,2]`；两个类别；6 个参数。每进程一次预热前向，再执行 100 次无梯度前向。每个进程先测 torch import，再测单一源码模块增量 import。未清空操作系统文件缓存，不能称为“冷磁盘读取”。

| 测量 | 三次观测的中位数 | 范围 |
| --- | ---: | ---: |
| 新进程内 torch import | 0.859135 s | 0.791962–0.962508 s |
| torch 已导入后的基线模块 import | 0.201664 ms | 0.154012–0.738452 ms |
| 100 次前向累计 | 2.083534 ms | 2.062483–2.765901 ms |
| 进程生命周期墙钟 | 1.810962 s | 1.718754–1.911923 s |
| 该进程观测 VmHWM | 250.218750 MiB | 250.199219–250.250000 MiB |

原始样本见 [RESOURCE_LOCAL_BASELINE.csv](RESOURCE_LOCAL_BASELINE.csv)。
VmHWM 来自 Linux `/proc/self/status`，原始 kB 除以 1024 转为 MiB；它包含解释器、框架和分配缓存，**不是模型净内存或训练峰值**。不能将两个阶段的高水位差当独立分配量。所有计时使用 perf_counter；新进程退出检查为非零即失败。

复现脚本、原样模型源码和 JSON 留在本轮交付包 `evidence/`，没有添加生产脚本。本地结果不与 Python 3.10 / torch 2.6 的 CI 时间直接比较；没有由这三个样本推导总体置信区间。

## 3. 保留的 CI 时间警戒值：工程决策而非性能事实

| 对象 | 已有依据 | 暂定警戒值 |
| --- | --- | ---: |
| focused 组件文件组 | 最大现有文件组 3.640 s，含导入/Task 成本 | 8 s |
| 单模型公共 Dummy CLI | 最大已测单项 7.918 s | 16 s |
| 同规模 native suite | 119.890 s | 240 s |
| 同配置环境 native job | 205 s | 410 s |

这些值沿用先前“约两倍余量”的提案，只适合同 runner 和相近 tiny 配置的复核，不是普遍模型配额。尤其不把测试文件组时间改称单模型时间。超出先检查 fixture、算法、导入和依赖，不缩短输入、换模型、删反例或改 timeout 来过关。本轮未改变任何 workflow 阈值。

## 4. 当前仍未具备的资源证据

| 维度 | 状态 | 不能做的推论 |
| --- | --- | --- |
| 当前基线的独立源码导入和推断 RSS | LOCAL_SLICE_MEASURED | 不等于完整框架、安装后或训练资源 |
| PHMFactory 整体冷进程 import | NOT_RUN | 不能把单模块 import 当整个 package import |
| 完整训练峰值 CPU/GPU 内存 | NOT_RUN | 不能用 6 参数基线的推断高水位给所有模型统一配额 |
| 正常 wheel 安装及可选依赖增量 | NOT_RUN | 不用现有环境或 --no-deps 冒充独立解析安装 |
| 真实 PHM/GPU 资源预算 | NOT_IN_SCOPE | 不为 B−1 文档重新训练 THU |
| 新 nightly 矩阵 | NOT_CREATED | 不为了补一个预算字段引入周期基础设施 |

本容器对 github.com 的 Git 访问返回 DNS 解析失败，因此没有完成完整 checkout 安装。已通过连接器取得的源码只按上述切片范围使用。该环境限制不是仓库产品缺陷。

## 5. G7 和阶段验收

B−1 可以冻结科学选择规则、测量方法和已获得的局部基线；它不自动批准任何候选入队。对于将进入实现的模型，G7 必须给出该模型适用环境的预算依据；未知项仍未知。E0/E1/E2/E3 不由资源测量替代。

本轮完成局部导入/RSS补测，完整资源包络与 fresh-context reviewer 尚未完成，不将 #267 标为已独立验收或合入 dev。保持 #266 冻结；B00 的实际拆分/关闭不在本轮执行。

PR 验证仍是对应合同、fidelity 和 tiny 公共路径；共享层变动覆盖其真实消费者。可选重依赖和真实 PHM 仅在已批准的具体范围运行，不将 skip 当通过，不创建新通用 registry 或资源管理器。
