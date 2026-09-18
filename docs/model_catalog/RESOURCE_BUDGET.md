# B−1：已有资源证据与预算

仅复用既有执行，不启动模型训练。所有数值是一个 runner、一个配置集合的一次运行，不代表跨机器上限。

## 1. 证据

- 被观测源码：PR #266 `c31c888ec85c18c3db2f2cd2af89c5ec3e840867`；此快照未入 dev。
- [GitHub run 35315838858](https://github.com/PHMbench/PHM-Vibench/actions/runs/35315838858)，job 105507178658。
- 下载 artifact 10535450712，读取 `native-models.xml`，没有重跑。
- 321 个 testcase，0 failure、0 error、0 skip；Junit suite time 119.890 s。
- 306 个非公共运行用例时间之和 5.499 s；15 个公共 CLI Dummy 用例时间之和 112.556 s，单个范围 7.300–7.918 s。
- 单个非公共测试最大 3.423 s；按测试文件分组的最大总和 3.640 s。
- GitHub job 墙钟 205 s；依赖安装步骤 75 s；pytest 步骤 121 s。三种计时口径不相加、不混作模型训练时间。

按文件统计见 [RESOURCE_OBSERVATIONS.csv](RESOURCE_OBSERVATIONS.csv)。完整原 JUnit 在交付包 `evidence/`。已有测试证明该 CI 范围的执行，不单独构成所有移植的 E2 或真实 PHM 的 E3。

## 2. 可冻结的暂定时间包络

下列是预算决策，不是新的测量。先用同 runner/相近 tiny 配置，按已观测值的 2 倍给波动余量，取整到秒：

| 对象 | 依据 | 本轮暂定警戒值 | 超出时的处理 |
| --- | --- | --- | --- |
| 一模型 focused 组件验证 | 最大现有文件组 3.640 s；组含导入/Task 成本，不等价逐模型成本 | 8 s | 记录实际范围；复测/解释，不立即定性回归 |
| 一模型公共 Dummy CLI | 最大现有单项 7.918 s | 16 s | 核对 fixture、import 和模型；不能降低模型配置偷过预算 |
| 同规模 native suite | 119.890 s | 240 s | 防止无关模型每次重跑；不是允许无限扩容 |
| 同配置环境的 native job | 205 s | 410 s | 分开记录安装与运行；依赖网络波动不伪称算法退化 |

2 倍余量是显式工程选择，不是从单样本推导的置信区间。现有 workflow timeout 和测试超时一律不在 B−1 中修改。复杂模型可有单独 optional 预算；超出上述值先 review，不自动替换网络、删测试或标失败。

## 3. 未测维度

| 维度 | 当前状态 | 实现入队前所需动作 |
| --- | --- | --- |
| 峰值 CPU/GPU 内存 | UNMEASURED | 先在同环境量现有 simple baseline，再测拟选模型；记录方法与单位 |
| 冷 import 时间 | UNMEASURED | 分离进程启动、框架 import、模型 import；禁止凭源码行数估计 |
| wheel 大小/可选依赖增量 | UNMEASURED | 对同 Python/平台正常解析并记录增量；不得用 --no-deps 证明 |
| GPU/真实 PHM 训练预算 | NOT_IN_THIS_RUN | 选定 exact Task/config 后定义；不为文档重跑 THU |
| nightly 总预算 | NOT_DEFINED | 本仓没有本次批准的新周期任务，不创建空 nightly 矩阵 |

因此，本轮完成时间基线和预算方法冻结，但不能宣称完整资源预算全部校准。未知维度明确保留；任何新模型 G7 不能仅凭这个文档自动 PASS。

## 4. 测试分层决策

PR：本次变更对应的 E1/E2 + tiny 公共路径；共享算子变动重跑实际消费者。安装/打包变动跑正常 wheel；不无差别下载权重或数据。

可选周期/人工：确有维护需求及预算后执行可选依赖矩阵、真实 PHM 小运行。不是本轮创建的计划任务。

Release：正常 installed-wheel、已批准 task/model 代表矩阵，以及实际声明的 E3。支持未完成的 optional 条目保持未完成，不用 skip 凑全绿。
