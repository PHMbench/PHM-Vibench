---
license: Apache License 2.0
---

# 🛠️ PHM-Vibench振动信号基准数据库

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![Size](https://img.shields.io/badge/Size-163GB-orange)](https://github.com/example/PHM-Vibench)

## 🔗 数据获取平台

[![ModelScope](https://img.shields.io/badge/ModelScope-PHM--Vibench-red)](https://www.modelscope.cn/datasets/PHMbench/PHM-Vibench/files)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-PHM--Vibench-yellow)](https://huggingface.co/datasets/PHMbench/PHM-Vibench/tree/main)

> **PHM-Vibench** 是一个全面的工业设备振动信号分析基准平台，该项目是其中用到的数据，专注于机械故障诊断和预测性维护研究。

## 📋 目录

- [📊 项目概览](#-项目概览)
- [🗂️ 数据集目录](#️-数据集目录)
- [🚀 快速开始](#-快速开始)
- [📁 项目结构](#-项目结构)
- [🔧 工具脚本](#-工具脚本)
- [📈 数据统计](#-数据统计)
- [🤝 如何贡献](#-如何贡献)
- [📜 更新日志](#-更新日志)

## 📊 项目概览

PHM-Vibench是一个包含多个轴承故障诊断数据集的集合项目。该项目收集了来自全球多个研究机构的轴承振动数据，用于机械故障诊断和预测性维护研究。

### 技术特性

- **数据规模**: 163GB+ 原始振动数据
- **数据集数量**: 30+ 个测试台数据
- **样本总数**: 49,000+ 个振动样本
- **故障类型**: 正常、内圈故障、外圈故障、滚动体故障、复合故障等
- **数据格式**: .mat, .csv, .xlsx, .h5 等多种格式，整合成h5文件
- **采样频率**: 6Hz - 512kHz 范围
- **应用场景**: 机械故障诊断、预测性维护、状态监测

### 数据集统计

| 统计项        | 数值                               |
| ------------- | ---------------------------------- |
| 📁 数据集数量 | 19个主要数据集 + 更多收集中        |
| 📊 样本总数   | 49,000+                            |
| 💾 数据大小   | 163GB+                             |
| 🏭 测试台数量 | 30+                                |
| 🌍 来源国家   | 中国、美国、加拿大、德国、意大利等 |

## 🗂️ 数据集目录

### 🔍 轴承故障数据集总览（调整数据顺序）

| 序号 |   数据集名称   |   缩写   |      VB_name      |  开源状态  |                详情链接(TODO)                |
| :--: | :-------------: | :------: | :----------------: | :---------: | :-------------------------------------------: |
|  1  |  凯斯西储大学  |   CWRU   |    RM_001_CWRU    | ✅ 经典开源 |  [详情](#rm_001_cwru---凯斯西储大学轴承数据集)  |
|  2  |  西安交通大学  |   XJTU   |    RM_002_XJTU    |   ✅ 开源   |  [详情](#rm_002_xjtu---西安交通大学轴承数据集)  |
|  3  | FEMTO-ST研究所 |  FEMTO  |    RM_003_FEMTO    |   ✅ 开源   |   [详情](#rm_003_femto---femto-st研究所数据集)   |
|  4  | IMS齿轮轴承数据 |   IMS   |     RM_004_IMS     |   ✅ 开源   |      [详情](#rm_004_ims---ims齿轮轴承数据)      |
|  5  | 渥太华大学2023 | Ottawa23 |  RM_005_Ottawa23  |   ✅ 开源   | [详情](#rm_005_ottawa23---渥太华大学2023数据集) |
|  6  |    清华大学    |   THU   |     RM_006_THU     |   ✅ 开源   |       [详情](#rm_006_thu---清华大学数据集)       |
|  7  |    MFPT学会    |   MFPT   |    RM_007_MFPT    |   ✅ 开源   |      [详情](#rm_007_mfpt---mfpt学会数据集)      |
|  8  | 新南威尔士大学 |   UNSW   |    RM_008_UNSW    |   ✅ 开源   |   [详情](#rm_008_unsw---新南威尔士大学数据集)   |
|  9  | 东南大学(轴承) |   SEU   |     RM_010_SEU     |   ✅ 开源   | [详情](#rm_009_seu_bearing---东南大学轴承数据集) |
|  10  |        -        |    -    | **暂无数据** |   🔒 空缺   |                 RM_010不存在                 |
|  11  |      越南      |   susu   |    RM_015_susu    |   ✅ 开源   |   [详情](#rm_011_susu---苏州大学苏老师数据集)   |
|  12  |    江南大学    |   JNU   |     RM_016_JNU     |   ✅ 开源   |       [详情](#rm_012_jnu---江南大学数据集)       |
|  13  | 渥太华大学2019 | Ottawa19 |  RM_017_Ottawa19  |   ✅ 开源   | [详情](#rm_013_ottawa19---渥太华大学2019数据集) |
|  14  |  清华大学2024  |  THU24  |    RM_018_THU24    |   ✅ 开源   |    [详情](#rm_014_thu24---清华大学2024数据集)    |
|  15  | 东南大学(齿轮) |   SEU   |     RM_010_SEU     |   ✅ 开源   |  [详情](#rm_015_seu_gear---东南大学齿轮数据集)  |
|  16  |  都灵理工大学  |   DIRG   |    RM_020_DIRG    |   ✅ 开源   |    [详情](#rm_016_dirg---都灵理工大学数据集)    |
|  17  |   哈工大2023   |  HIT23  |    RM_023_HIT23    |   ✅ 开源   |     [详情](#rm_017_hit23---哈工大2023数据集)     |
|  18  |  江苏科技大学  |   JUST   |    RM_024_JUST    |   ✅ 开源   |      [详情](#rm_018_just---just大学数据集)      |
|  19  |    华科2024    |  HUST24  |   RM_031_HUST24   |   ✅ 开源   |     [详情](#rm_019_hust24---华科2024数据集)     |
|  20  |  帕德博恩大学  |    PU    |     RM_027_PU     |   ✅ 开源   |     [详情](#rm_020_pu---帕德博恩大学数据集)     |


# 🛠️ [Vbench](https://github.com/PHMbench/PHM-Vibench/tree/main/src/data_factory)  

Vbench是一个包含多个轴承故障诊断数据集的集合项目。该项目收集了来自全球多个研究机构的轴承振动数据，用于机械故障诊断和预测性维护研究。

## TODO

- [ ] 完整数据介绍
- [ ] 完整数据分析
- [ ] 贡献方式
- [x] 上传demo数据

## 📂 数据集文件
数据集文件元信息以及数据文件，请浏览"数据集文件"页面获取。



<!-- ## 📊 数据集介绍 --> 已更新

<!-- Vbench包含以下轴承故障诊断数据集:

<div align="center" id="dataset-table">

### 🔍 轴承故障数据集总览

| 序号 | 数据集名称 | 缩写 | VB_name | 开源状态 | 详情链接 |
|:----:|:--------:|:----:|:-------:|:-------:|:-------:|
| 1 | 凯斯西储 | CWRU | RM_001_CWRU | ✅ 经典开源 | [详情](#cwru) |
| 2 | 西安交通大学 | XJTU | RM_002_XJTU | ✅ 开源 | [详情](#xjtu) |
| 3 | 渥太华变转速数据集1 | Ottawa | RM_003_Ottawa | ✅ 开源 | [详情](#ottawa) |
| 4 | 清华大学压电数据集 | THU | RM_004_THU | 🔒 非开源 | [详情](#thu) |
| 5 | 美国机械故障预防技术学会 | MFPT | RM_005_MFPT | ✅ 开源 | [详情](#mfpt) |
| 6 | 新南威尔士大学轴承衰退数据集 | UNSW | RM_006_UNSW | ✅ 开源 | [详情](#unsw) |
| 7 | 东南大学数据集 | SEU | RM_007_SEU | ✅ 开源 | [详情](#seu) |
| 8 | 山东科技大学轴承数据集 | SDUST | RM_008_SDUST | 🔒 非开源 | [详情](#sdust) |
<!-- | 9 | 苏大小轴承数据集 | SUDA_shen | RM_009_SUDA_shen | 🔒 非开源 | [详情](#suda_shen) | -->
<!-- | 10 | 江南大学轴承数据集 | JNU | RM_010_JNU | ✅ 开源 | [详情](#jnu) |
| 11 | 渥太华变转速数据集2 | Ottawa19 | RM_011_Ottawa19 | ✅ 开源 | [详情](#ottawa19) |
| 12 | 清华大学压电、摩擦电数据集 | THU24 | RM_012_THU24 | 🔒 非开源 | [详情](#thu24) |
| 13 | 哈工大双转子数据集 | HIT | RM_013_HIT | ✅ 开源| [详情](#hit) |
| 14 | 都灵理工数据集 | DIRG | RM_014_DIRG | ✅ 开源 | [详情](#dirg) |
| 15 | 德国帕德博恩大学 | KAT(PU) | RM_015_KAT | ✅ 开源 | [详情](#katpu) |
<!-- | 16 | 苏大飞轮轴承数据 | SUDA_FW | RM_016_SUDA_FW | 🔒 非开源 | [详情](#suda_fw) |
| 17 | 苏大2023新轮对轴承数据 | SUDA23 | RM_017_SUDA23 | 🔒 非开源 | [详情](#suda23) | -->
<!-- | 18 | 华科数据集 | HUST | RM_018_HUST | ✅ 开源 | [详情](#hust) |
| 19 | 讯飞数据集 | IFlytek | RM_019_IFlytek | ✅ 开源 | [详情](#iflytek) |
| 20 | 哈工大轴承数据集 | HIT23 | RM_020_HIT | ✅ 开源 | [详情](#hit_bearing) |
| 21 | 苏科大轴承数据集 | SUST | RM_021_SUST | ✅ 开源 | [详情](#sust) | --> -->

### 🔍 齿轮箱数据集总览 -->


</div>

## 📋 详细数据集信息

<details>
<summary><strong>点击展开所有数据集详细信息</strong></summary>

<a id="cwru"></a>
<details>
<summary><h3>1️⃣ 凯斯西储大学轴承数据集 (CWRU) - RM_001_CWRU</h3></summary>

- **轴承型号**: 🔩 SKF6205/SKF6203
- **转速范围**: ⚙️ 1730~1797rpm
- **负载条件**: ⚖️ 0hp,1hp,2hp,3hp
- **故障类型**: ⚠️ 正常(NC),内圈故障(IF),滚动体故障(BF),外圈故障(OF)
- **故障程度**: 📏 0.007",0.014",0.021"
- **采样频率**: 📶 12kHz,48kHz
- **采样时长**: ⏱️ 40.325s
- **通道数量**: 🔌 3
- **特别说明**: 📝 NC,IF,BF,OF分别为正常，内圈故障，滚动体故障，外圈故障
- **引用文献**: 📚 Rolling element bearing diagnostics using the case western reserve university data: A benchmark study

[返回数据集列表](#dataset-table)
</details>

<a id="xjtu"></a>
<details>
<summary><h3>2️⃣ 西安交通大学轴承数据集 (XJTU) - RM_002_XJTU</h3></summary>

- **轴承型号**: 🔩 LDK UER204
- **转速范围**: ⚙️ 2100/2250/2400rpm
- **负载条件**: ⚖️ 12/11/10kN
- **故障类型**: ⚠️ 内圈磨损/外圈磨损/外圈裂损/保持架断裂
- **故障程度**: 📏 /
- **采样频率**: 📶 25.6kHz
- **采样时长**: ⏱️ 1.28s
- **通道数量**: 🔌 2
- **引用文献**: 📚 XJTU-SY滚动轴承加速寿命试验数据集解读

[返回数据集列表](#dataset-table)
</details>

<a id="ottawa"></a>
<details>
<summary><h3>3️⃣ 渥太华变转速数据集1 (Ottawa) - RM_003_Ottawa</h3></summary>

- **轴承型号**: 🔩 MFS-PK5M
- **转速范围**: ⚙️ 变转速(升速、降速、先升后降、先降后升)
- **故障类型**: ⚠️ 正常/内圈缺陷/外圈缺陷
- **采样频率**: 📶 200kHz
- **采样时长**: ⏱️ 10s
- **通道数量**: 🔌 2
- **特别说明**: 📝 一个通道为振动，一个通道为转速
- **引用文献**: 📚 Bearing vibration data collected under time-varying rotational speed conditions

[返回数据集列表](#dataset-table)
</details>

<a id="thu"></a>
<details>
<summary><h3>4️⃣ 清华大学压电数据集 (THU) - RM_004_THU</h3></summary>

- **轴承型号**: 🔩 6204
- **转速范围**: ⚙️ 1Hz,10Hz,15Hz
- **故障类型**: ⚠️ 正常(NC),内圈故障(IF),滚动体故障(BF),外圈故障(OF)
- **故障程度**: 📏 0.5mm
- **采样频率**: 📶 20480Hz
- **采样时长**: ⏱️ 60s
- **通道数量**: 🔌 2
- **特别说明**: 📝 一个通道为振动，一个通道为电压
- **引用文献**: 📚 Piezoelectric energy harvester for rolling bearings with capability of self-powered condition monitoring

[返回数据集列表](#dataset-table)
</details>

<a id="mfpt"></a>
<details>
<summary><h3>5️⃣ 美国机械故障预防技术学会数据集 (MFPT) - RM_005_MFPT</h3></summary>

- **轴承型号**: 🔩 NICE轴承
- **转速范围**: ⚙️ 25Hz
- **负载条件**: ⚖️ 0-300lbs
- **故障类型**: ⚠️ 正常/外圈故障/内圈故障
- **采样频率**: 📶 97656/48828sps
- **采样时长**: ⏱️ 3/6s
- **数据链接**: 🔗 https://mfpt.org/fault-data-sets/

[返回数据集列表](#dataset-table)
</details>

<a id="unsw"></a>
<details>
<summary><h3>6️⃣ 新南威尔士大学轴承衰退数据集 (UNSW) - RM_006_UNSW</h3></summary>

- **转速范围**: ⚙️ 6Hz,12Hz,15Hz,20Hz
- **故障类型**: ⚠️ 正常/内圈故障/外圈故障
- **采样频率**: 📶 6Hz
- **通道数量**: 🔌 6
- **特别说明**: 📝 水平、垂直加速度，编码器信号，负载和转速信号
- **引用文献**: 📚 A benchmark of measurement approaches to track the natural evolution of spall severity in rolling element bearings

[返回数据集列表](#dataset-table)
</details>

<a id="seu"></a>
<details>
<summary><h3>7️⃣ 东南大学数据集 (SEU) - RM_007_SEU</h3></summary>

- **转速范围**: ⚙️ 1200/1800rpm
- **负载条件**: ⚖️ 0/2V
- **故障类型**: ⚠️ 正常/滚动体/内圈/外圈/复合故障
- **采样频率**: 📶 5120Hz
- **通道数量**: 🔌 8
- **特别说明**: 📝 电机振动、行星齿轮箱三轴振动、电机扭矩、平行齿轮箱三轴振动
- **引用文献**: 📚 Highly-Accurate Machine Fault Diagnosis Using Deep Transfer Learning

[返回数据集列表](#dataset-table)
</details>

<a id="sdust"></a>
<details>
<summary><h3>8️⃣ 山东科技大学轴承数据集 (SDUST) - RM_008_SDUST</h3></summary>

- **转速范围**: ⚙️ 1000-6000r/min
- **负载条件**: ⚖️ 0/20/40/60N
- **故障类型**: ⚠️ 正常/内圈故障/外圈故障
- **故障程度**: 📏 0.2mm-0.4mm
- **采样频率**: 📶 25.6kHz
- **采样时长**: ⏱️ 40s
- **特别说明**: 📝 包含位移和加速度信号
- **引用文献**: 📚 Han Baokun et al. Hybrid distance-guided adversarial network for intelligent fault diagnosis under different working conditions. Measurement, 2021

[返回数据集列表](#dataset-table)
</details>

<a id="suda_shen"></a>
<details>
<summary><h3>9️⃣ 苏大小轴承数据集 (SUDA_shen) - RM_009_SUDA_shen</h3></summary>

- **特别说明**: 📝 引用MSSP的KMADA，TII的ADIG
- **数据获取**: 🔒 需联系苏州大学沈长青老师

[返回数据集列表](#dataset-table)
</details>

<a id="jnu"></a>
<details>
<summary><h3>1️⃣0️⃣ 江南大学轴承数据集 (JNU) - RM_010_JNU</h3></summary>

- **转速范围**: ⚙️ 600/800/1000rpm
- **故障类型**: ⚠️ 正常/内圈/外圈/滚动体故障
- **故障程度**: 📏 0.3×0.25/0.5×0.15mm
- **采样频率**: 📶 50kHz
- **采样时长**: ⏱️ 20s
- **通道数量**: 🔌 1
- **特别说明**: 📝 单一加速度传感器在垂直方向采集
- **引用文献**: 📚 Sequential fuzzy diagnosis method for motor roller bearing in variable operating conditions based on vibration analysis

[返回数据集列表](#dataset-table)
</details>

<a id="ottawa19"></a>
<details>
<summary><h3>1️⃣1️⃣ 渥太华变转速数据集2 (Ottawa19) - RM_011_Ottawa19</h3></summary>

- **转速范围**: ⚙️ 1700-2000rpm
- **负载条件**: ⚖️ 0/400
- **故障类型**: ⚠️ 正常/内圈/球/外圈/保持架故障
- **采样频率**: 📶 42000Hz
- **采样时长**: ⏱️ 10s
- **通道数量**: 🔌 6
- **特别说明**: 📝 四种转速变化下的故障数据
- **引用文献**: 📚 University of ottawa constant load and speed rolling-element bearing vibration and acoustic fault signature datasets

[返回数据集列表](#dataset-table)
</details>

<a id="thu24"></a>
<details>
<summary><h3>1️⃣2️⃣ 清华大学压电、摩擦电数据集 (THU24) - RM_012_THU24</h3></summary>

- **转速范围**: ⚙️ 12/15/16/20Hz
- **故障类型**: ⚠️ 正常/内圈/外圈/滚子/带状笼裂纹
- **引用文献**: 📚 A hybrid triboelectric-piezoelectric smart squirrel cage with self-sensing and self-powering capabilities
- **数据获取**: 🔒 需联系清华大学秦朝烨老师

[返回数据集列表](#dataset-table)
</details>

<a id="hit"></a>
<details>
<summary><h3>1️⃣3️⃣ 哈工大双转子数据集 (HIT) - RM_013_HIT</h3></summary>

- **转速范围**: ⚙️ 1000-6000rpm
- **故障类型**: ⚠️ 正常/内圈故障/外圈故障
- **故障程度**: 📏 0.5×0.5/0.5×1.0mm
- **采样频率**: 📶 25000Hz
- **采样时长**: ⏱️ 15s
- **通道数量**: 🔌 8
- **引用文献**: 📚 Inter-shaft bearing fault diagnosis based on aero-engine system: A benchmarking dataset study

[返回数据集列表](#dataset-table)
</details>

<a id="dirg"></a>
<details>
<summary><h3>1️⃣4️⃣ 都灵理工数据集 (DIRG) - RM_014_DIRG</h3></summary>

- **转速范围**: ⚙️ 100-500Hz
- **负载条件**: ⚖️ 100-500Hz
- **故障类型**: ⚠️ 正常/内圈故障/滚子故障
- **故障程度**: 📏 0/150/250/450μm
- **采样频率**: 📶 51200Hz
- **采样时长**: ⏱️ 10s
- **通道数量**: 🔌 6
- **引用文献**: 📚 The politecnico di torino rolling bearing test rig: Description and analysis of open access data

[返回数据集列表](#dataset-table)
</details>

<a id="katpu"></a>
<details>
<summary><h3>1️⃣5️⃣ 德国帕德博恩大学数据集 (KAT(PU)) - RM_015_KAT</h3></summary>

- **轴承型号**: 🔩 6203
- **转速范围**: ⚙️ 900-1500rpm
- **负载条件**: ⚖️ 0.7-1kN
- **故障类型**: ⚠️ Single/Repetitive/Multiple damage
- **故障程度**: 📏 ≤2->31.5mm
- **采样频率**: 📶 64kHz
- **采样时长**: ⏱️ 4s
- **通道数量**: 🔌 电流2
- **特别说明**: 📝 N15_M07_F10代表转速1500rpm，扭矩0.7Nm，径向力1000N
- **数据链接**: 🔗 https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter/data-sets-and-download

[返回数据集列表](#dataset-table)
</details>

<a id="suda_fw"></a>
<details>
<summary><h3>1️⃣6️⃣ 苏大飞轮轴承数据 (SUDA_FW) - RM_016_SUDA_FW</h3></summary>

- **特别说明**: 📝 见沈长青老师网盘
- **数据获取**: 🔒 需联系苏州大学沈长青老师

[返回数据集列表](#dataset-table)
</details>

<a id="suda23"></a>
<details>
<summary><h3>1️⃣7️⃣ 苏大2023新轮对轴承数据 (SUDA23) - RM_017_SUDA23</h3></summary>

- **特别说明**: 📝 见沈长青老师网盘
- **数据获取**: 🔒 需联系苏州大学沈长青老师

[返回数据集列表](#dataset-table)
</details>

<a id="hust"></a>
<details>
<summary><h3>1️⃣8️⃣ 华科数据集 (HUST) - RM_018_HUST</h3></summary>

- **轴承型号**: 🔩 ER-16K
- **转速范围**: ⚙️ 20-80Hz(恒定)，0-40-0Hz(时变)
- **故障类型**: ⚠️ 正常/内圈/外圈/球/组合故障(中度和严重)
- **故障程度**: 📏 Inner/outer:0.3/0.15mm，Ball:0.5/0.25mm
- **采样频率**: 📶 25.6kHz
- **采样时长**: ⏱️ 10.2s
- **通道数量**: 🔌 3
- **引用文献**: 📚 Chao Zhao et al. Domain Generalization for Cross-Domain Fault Diagnosis: an Application-oriented Perspective and a Benchmark Study. RESS, 2024

[返回数据集列表](#dataset-table)
</details>

</details>

## 🔗 项目链接

GitHub: [https://github.com/PHMbench/Vbench](https://github.com/PHMbench/Vbench)

