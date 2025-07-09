---
license: Apache License 2.0
---
# 🛠️ Vbench - 机械振动数据集集合

Vbench是一个包含多个轴承故障诊断数据集的集合项目。该项目收集了来自全球多个研究机构的轴承振动数据，用于机械故障诊断和预测性维护研究。

## TODO

- [ ] 完整数据介绍
- [ ] 完整数据分析
- [ ] 贡献方式
- [x] 上传demo数据
- [ ] ID dataset 设置

## 📂 数据集文件
数据集文件元信息以及数据文件，请浏览"数据集文件"页面获取。

- *.csv 元数据
- *.h5 根据元数据组织的数据集


当前数据集卡片使用的是默认模版，数据集的贡献者未提供更加详细的数据集介绍，但是您可以通过如下GIT Clone命令，或者ModelScope SDK来下载数据集

### ⬇️ 下载方法 
:modelscope-code[]{type="sdk"}
:modelscope-code[]{type="git"}

## 📊 数据集介绍

Vbench包含以下轴承故障诊断数据集:

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
| 10 | 江南大学轴承数据集 | JNU | RM_010_JNU | ✅ 开源 | [详情](#jnu) |
| 11 | 渥太华变转速数据集2 | Ottawa19 | RM_011_Ottawa19 | ✅ 开源 | [详情](#ottawa19) |
| 12 | 清华大学压电、摩擦电数据集 | THU24 | RM_012_THU24 | 🔒 非开源 | [详情](#thu24) |
| 13 | 哈工大双转子数据集 | HIT | RM_013_HIT | ✅ 开源| [详情](#hit) |
| 14 | 都灵理工数据集 | DIRG | RM_014_DIRG | ✅ 开源 | [详情](#dirg) |
| 15 | 德国帕德博恩大学 | KAT(PU) | RM_015_KAT | ✅ 开源 | [详情](#katpu) |
| 18 | 华科数据集 | HUST | RM_018_HUST | ✅ 开源 | [详情](#hust) |
| 19 | 讯飞数据集 | IFlytek | RM_019_IFlytek | ✅ 开源 | [详情](#iflytek) |
| 20 | 哈工大轴承数据集 | HIT23 | RM_020_HIT | ✅ 开源 | [详情](#hit_bearing) |
| 21 | 苏科大轴承数据集 | SUST | RM_021_SUST | ✅ 开源 | [详情](#sust) |


<!-- | 9 | 苏大小轴承数据集 | SUDA_shen | RM_009_SUDA_shen | 🔒 非开源 | [详情](#suda_shen) | -->
<!-- | 16 | 苏大飞轮轴承数据 | SUDA_FW | RM_016_SUDA_FW | 🔒 非开源 | [详情](#suda_fw) |
| 17 | 苏大2023新轮对轴承数据 | SUDA23 | RM_017_SUDA23 | 🔒 非开源 | [详情](#suda23) | -->
### 🔍 齿轮箱数据集总览


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


## dataset structure

```
src/data_factory/dataset_task/
├── __init__.py
├── Default_dataset.py
├── DG/
│   └── classification_dataset.py
├── CDDG/
│   └── classification_dataset.py
├── FS/               
│   └── classification_dataset.py
├── Pretrain/
│   └── classification_dataset.py
└── ... (其他任务类型)
```

## 🔗 项目链接

GitHub: [https://github.com/PHMbench/Vbench](https://github.com/PHMbench/Vbench)

