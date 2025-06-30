# ISFM model family

## 01

### embedding
- 基于HSE的embedding
### backbone

### head


## 02

### 1D MFDIT Embedding
- 该模块负责将输入的多通道时间序列数据进行补丁化和嵌入。
- 使用了可重用的 `SequencePatcher` 工具来实现高效的补丁化操作。
- 支持多种信号源（如振动、温度等）的通道映射。

### backbone

- 基于sota diffuser模型的backbone。


### head
- 涉及到pretraining 和 meanflow 的loss