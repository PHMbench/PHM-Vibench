from src.model_factory.ISFM.embedding import *
from src.model_factory.ISFM.backbone import *
from src.model_factory.ISFM.task_head import H_02_Linear_cla_heterogeneous_batch
from src.model_factory.ISFM.system_utils import resolve_batch_metadata
from src.utils.utils import get_num_classes, get_num_channels

import torch.nn as nn
import torch


Embedding_dict = {
    "E_01_HSE": E_01_HSE,
    "E_02_HSE_v2": E_02_HSE_v2,
    "E_03_Patch": E_03_Patch,
}

Backbone_dict = {
    "B_01_basic_transformer": B_01_basic_transformer,
    "B_03_FITS": B_03_FITS,
    "B_04_Dlinear": B_04_Dlinear,
    "B_05_Mamba": B_05_Mamba,
    "B_06_TimesNet": B_06_TimesNet,
    "B_07_TSMixer": B_07_TSMixer,
    "B_08_PatchTST": B_08_PatchTST,
    "B_09_FNO": B_09_FNO,
}


class Model(nn.Module):
    """
    M_02_ISFM_heterogeneous_batch:

    - 支持“一个 batch 内混合多个 Dataset_id”的多系统场景；
    - 通过 resolve_batch_metadata 获取 per-sample system_id / Sample_rate；
    - 结合 H_02_Linear_cla_heterogeneous_batch，实现真正的异构 batch 分类。
    """

    def __init__(self, args_m, metadata):
        super().__init__()
        self.metadata = metadata
        self.args_m = args_m

        # num_classes / num_channels 映射
        self.num_classes = get_num_classes(metadata)
        self.num_channels = get_num_channels(metadata)
        args_m.num_classes = self.num_classes
        args_m.num_channels = self.num_channels

        # Embedding / Backbone / Head
        self.embedding = Embedding_dict[args_m.embedding](args_m)
        self.backbone = Backbone_dict[args_m.backbone](args_m)
        self.task_head = H_02_Linear_cla_heterogeneous_batch(args_m)

    def _embed(self, x, file_id):
        """1. Embedding：按样本使用 system_id / Sample_rate。"""
        system_ids_tensor, sample_f_tensor = resolve_batch_metadata(
            self.metadata, file_id_batch=file_id, device=x.device
        )

        if self.args_m.embedding == "E_01_HSE":
            x = self.embedding(x, sample_f_tensor)
            c = None
        elif self.args_m.embedding == "E_02_HSE_v2":
            x, c = self.embedding(x, system_ids_tensor, sample_f_tensor)
        else:
            x = self.embedding(x)
            c = None
        return x, c

    def _encode(self, x, c=None):
        """2. Backbone 编码。"""
        if c is None:
            return self.backbone(x)
        return self.backbone(x, c)

    def _head(self, x, file_id=False, task_id=False, return_feature=False):
        """3. Head：传入 per-sample system_id，让 head 处理异构 batch。"""
        system_ids_tensor, _ = resolve_batch_metadata(
            self.metadata, file_id_batch=file_id, device=x.device
        )
        system_ids = [int(v) for v in system_ids_tensor.view(-1).tolist()]

        if task_id in ["classification"]:
            return self.task_head(
                x, system_id=system_ids, return_feature=return_feature, task_id=task_id
            )
        elif task_id in ["prediction"]:
            # 对预测任务，这里可以根据需要扩展；当前简单返回 backbone 输出
            return x

    def forward(self, x, file_id=False, task_id=False, return_feature=False):
        """
        Forward pass.

        Args
        ----
        x : Tensor
            输入序列 [B, L, C]。
        file_id : Any
            用于在 metadata 中查找 Dataset_id / Sample_rate。
        task_id : str
            'classification' 或 'prediction' 等。
        return_feature : bool
            若 True，返回 (logits, features)。
        """
        self.shape = x.shape
        x, c = self._embed(x, file_id)
        features = self._encode(x, c)
        logits = self._head(features, file_id, task_id, return_feature=False)

        if return_feature:
            return logits, features
        return logits

