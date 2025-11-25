# from .backbone import *
# from .task_head import *
from src.model_factory.ISFM.embedding import *
# from src.model_factory.ISFM.embedding import E_03_Patch_DPOT
from src.model_factory.ISFM.backbone import *
from src.model_factory.ISFM.task_head import *
from src.model_factory.ISFM.system_utils import resolve_batch_metadata
import torch.nn as nn
import numpy as np
import os
import torch
import pandas as pd
from src.utils.utils import get_num_classes, get_num_channels
# from src.model_factory.ISFM.layers.StandardNorm import Normalize
Embedding_dict = {

    'E_01_HSE': E_01_HSE,
    'E_02_HSE_v2': E_02_HSE_v2,  # Reconstruction-focused HSE variant

}
Backbone_dict = {
    'B_01_basic_transformer': B_01_basic_transformer,
    'B_03_FITS': B_03_FITS,
    'B_04_Dlinear': B_04_Dlinear,
    'B_05_Manba': B_05_Manba,
    'B_06_TimesNet': B_06_TimesNet,
    'B_07_TSMixer': B_07_TSMixer,
    'B_08_PatchTST': B_08_PatchTST,
    'B_09_FNO': B_09_FNO,
    'B_10_VIBT': B_10_VIBT,  # Vibration Transformer Backbone
    
}
TaskHead_dict = {
    'H_01_Linear_cla': H_01_Linear_cla,
    'H_02_distance_cla': H_02_distance_cla,
    'H_03_Linear_pred': H_03_Linear_pred,
    'H_09_multiple_task': H_09_multiple_task, 
    'H_04_VIB_pred': H_04_VIB_pred
}

class Model(nn.Module):
    """ISFM variant with channel-aware embedding.

    Parameters
    ----------
    args_m : Namespace
        Configuration including embedding, backbone and head choices.
    metadata : Any
        Metadata accessor used for channel and label counts.

    Notes
    -----
    Expects inputs shaped ``(B, L, C)``.
    """

    def __init__(self, args_m, metadata):
        super(Model, self).__init__()
        self.metadata = metadata
        self.args_m = args_m

        self.num_classes = self.get_num_classes()  # TODO prediction 任务不需要label？ @liq22
        args_m.num_classes = self.num_classes  # Ensure num_classes is set in args_m
        args_m.num_channels = self.get_num_channels()  # Ensure num_channels is set in args_m

        self.embedding = Embedding_dict[args_m.embedding](args_m)
        self.backbone = Backbone_dict[args_m.backbone](args_m)
        if args_m.task_head == 'H_04_VIB_pred' and args_m.embedding == 'E_02_HSE_v2':
            # If using multiple task head, we need to pass the metadata
            self.task_head = TaskHead_dict[args_m.task_head](args_m, self.embedding.patcher)
        else:
            self.task_head = TaskHead_dict[args_m.task_head](args_m)


    def get_num_classes(self):
        """获取数据集类别数映射"""
        return get_num_classes(self.metadata)

    def get_num_channels(self):
        """获取数据集通道数映射"""
        return get_num_channels(self.metadata)

    def _embed(self, x, file_id):
        """1 Embedding：支持 per-sample system_id / Sample_rate。"""
        system_ids_tensor, sample_f_tensor = resolve_batch_metadata(self.metadata, file_id, device=x.device)

        if self.args_m.embedding == 'E_01_HSE':
            # HSE v1 仅依赖 fs
            x = self.embedding(x, sample_f_tensor)
            c = None
        elif self.args_m.embedding == 'E_02_HSE_v2':
            # HSE v2 使用 system_id + sample_f
            x, c = self.embedding(x, system_ids_tensor, sample_f_tensor)
        else:
            x = self.embedding(x)
            c = None

        return x, c

    def _encode(self, x,c=False):
        """2 Backbone"""
        return self.backbone(x,c)

    def _head(self, x, file_id = False, task_id = False, return_feature=False):
        """3 Task Head：按样本传递 system_id 给多头分类器/预测头。"""
        B = x.size(0)
        system_ids_tensor, _ = resolve_batch_metadata(self.metadata, file_id, device=x.device)
        system_ids = [int(v) for v in system_ids_tensor.view(-1).tolist()]

        if task_id in ['classification']:
            return self.task_head(x, system_id=system_ids, return_feature=return_feature, task_id=task_id)

        elif task_id in ['prediction']:
            if self.args_m.task_head == 'H_04_VIB_pred':
                return self.task_head(x, system_id=system_ids)
            else:
                shape = (self.shape[1], self.shape[2]) if len(self.shape) > 2 else (self.shape[1],)
                return self.task_head(x, shape=shape, system_id=system_ids, return_feature=return_feature, task_id=task_id)

    def forward(self, x, file_id=False, task_id=False, return_feature=False):
        """Forward pass.

        Args:
            x: 输入序列 ``(B, L, C)``。
            file_id: 样本索引。
            task_id: 任务标识。
            return_feature: 是否返回特征表示。

        Returns:
            模型输出张量。
        """
        self.shape = x.shape
        x, c = self._embed(x, file_id)
        x = self._encode(x, c)
        x = self._head(x, file_id, task_id, return_feature)
        return x

    def get_rep(self, x, file_id = False):
        x,c = self._embed(x, file_id)
        x = self._encode(x,c)
        return x
