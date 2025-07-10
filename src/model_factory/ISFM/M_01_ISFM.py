# from .backbone import *
# from .task_head import *
from src.model_factory.ISFM.embedding import *
from src.model_factory.ISFM.embedding import E_03_Patch_DPOT
from src.model_factory.ISFM.backbone import *
from src.model_factory.ISFM.task_head import *
import torch.nn as nn
import numpy as np
import os
import torch

Embedding_dict = {

    'E_01_HSE': E_01_HSE,
    'E_02_HSE_v2': E_02_HSE_v2,  # Updated to use the new HSE class
    'E_03_Patch_DPOT': E_03_Patch_DPOT,

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
    
}
TaskHead_dict = {
    'H_01_Linear_cla': H_01_Linear_cla,
    'H_02_distance_cla': H_02_distance_cla,
    'H_03_Linear_pred': H_03_Linear_pred,
    'H_09_multiple_task': H_09_multiple_task, # Add the new multiple task head
}


class Model(nn.Module):
    """ISFM architecture with flexible embedding/backbone/head.

    Parameters
    ----------
    args_m : Namespace
        Defines ``embedding``, ``backbone`` and ``task_head`` as well as
        ``num_classes``.
    metadata : Any
        Metadata accessor providing dataset information.

    Notes
    -----
    Input tensors are expected with shape ``(B, L, C)`` and outputs depend on
    the selected task head.
    """

    def __init__(self, args_m, metadata):
        super(Model, self).__init__()
        self.metadata = metadata
        self.args_m = args_m
        self.embedding = Embedding_dict[args_m.embedding](args_m)
        self.backbone = Backbone_dict[args_m.backbone](args_m)
        self.num_classes = self.get_num_classes()  # TODO prediction 任务不需要label？ @liq22
        args_m.num_classes = self.num_classes  # Ensure num_classes is set in args_m
        self.task_head = TaskHead_dict[args_m.task_head](args_m)

    def get_num_classes(self):
        num_classes = {}
        for key in np.unique(self.metadata.df['Dataset_id']):
            num_classes[key] = max(self.metadata.df[self.metadata.df['Dataset_id'] == key]['Label']) + 1
        return num_classes
    


    def _embed(self, x, file_id):
        """1 Embedding"""
        if self.args_m.embedding in ('E_01_HSE', 'E_02_HSE_v2'):
            fs = self.metadata[file_id]['Sample_rate']
            # system_id = self.metadata[file_id]['Dataset_id']
            x = self.embedding(x, fs)
        else:
            x = self.embedding(x)
        return x

    def _encode(self, x):
        """2 Backbone"""
        return self.backbone(x)

    def _head(self, x, file_id = False, task_id = False, return_feature=False):
        """3 Task Head"""
        system_id = self.metadata[file_id]['Dataset_id']
        # check if task_id is in the task head
        # check if task have its head

        if task_id in ['classification']:
            # For classification or prediction tasks, we need to pass system_id
            return self.task_head(x, system_id=system_id, return_feature=return_feature, task_id=task_id)
        elif task_id in ['prediction']: # TODO individual prediction head
            shape = (self.shape[1], self.shape[2]) if len(self.shape) > 2 else (self.shape[1],)
            # For prediction tasks, we may not need system_id
            return self.task_head(x, return_feature=return_feature, task_id=task_id, shape=shape)
        # if task_id in ['classification', 'prediction']:
        #     # For classification or prediction tasks, we need to pass system_id
        #     return self.task_head(x, system_id=system_id, return_feature=return_feature)
        # elif task_id in ['multitask']:
        # return self.task_head(x, system_id=system_id, return_feature=return_feature, task_id=task_id)

    def forward(self, x, file_id=False, task_id=False, return_feature=False):
        """Forward pass through embedding, backbone and head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor ``(B, L, C)``.
        file_id : Any, optional
            Key used to fetch metadata for the sample.
        task_id : str, optional
            Task type such as ``"classification"`` or ``"prediction"``.
        return_feature : bool, optional
            If ``True`` return features instead of logits.

        Returns
        -------
        torch.Tensor
            Model output defined by the task head.
        """
        self.shape = x.shape
        x = self._embed(x, file_id)
        x = self._encode(x)
        x = self._head(x, file_id, task_id, return_feature)
        return x
    

