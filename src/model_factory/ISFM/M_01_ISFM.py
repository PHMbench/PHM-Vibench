from src.model_factory.ISFM.embedding import *
from src.model_factory.ISFM.embedding import E_03_Patch
from src.model_factory.ISFM.backbone import *
from src.model_factory.ISFM.task_head import *
import torch.nn as nn
import numpy as np
import os
import torch
from src.utils.utils import get_num_classes

Embedding_dict = {
    'E_01_HSE': E_01_HSE,
    # 'E_01_HSE_Prompt': E_01_HSE_Prompt,  # Prompt-guided HSE for contrastive learning
    'E_02_HSE_v2': E_02_HSE_v2,  # Updated to use the new HSE class
    # Patch-based embedding as basic baseline (P1)
    'E_03_Patch': E_03_Patch,
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
    'MultiTaskHead': MultiTaskHead, # Add the enhanced multi-task head
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
        """获取数据集类别数映射"""
        return get_num_classes(self.metadata)
    


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
        system_id = str(self.metadata[file_id]['Dataset_id'])  # Convert to string
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

    def forward(self, x, file_id=False, task_id=False, return_feature=False, return_prompt=False):
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
        return_prompt : bool, optional
            If ``True`` return prompts alongside output for contrastive learning.

        Returns
        -------
        torch.Tensor or tuple
            Model output defined by the task head, optionally with prompts and features.
        """
        # Input validation to prevent None tensor errors
        if x is None:
            raise ValueError("Input tensor x cannot be None")

        self.shape = x.shape
        x = self._embed(x, file_id)
        x = self._encode(x)

        # Extract prompt information if available and requested
        prompts = None
        if return_prompt and hasattr(self.embedding, 'last_prompt_vector'):
            prompts = self.embedding.last_prompt_vector

        # Get task head output
        head_output = self._head(x, file_id, task_id, return_feature)

        # Handle different return combinations for contrastive learning
        if return_prompt and return_feature:
            # Return logits, prompts, and features
            if isinstance(head_output, tuple):
                # _head already returns features
                features = head_output[0] if return_feature else head_output
            else:
                features = head_output
            return head_output, prompts, features
        elif return_prompt:
            # Return logits and prompts
            return head_output, prompts
        elif return_feature:
            # Return features (already handled by _head)
            return head_output
        else:
            # Standard forward - return logits only
            return head_output
    
