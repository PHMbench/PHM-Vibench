# from .backbone import *
# from .task_head import *
from src.model_factory.ISFM.embedding import *
from src.model_factory.ISFM.embedding import E_03_Patch_DPOT
from src.model_factory.ISFM.backbone import *
from src.model_factory.ISFM.task_head import *
from src.model_factory.ISFM.task_head.H_05_RUL_pred import H_05_RUL_pred
from src.model_factory.ISFM.task_head.H_06_Anomaly_det import H_06_Anomaly_det
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
    'H_05_RUL_pred': H_05_RUL_pred,     # RUL prediction head
    'H_06_Anomaly_det': H_06_Anomaly_det, # Anomaly detection head
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
        
        # Initialize task head management system
        self._init_task_heads(args_m)

    def get_num_classes(self):
        num_classes = {}
        for key in np.unique(self.metadata.df['Dataset_id']):
            # Filter out NaN and -1 values (following existing pattern from Get_id.py)
            # -1 typically indicates samples that don't participate in classification training
            labels = self.metadata.df[self.metadata.df['Dataset_id'] == key]['Label']
            valid_labels = labels[labels.notna() & (labels >= 0)]
            
            if len(valid_labels) > 0:
                # Use valid labels to calculate class count
                num_classes[key] = int(valid_labels.max()) + 1
            else:
                # Default to binary classification if no valid labels exist
                # This handles edge cases where entire datasets have only -1/NaN labels
                num_classes[key] = 2
                
        return num_classes
    
    # Task Head Management Functions (Decoupled)
    
    def _create_task_head(self, task_name, args_m):
        """创建单个task head"""
        head_class_name = self.task_head_mapping.get(task_name, getattr(args_m, 'task_head', 'H_01_Linear_cla'))
        if head_class_name in TaskHead_dict:
            return TaskHead_dict[head_class_name](args_m)
        raise ValueError(f"Unknown task head: {head_class_name} for task: {task_name}")
    
    def _get_or_create_task_head(self, task_name):
        """获取或动态创建task head"""
        # 如果已存在，直接返回
        if task_name in self.task_heads:
            return self.task_heads[task_name]
        
        # 动态创建
        head_class_name = self.task_head_mapping.get(task_name)
        if head_class_name and head_class_name in TaskHead_dict:
            head = TaskHead_dict[head_class_name](self.args_m)
            self.task_heads[task_name] = head
            print(f"[ISFM] Dynamically created {head_class_name} for task '{task_name}'")
            return head
        
        # 使用fallback
        if self._single_task_head:
            print(f"[WARNING] Unknown task '{task_name}', using single task head as fallback")
            return self._single_task_head
        
        raise ValueError(f"No head available for task: {task_name}")
    
    def _prepare_task_params(self, task_name, file_id, return_feature):
        """准备任务特定的参数。支持单个file_id或file_ids列表。"""
        params = {'return_feature': return_feature}
        
        if task_name == 'classification':
            # Support both single file_id and batch file_ids
            if isinstance(file_id, (list, tuple)):
                # Batch processing: extract system_id for each file_id
                system_ids = []
                for fid in file_id:
                    if fid in self.metadata and 'Dataset_id' in self.metadata[fid]:
                        system_ids.append(self.metadata[fid]['Dataset_id'])
                    elif fid in self.metadata and 'System_id' in self.metadata[fid]:
                        system_ids.append(self.metadata[fid]['System_id'])
                    else:
                        fallback_id = fid if isinstance(fid, int) else 0
                        system_ids.append(fallback_id)
                        self._warn_missing_dataset_id(fid, fallback_id)
                params['system_id'] = system_ids
            else:
                # Single file_id: original logic
                if file_id in self.metadata and 'Dataset_id' in self.metadata[file_id]:
                    params['system_id'] = self.metadata[file_id]['Dataset_id']
                elif file_id in self.metadata and 'System_id' in self.metadata[file_id]:
                    params['system_id'] = self.metadata[file_id]['System_id']
                else:
                    fallback_id = file_id if isinstance(file_id, int) else 0
                    params['system_id'] = fallback_id
                    self._warn_missing_dataset_id(file_id, fallback_id)
                    
        elif task_name in ['signal_prediction', 'prediction']:  # Support legacy 'prediction' name
            params['shape'] = (self.shape[1], self.shape[2]) if len(self.shape) > 2 else (self.shape[1],)
        # rul_prediction和anomaly_detection只需要return_feature
        
        return params
    
    def _warn_missing_dataset_id(self, file_id, fallback_id):
        """为缺失的Dataset_id记录警告信息（去重）。"""
        if not hasattr(self, '_dataset_id_warnings'):
            self._dataset_id_warnings = set()
        if file_id not in self._dataset_id_warnings:
            print(f"Warning: Missing Dataset_id for file {file_id}, using fallback system_id={fallback_id}")
            self._dataset_id_warnings.add(file_id)
    
    def _execute_single_task(self, x, task_name, file_id, return_feature):
        """执行单个任务。支持单个file_id或batch file_ids。"""
        head = self._get_or_create_task_head(task_name)
        params = self._prepare_task_params(task_name, file_id, return_feature)
        return head(x, **params)
    
    def _init_task_heads(self, args_m):
        """初始化task head管理系统"""
        self.task_heads = nn.ModuleDict()
        self._single_task_head = None
        
        self.task_head_mapping = {
            'classification': 'H_01_Linear_cla',
            'rul_prediction': 'H_05_RUL_pred',
            'anomaly_detection': 'H_06_Anomaly_det',
            'signal_prediction': 'H_03_Linear_pred'
        }
        
        enabled_tasks = getattr(args_m, 'enabled_tasks', None)
        
        if enabled_tasks and len(enabled_tasks) > 1:
            # 多任务模式：预加载所有enabled tasks的heads
            for task in enabled_tasks:
                try:
                    head = self._create_task_head(task, args_m)
                    self.task_heads[task] = head
                except ValueError as e:
                    print(f"[WARNING] Failed to create head for task '{task}': {e}")
            print(f"[ISFM] Multi-task mode: Loaded {len(self.task_heads)} task heads for tasks: {list(self.task_heads.keys())}")
        else:
            # 单任务模式：只加载fallback head
            task_head_name = getattr(args_m, 'task_head', 'H_01_Linear_cla')
            if task_head_name in TaskHead_dict:
                self._single_task_head = TaskHead_dict[task_head_name](args_m)
                print(f"[ISFM] Single-task mode: Loaded {task_head_name}")
            else:
                raise ValueError(f"Unknown task head: {task_head_name}")


    def _embed(self, x, file_id):
        """1 Embedding - supports both single file_id and batch file_ids"""
        if self.args_m.embedding in ('E_01_HSE', 'E_02_HSE_v2'):
            # Handle both single file_id and batch file_ids
            if isinstance(file_id, (list, tuple)):
                # For batch processing, use the first file_id for embedding parameters
                # Since embedding typically uses dataset-level parameters (sample rate)
                # which should be consistent within a batch for most use cases
                primary_file_id = file_id[0]
            else:
                primary_file_id = file_id
                
            fs = self.metadata[primary_file_id]['Sample_rate']
            x = self.embedding(x, fs)
        else:
            x = self.embedding(x)
        return x

    def _encode(self, x):
        """2 Backbone"""
        return self.backbone(x)

    def _head(self, x, file_id=False, task_id=False, return_feature=False):
        """3 Task Head - 简化后的任务头调度器"""
        if file_id is False:
            raise ValueError("file_id must be provided for task head")
        
        # 记住原始task_id类型
        original_task_id = task_id
        
        # 统一处理为列表
        task_list = [task_id] if isinstance(task_id, str) else task_id
        
        # 执行任务
        results = {}
        for task in task_list:
            results[task] = self._execute_single_task(x, task, file_id, return_feature)
        
        # 返回结果 - 保持一致性：list输入总是返回dict，string输入返回单值
        if isinstance(original_task_id, list):
            return results  # Always return dict for list input (multi-task consistency)
        else:
            return list(results.values())[0] if len(results) == 1 else results


    def forward(self, x, file_id=False, task_id=False, return_feature=False):
        """Forward pass through embedding, backbone and head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor ``(B, L, C)``.
        file_id : Any, optional
            Key used to fetch metadata for the sample.
        task_id : str or list, optional
            Task type(s) such as ``"classification"`` or list of tasks.
        return_feature : bool, optional
            If ``True`` return features instead of logits.

        Returns
        -------
        torch.Tensor or Dict[str, torch.Tensor]
            Model output defined by the task head(s).
        """
        self.shape = x.shape
        x = self._embed(x, file_id)
        x = self._encode(x)
        x = self._head(x, file_id, task_id, return_feature)
        return x
    


if __name__ == '__main__':
    """Unit tests for M_01_ISFM module."""
    import pandas as pd
    from argparse import Namespace
    import torch
    
    print("=== Testing M_01_ISFM Module ===")
    
    # Create mock metadata
    class MockMetadata:
        def __init__(self):
            # Simulate metadata with dataset IDs as numpy integers
            import numpy as np
            self.df = pd.DataFrame({
                'Dataset_id': [np.int64(1), np.int64(2), np.int64(5), np.int64(6)],
                'Label': [0, 1, 2, 3],
                'Sample_rate': [12000, 12000, 12000, 12000]
            })
        
        def __getitem__(self, file_id):
            # Mock metadata access by file_id
            return {
                'Dataset_id': np.int64(1),
                'Sample_rate': 12000
            }
    
    # Test 1: Single-task mode
    print("\n--- Test 1: Single-task Mode ---")
    args_single = Namespace(
        embedding='E_01_HSE',
        backbone='B_04_Dlinear',
        task_head='H_01_Linear_cla',
        output_dim=512,
        d_model=128,
        num_layers=3,
        dropout=0.1,
        num_patches=64,      # Required for B_04_Dlinear (patch_size_L)
        patch_size_L=256,    # Required for some task heads
        patch_size_C=1       # Required for patch-based models
    )
    
    try:
        metadata = MockMetadata()
        model = Model(args_single, metadata)
        print(f"✅ Single-task model created successfully")
        print(f"   Task heads loaded: {list(model.task_heads.keys())}")
        print(f"   Single task head: {model._single_task_head.__class__.__name__ if model._single_task_head else None}")
        
        # Test forward pass
        batch_size, seq_len, channels = 2, 1024, 1
        x = torch.randn(batch_size, seq_len, channels)
        file_id = 0
        
        output = model(x, file_id=file_id, task_id='classification')
        print(f"   Forward pass output shape: {output.shape}")
        
    except Exception as e:
        print(f"❌ Single-task test failed: {e}")
    
    # Test 2: Multi-task mode
    print("\n--- Test 2: Multi-task Mode ---")
    args_multi = Namespace(
        embedding='E_01_HSE',
        backbone='B_04_Dlinear',
        task_head='MultiTaskHead',
        enabled_tasks=['classification', 'rul_prediction', 'anomaly_detection', 'signal_prediction'],
        output_dim=512,
        d_model=128,
        num_layers=3,
        dropout=0.1,
        num_patches=64,      # Required for B_04_Dlinear (patch_size_L)
        patch_size_L=256,    # Required for some task heads
        patch_size_C=1       # Required for patch-based models
    )
    
    try:
        metadata = MockMetadata()
        model = Model(args_multi, metadata)
        print(f"✅ Multi-task model created successfully")
        print(f"   Task heads loaded: {list(model.task_heads.keys())}")
        print(f"   Number of task heads: {len(model.task_heads)}")
        
        # Test forward pass with multiple tasks
        x = torch.randn(batch_size, seq_len, channels)
        file_id = 0
        
        # Test with single task
        output_single = model(x, file_id=file_id, task_id='classification')
        print(f"   Single task output shape: {output_single.shape}")
        
        # Test with multiple tasks
        output_multi = model(x, file_id=file_id, task_id=['classification', 'rul_prediction'])
        print(f"   Multi-task output type: {type(output_multi)}")
        if isinstance(output_multi, dict):
            for task, output in output_multi.items():
                print(f"     {task}: {output.shape}")
        
    except Exception as e:
        print(f"❌ Multi-task test failed: {e}")
    
    # Test 3: Task head mapping consistency
    print("\n--- Test 3: Task Head Mapping Consistency ---")
    try:
        metadata = MockMetadata()
        model = Model(args_multi, metadata)
        
        # Check all mappings are valid
        for task, head_name in model.task_head_mapping.items():
            if head_name in TaskHead_dict:
                print(f"   ✅ {task} -> {head_name} (valid)")
            else:
                print(f"   ❌ {task} -> {head_name} (invalid - not in TaskHead_dict)")
        
        # Check loaded heads match mapping
        for task in model.task_heads.keys():
            if task in model.task_head_mapping:
                expected_head = model.task_head_mapping[task]
                actual_head = model.task_heads[task].__class__.__name__
                print(f"   {task}: expected={expected_head}, actual={actual_head}")
        
    except Exception as e:
        print(f"❌ Task head mapping test failed: {e}")
    
    # Test 4: Dynamic task head creation
    print("\n--- Test 4: Dynamic Task Head Creation ---")
    try:
        metadata = MockMetadata()
        # Create model with single task first
        model = Model(args_single, metadata)
        
        # Test dynamic creation through full forward pass
        x = torch.randn(2, 1024, 1)  # Raw input tensor
        file_id = 0
        
        # This should dynamically create the head if not already loaded
        output = model(x, file_id=file_id, task_id=['anomaly_detection'])
        print(f"   ✅ Dynamic task head creation successful")
        print(f"   Current task heads: {list(model.task_heads.keys())}")
        
    except Exception as e:
        print(f"❌ Dynamic task head creation test failed: {e}")
    
    print("\n=== M_01_ISFM Tests Complete ===")