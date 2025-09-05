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
        
        # Initialize task head(s) - support both single and multi-task modes
        self.task_heads = nn.ModuleDict()
        self._single_task_head = None
        
        # Available task head mapping for dynamic creation
        self.task_head_mapping = {
            'classification': 'H_01_Linear_cla',
            'rul_prediction': 'H_05_RUL_pred',
            'anomaly_detection': 'H_06_Anomaly_det', 
            'signal_prediction': 'H_03_Linear_pred'
        }
        
        # Check if this is a multi-task configuration
        enabled_tasks = getattr(args_m, 'enabled_tasks', None)
        
        if enabled_tasks and len(enabled_tasks) > 1:
            # Multi-task mode: preload individual heads for enabled tasks
            for task in enabled_tasks:
                head_name = self.task_head_mapping.get(task, args_m.task_head)
                if head_name in TaskHead_dict:
                    self.task_heads[task] = TaskHead_dict[head_name](args_m)
                else:
                    raise ValueError(f"Unknown task head: {head_name} for task: {task}")
                    
            print(f"[ISFM] Multi-task mode: Loaded {len(self.task_heads)} task heads for tasks: {list(self.task_heads.keys())}")
        else:
            # Single task mode or dynamic multi-task: use specified head as fallback
            task_head_name = getattr(args_m, 'task_head', 'H_01_Linear_cla')
            if task_head_name in TaskHead_dict:
                self._single_task_head = TaskHead_dict[task_head_name](args_m)
                print(f"[ISFM] Single-task mode: Loaded {task_head_name}")
            else:
                raise ValueError(f"Unknown task head: {task_head_name}")

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

    def _head(self, x, file_id=False, task_id=False, return_feature=False):
        """3 Task Head - Orchestrates individual task heads"""
        if file_id is False:
            raise ValueError("file_id must be provided for task head")
            
        system_id = self.metadata[file_id]['Dataset_id']
        shape = (self.shape[1], self.shape[2]) if len(self.shape) > 2 else (self.shape[1],)
        
        # Handle task_id as string or list
        if isinstance(task_id, str):
            task_id = [task_id]
        
        # Check if we need multiple task heads (multi-task scenario)
        if len(task_id) > 1 or any(task in self.task_heads for task in task_id):
            # Multi-task mode: use/create individual heads
            results = {}
            
            for task in task_id:
                # Get or create task head
                if task in self.task_heads:
                    head = self.task_heads[task]
                elif task in self.task_head_mapping:
                    # Dynamically create head if not already loaded
                    head_name = self.task_head_mapping[task]
                    head = TaskHead_dict[head_name](self.args_m)
                    self.task_heads[task] = head
                    print(f"[ISFM] Dynamically created {head_name} for task '{task}'")
                else:
                    print(f"[WARNING] Unknown task '{task}', using single task head as fallback")
                    head = self._single_task_head
                    if head is None:
                        raise ValueError(f"No fallback head available for unknown task: {task}")
                
                # Call appropriate head with task-specific parameters
                if task == 'classification':
                    results[task] = head(x, system_id=system_id, return_feature=return_feature)
                elif task == 'signal_prediction':
                    results[task] = head(x, return_feature=return_feature, shape=shape)
                elif task in ['rul_prediction', 'anomaly_detection']:
                    results[task] = head(x, return_feature=return_feature)
                else:
                    # Generic fallback
                    results[task] = head(x, return_feature=return_feature)
            
            # Return single result if only one task, otherwise return dict
            if len(results) == 1:
                return list(results.values())[0]
            else:
                return results
        else:
            # Single task mode: use the single task head
            task = task_id[0]
            head = self._single_task_head
            
            if head is None:
                raise ValueError("No single task head available")
                
            if task == 'classification':
                return head(x, system_id=system_id, return_feature=return_feature)
            elif task == 'prediction' or task == 'signal_prediction':
                return head(x, return_feature=return_feature, shape=shape)
            else:
                # Generic fallback
                return head(x, system_id=system_id, return_feature=return_feature)


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