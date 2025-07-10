if __name__ == '__main__':
    
    # 在 M_01_ISFM.py 文件开头添加
    import sys
    import os

    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 计算项目根目录路径（假设文件在 src/models/ 下）
    project_root = os.path.abspath(os.path.join(current_dir, "..","..", ".."))
    sys.path.append(project_root)
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# from .embedding import *
# from .backbone import *
# from .task_head import *
from src.model_factory.ISFM.embedding import *
from src.model_factory.ISFM.backbone import *
from src.model_factory.ISFM.task_head import *
import torch.nn as nn
import numpy as np
import os
import torch

Embedding_dict = {

    'E_01_HSE': E_01_HSE,
    'E_02_HSE_v2': E_02_HSE_v2,  # Updated to use the new HSE class

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
    """ISFM model without dataset-specific label counts."""

    def __init__(self, args_m, metadata):
        super(Model, self).__init__()
        self.metadata = metadata
        self.args_m = args_m
        self.embedding = Embedding_dict[args_m.embedding](args_m)
        self.backbone = Backbone_dict[args_m.backbone](args_m)
        # self.num_classes = self.get_num_classes()  # TODO prediction 任务不需要label？ @liq22
        # args_m.num_classes = self.num_classes  # Ensure num_classes is set in args_m
        self.task_head = TaskHead_dict[args_m.task_head](args_m)

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
        """Run the model on a batch."""
        self.shape = x.shape
        x = self._embed(x, file_id)
        x = self._encode(x)
        x = self._head(x, file_id, task_id, return_feature)
        return x
    
if __name__ == '__main__':
    import sys
    import os

    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 计算项目根目录路径
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    sys.path.append(project_root)
    
    from utils.config_utils import *
    import torch
    
    # 使用指定的配置文件
    # config_path = os.path.join(project_root, 'configs/demo/Multiple_DG/CWRU_THU_using_ISFM.yaml')
    config_path = os.path.join(project_root, 'script/LQ1/Pretraining/Pretraining_C+P.yaml')
    # config_path = os.path.join(project_root, 'script/LQ1/GFS/GFS_C+M.yaml')
    print(f"加载配置文件: {config_path}")
    
    try:
        configs = load_config(config_path)
# 设置环境变量和命名空间
        args_environment = transfer_namespace(configs.get('environment', {}))
        args_data = transfer_namespace(configs.get('data', {}))
        args_model = transfer_namespace(configs.get('model', {}))
        args_task = transfer_namespace(configs.get('task', {}))
        args_trainer = transfer_namespace(configs.get('trainer', {}))
        args_model.num_classes = {'CWRU':2, 'THU': 3}  # 示例，实际应从配置中获取
        
        print("模型配置:", args_model)
        print("数据集配置:", args_data)
        
        class MockMetadata:
            def __getitem__(self, idx):
                return {
                    'Sample_rate': 16000,
                    'Dataset_id': 'CWRU' if idx % 2 == 0 else 'THU',
                    'Name': f"Sample_{idx}"
                }
        
        metadata = MockMetadata()
        
        # 初始化模型
        model = Model(args_model, metadata)
        print(model)
        
        # 创建随机输入进行测试
        batch_size = 2
        seq_len = 128
        feature_dim = 3
        x = torch.randn(batch_size, seq_len, feature_dim)

        # 运行前向传播
        # y = model(x, file_id=0, task_id='classification', return_feature=False)
        y = model(x, file_id=0, task_id='prediction', return_feature=True)
        print("输出形状:", y.shape)
        
    except Exception as e:
        import traceback
        print(f"错误: {e}")
        traceback.print_exc()
