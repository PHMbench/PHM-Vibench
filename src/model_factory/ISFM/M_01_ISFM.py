if __name__ == '__main__':
    
    # 在 M_01_ISFM.py 文件开头添加
    import sys


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

}
Backbone_dict = {
    'B_01_basic_transformer': B_01_basic_transformer,
    
    
    'B_03_FITS': B_03_FITS,
    'B_04_Dlinear': B_04_Dlinear,
    'B_05_Manba': B_05_Manba,
    
}
TaskHead_dict = {
    'H_01_Linear_cla': H_01_Linear_cla,
    'H_02_distance_cla': H_02_distance_cla,
    'H_03_Linear_pred': H_03_Linear_pred,
}


class Model(nn.Module):
    def __init__(self, args_m,metadata): # args_d = False when not using H_02_distance_cla
        super(Model, self).__init__()
        self.metadata = metadata
        self.args_m = args_m
        self.embedding = Embedding_dict[args_m.embedding](args_m)
        self.backbone = Backbone_dict[args_m.backbone](args_m)
        self.num_classes = self.get_num_classes()
        self.task_head = TaskHead_dict[args_m.task_head](args_m, self.num_classes)
    
            
    def get_num_classes(self):
        num_classes = {}
        for key in np.unique(self.metadata.df['Dataset_id']):
            num_classes[key] = max(self.metadata.df[self.metadata.df['Dataset_id'] == key]['Label']) + 1
        return num_classes
    
    def forward(self, x, File_id = False,Task_id = False):

        
        if self.args_m.embedding == 'E_01_HSE':
            fs = self.metadata[File_id]['Sample_rate']
            x = self.embedding(x,fs)
        else:
            x = self.embedding(x)
        x = self.backbone(x)
        
        # TODO multiple task head 判断 data
        System_id = self.metadata[File_id]['Dataset_id']
        x = self.task_head(x,System_id,Task_id, return_feature=False)
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
    config_path = os.path.join(project_root, 'configs/demo/Multiple_DG/CWRU_THU_using_ISFM.yaml')
    print(f"加载配置文件: {config_path}")
    
    try:
        configs = load_config(config_path)
# 设置环境变量和命名空间
        args_environment = transfer_namespace(configs.get('environment', {}))

        args_data = transfer_namespace(configs.get('data', {}))

        args_model = transfer_namespace(configs.get('model', {}).get('args', {}))
        args_model.name = configs['model'].get('name', 'default')

        args_task = transfer_namespace(configs.get('task', {}).get('args', {}))
        args_task.name = configs['task'].get('name', 'default')

        args_trainer = transfer_namespace(configs.get('trainer', {}).get('args', {}))
        args_trainer.name = configs['trainer'].get('name', 'default')
        
        print("模型配置:", args_model)
        print("数据集配置:", args_data)
        
        # 创建模拟数据以便测试
        class MockMetadata:
            def __init__(self):
                import pandas as pd
                self.df = pd.DataFrame({
                    'Dataset_id': [0, 0, 1, 1],
                    'Label': [0, 1, 0, 2]
                })
            
            def __getitem__(self, key):
                return {'Sample_rate': 1000}
        
        metadata = MockMetadata()
        
        # 初始化模型
        model = Model(args_model, metadata)
        print(model)
        
        # 创建随机输入进行测试
        batch_size = 2
        seq_len = 128
        feature_dim = 3
        x = torch.randn(batch_size, seq_len, feature_dim)
        id = 0
        # 运行前向传播
        y = model(x,id)
        print("输出形状:", y.shape)
        
    except Exception as e:
        import traceback
        print(f"错误: {e}")
        traceback.print_exc()