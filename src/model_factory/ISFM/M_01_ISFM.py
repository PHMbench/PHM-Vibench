if __name__ == '__main__':
    
    # 在 M_01_ISFM.py 文件开头添加
    import sys
    import os

    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 计算项目根目录路径（假设文件在 src/models/ 下）
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.append(project_root)
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    
from src.models.embedding import *
from src.models.backbone import *
from src.models.task_head import *
import torch.nn as nn

Embedding_dict = {
    'E_01_HTFE': E_01_HTFE,
    'E_01_HSE': E_01_HSE,
    
    
    'E_com_00_PE':E_com_00_PE,
    'E_01_HSE_abalation':E_01_HSE_abalation,
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
}
class M_01_ISFM(nn.Module):
    def __init__(self, args_m,args_d = False): # args_d = False when not using H_02_distance_cla
        super(M_01_ISFM, self).__init__()
        self.embedding = Embedding_dict[args_m.embedding](args_m,args_d)
        self.backbone = Backbone_dict[args_m.backbone](args_m)
        self.task_head = TaskHead_dict[args_m.task_head](args_m,args_d)
        
        if args_m.task_head == 'H_02_distance_cla':
            self.category_tokes = {}
            for key in args_d.task.keys():
                self.category_tokes[key] = nn.Parameter(torch.randn(args_m.n_classes, args_m.output_dim))
            self.category_token = nn.Parameter(torch.randn(args_m.n_classes, args_m.output_dim))
        
    def forward(self, x,data_name = False,task_name = False):
        x = self.embedding(x,data_name)
        x = self.backbone(x)
        
        # TODO multiple task head 判断 data
        x = self.task_head(x,data_name,task_name)
        return x
    
if __name__ == '__main__':
    
    # 在 M_01_ISFM.py 文件开头添加
    import sys
    import os

    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 计算项目根目录路径（假设文件在 src/models/ 下）
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.append(project_root)
    
    from utils.config_utils import *
    import torch
    config_path = 'configs/basic.yaml'
    args = load_config(config_path)
    args_m = transfer_namespace(args['model'])
    args_d = transfer_namespace(args['dataset'])    
    model = M_01_ISFM(args_m,args_d)
    print(model)
    x = torch.randn(2, 1280, 3)
    y = model(x)
    print(y.shape)