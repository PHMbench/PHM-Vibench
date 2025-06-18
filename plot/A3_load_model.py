from ast import arg
import sys
sys.path.append('../')

import torch
import pandas as pd
import numpy as np
from pytorch_lightning import seed_everything
from configs.config import parse_arguments, config_network,yaml_arguments

from model.TSPN import Transparent_Signal_Processing_Network
from model.TSPN_KAN import Transparent_Signal_Processing_KAN
from model.NNSPN import NN_Signal_Processing_Network
from model.TFON import Time_Frequency_Operator_Network


from model_collection.Resnet import ResNet, BasicBlock
from model_collection.Sincnet import Sincnet,Sinc_net_m
from model_collection.WKN import WKN,WKN_m
from model_collection.EELM import Dong_ELM
from model_collection.MWA_CNN import A_cSE,Huan_net
from model_collection.TFN.Models.TFN import TFN_Morlet
from model_collection.MCN.models import MCN_GFK, MultiChannel_MCN_GFK
from model_collection.MCN.models import MCN_WFK,MultiChannel_MCN_WFK

from trainer.trainer_basic import Basic_plmodel
from trainer.trainer_set import trainer_set

def load_models(config_dir,best_model_path=None):
    # 解析配置文件
    configs, args, path = yaml_arguments(config_dir)
    seed_everything(args.seed)
    
    # 根据配置初始化模型
    if args.model in ['TSPN','TKAN','NNSPN','TFON']:
        signal_processing_modules, feature_extractor_modules = config_network(configs, args)
        MODEL_DICT = {
            'TSPN': lambda x: Transparent_Signal_Processing_Network(signal_processing_modules, feature_extractor_modules,x),
            'TKAN': lambda x: Transparent_Signal_Processing_KAN(signal_processing_modules, feature_extractor_modules,x),
            'NNSPN': lambda x: NN_Signal_Processing_Network(signal_processing_modules, feature_extractor_modules,x),
            'TFON': lambda args: Time_Frequency_Operator_Network(signal_processing_modules, feature_extractor_modules,args),
            
}
    else:
        args.ff = np.arange(0, args.in_dim//2 + 1) / args.in_dim//2 + 1

        MODEL_DICT = {
            'Resnet': lambda args: ResNet(BasicBlock, [2, 2, 2, 2], in_channel=args.in_channels, num_class=args.num_classes),
            'WKN_m': lambda args: WKN_m(BasicBlock, [2, 2, 2, 2], in_channel=args.in_channels, num_class=args.num_classes),
            'Sinc_net_m': lambda args: Sinc_net_m(BasicBlock, [2, 2, 2, 2], in_channel=args.in_channels, num_class=args.num_classes),
            'Huan_net': lambda args: Huan_net(input_size=args.in_channels, num_class=args.num_classes),
            'TFN_Morlet': lambda args: TFN_Morlet(in_channels=args.in_channels, out_channels=args.num_classes),
            'MCN_GFK': lambda args: MultiChannel_MCN_GFK(ff=args.ff, in_channels=args.in_channels, num_MFKs=8, num_classes=args.num_classes),
        }
    network = MODEL_DICT[args.model](args)

    args.wandb_flag = False if not hasattr(args, 'wandb_flag') else args.wandb_flag

    # 准备数据加载器
    _, _, _, test_dataloader = trainer_set(args, path)
    
    # 初始化PyTorch Lightning模型
    model = Basic_plmodel(network, args)
    
    # 如果有最佳检查点，加载最佳检查点
    # best_model_path = args.best_model_path  # 假设args中包含最佳模型路径
    if best_model_path:
        print(f'Loading best model from {best_model_path}')
        state_dict = torch.load(best_model_path)
        model.load_state_dict(state_dict['state_dict'])
    return model, test_dataloader, args
    # 获取测试数据和标签
    
def predict_from_loader(model, test_dataloader):
    test_dataset = test_dataloader.dataset.selected_data
    y_true = test_dataloader.dataset.selected_labels
    y_true = torch.tensor(y_true)#.cuda()
    data = torch.tensor(test_dataset)#.cuda()
    
    # 进行预测
    model.eval()
    with torch.no_grad():
        y_pred = model.network(data)
    
    return y_true, y_pred#.cuda()
    # test_dataset = test_dataloader.dataset.selected_data
    # y_true = test_dataloader.dataset.selected_labels
    # y_true = torch.tensor(y_true).cuda()
    # data = torch.tensor(test_dataset).cuda()
    
    # # 进行预测
    # model.eval()
    # with torch.no_grad():
    #     y_pred = model.network(data)
    
    # return model, data, y_true, y_pred
if __name__ == '__main__':
    pass
    # 使用示例
    # config_dir = 'configs/THU_006/config_TSPN.yaml'
    # model, test_data, test_labels, predictions = load_and_predict(config_dir,best_model_path='save/THU1+THU2剪枝/THU1_new/model-epoch=61-val_loss=0.0545.ckpt')
