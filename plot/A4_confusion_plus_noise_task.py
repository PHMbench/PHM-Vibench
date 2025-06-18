import sys

from pyparsing import line
sys.path.append('./')

from post.A1_plot_config import configure_matplotlib
configure_matplotlib(style='ieee', font_lang='en')

from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch

import pandas as pd
import numpy as np
###################### heatmap_confusion ######################


def heatmap_confusion(predictions, test_labels, args,
                      plot_dir='./plot', name='model', cmap='cool', save_type='.pdf'):
    
    pred_labels = predictions.argmax(dim=1) if len(predictions.shape) > 1 else predictions
    true_labels = test_labels.argmax(dim=1) if len(test_labels.shape) > 1 else test_labels

# 计算混淆矩阵
    conf_mat = ConfusionMatrix(task="multiclass",num_classes=args.num_classes).cuda()
    matrix = conf_mat(pred_labels, true_labels).cpu().numpy()
    # plot_dir 
    pd.DataFrame(matrix).to_csv(os.path.join(plot_dir, f'{name}_confusion_matrix.csv'), index=False)
    
    matrix_normalized = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix_normalized, cmap=cmap, annot=True, fmt=".0%", linewidths=.5,annot_kws={"size": 24})
    # plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{name}_confusion_matrix{save_type}'), transparent=True, dpi=512)
    plt.savefig(os.path.join(plot_dir, f'{name}_confusion_matrix.svg'), transparent=True, dpi=512)
    plt.show()
    plt.close()
    
    return matrix




###################### noise ######################

def add_noise(x, snr):
    """
    向数据添加高斯噪声。
    """
    # snr = 10 ** (snr / 10.0)
    # x_power = torch.sum(data ** 2) / data.numel()
    # noise_power = x_power / snr
    # noise = torch.randn_like(data) * torch.sqrt(noise_power)
    # return data + noise
    snr = 10**(snr/10.0)
    xpower = torch.sum(x**2)/(x.size(0)*x.size(1)*x.size(2))
    npower = xpower / snr
    return torch.randn(x.size()).cuda() * torch.sqrt(npower) + x

def plot_accuracy_vs_snr(test_data, test_labels, model_dict, snr_levels, plot_dir='./plot'):

    colors = ['r','y', 'g','c' , 'b', 'm'] 
    markers = ['*','o', 'x', 'v', '^', 's']
    line_styles = ['-', '--', '-.', ':','-', '--',]
    
    # lenth = len(model_dict)
    plt.figure(figsize=(10, 6))
    for name, model in model_dict.items():
        file_name = f'{name}_accuracy_vs_snr'
        accuracies = record_noise_accuracy(test_data, test_labels, model, snr_levels, plot_dir, file_name)
        
        
        index = list(model_dict.keys()).index(name)
        
        plt.plot(snr_levels, accuracies,
                 label=name, linewidth=2,
                 linestyle=line_styles[index],
                 c = colors[index],
                 marker=markers[index])
        
        plt.xlabel('SNR (dB)')
        plt.ylabel('Accuracy')
        # plt.title('Accuracy vs. SNR')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, file_name) + '.pdf', dpi=512)
    plt.savefig(os.path.join(plot_dir, file_name) + '.svg', dpi=512)
    plt.savefig(os.path.join(plot_dir, file_name) + '.png', dpi=512)
    plt.show()
    plt.close()

def record_noise_accuracy(test_data, test_labels, model, snr_levels, plot_dir, file_name):
    accuracies = []
    test_data = torch.tensor(test_data).cuda()
    test_labels = torch.tensor(test_labels).cuda()
    model = model.cuda()
    for snr in snr_levels:
        noisy_data = add_noise(test_data, snr)
        with torch.no_grad():
            preds = model(noisy_data).argmax(dim=1)
        acc = (preds == test_labels).float().mean().item()
        accuracies.append(acc)
    pd.DataFrame(accuracies, columns=['Accuracy']).to_csv(os.path.join(plot_dir, f'{file_name}_noise_accuracy_list.csv'), index=False)
    model = model.cpu()
    torch.cuda.empty_cache()
    return accuracies

def record_noise_attention(test_data, test_labels, model, snr_levels, plot_dir, file_name):
    accuracies = []
    test_data = torch.tensor(test_data).cuda()
    test_labels = torch.tensor(test_labels).cuda()
    model = model.cuda()
    for snr in snr_levels:
        noisy_data = add_noise(test_data, snr)
        with torch.no_grad():
            preds = model(noisy_data).argmax(dim=1)
            signal_attention = model.signal
                        
        acc = (preds == test_labels).float().mean().item()
        accuracies.append(acc)
    pd.DataFrame(accuracies, columns=['Accuracy']).to_csv(os.path.join(plot_dir, f'{file_name}_noise_accuracy_list.csv'), index=False)
    model = model.cpu()
    torch.cuda.empty_cache()
    return accuracies

def parse_attention(noisy_data,model):
    
    model = model.cuda()
    model(torch.tensor(noisy_data).float().cuda())

    model = model.network
    SP_attentions = []
    for layer in model.signal_processing_layers:
        SP_attentions.append(layer.channel_attention.gate)
    FE_attention = model.feature_extractor_layers.FEAttention.gate
    return SP_attentions,FE_attention

# # 设置噪声级别范围
# snr_levels = np.arange(-5, 16, 1)  # 示例：从-5dB到15dB

# # 进行噪声实验并绘制准确率与SNR的关系图
# plot_accuracy_vs_snr(test_data, true_labels, model, snr_levels, plot_dir='你的图表保存目录', file_name='accuracy_vs_snr')


# 绘制混淆矩阵
    



