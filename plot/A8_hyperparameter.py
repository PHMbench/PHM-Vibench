import sys
sys.path.append('./')
from A1_plot_config import configure_matplotlib
configure_matplotlib(style='ieee', font_lang='en')

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def plot_hyperparameters(x, y, z, xlabel, ylabel, path, filename,
                         method='cubic', log_scale=False, cmap='cool', vmin=None, vmax=None):
    """
    通用的绘图函数，用于超参数的二维平滑图像绘制。
    
    Args:
        x (np.array): x轴数据。
        y (np.array): y轴数据。
        z (np.array): 值数据。
        xlabel (str): x轴标签。
        ylabel (str): y轴标签。
        path (str): 图像保存路径。
        filename (str): 保存文件的名称。
        method (str): 网格数据插值方法，默认为'cubic'。
        log_scale (bool): 是否对x轴和y轴使用对数刻度，默认为False。
        cmap (str): 图像的颜色映射，默认为'cool'。
        vmin (float): 图像颜色映射的最小值。
        vmax (float): 图像颜色映射的最大值。
    """
    # 将数据转换为网格矩阵
    xi, yi = np.linspace(x.min(), x.max(), 1000), np.linspace(y.min(), y.max(), 1000)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method=method)
    
    # 绘制二维平滑图像
    plt.figure(figsize=(8, 6))
    plt.imshow(zi, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    if log_scale:
        plt.yscale("log")
        plt.xscale("log")
        
    # 设置坐标轴标签
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.grid(True, which='both', linestyle='--', linewidth=0.25)
    # 添加颜色条
    plt.colorbar()
    plt.savefig(f'{path}/{filename}.pdf', transparent=True, dpi=512)
    plt.savefig(f'{path}/{filename}.svg', transparent=True, dpi=512)
    plt.show()
    plt.close()

if __name__ == '__main__':
    import pandas as pd

    # 加载数据
    # download from wandb by sweep 
    all = pd.read_csv('post/ALL.csv')
    onlyi = pd.read_csv('post/onlyi.csv')
    onlym = pd.read_csv('post/onlym.csv')
    whot = pd.read_csv('post/whot.csv')
    woWF = pd.read_csv('post/woWF.csv')
    woht = pd.read_csv('post/woht.csv')

    # # 对于初始化参数绘制
    # plot_hyperparameters(init['f_b_mu'], init['f_c_mu'], init['valloss'], '$\mu_b$', '$\mu_c$', './', 'init', method='cubic')

    # # 对于学习率参数绘制
    # plot_hyperparameters(lr['lr'], lr['conv_lr'], lr['valloss'], '$\lambda_p$', '$\lambda_g$', './', 'lr', method='linear', log_scale=True)

    # # 对于有/无Skip连接的scale参数绘制
    # scale_with_skip = scale[scale['skip_connect'] == True]
    # plot_hyperparameters(scale_with_skip['scale'], scale_with_skip['expert_layer'], scale_with_skip['valloss'], '$N_C$', '$N_D$', './', 'scale', method='linear', cmap='cool', vmin=0.1, vmax=0.9)

    # scale_without_skip = scale[scale['skip_connect'] == False]
    # plot_hyperparameters(scale_without_skip['scale'], scale_without_skip['expert_layer'], scale_without_skip['valloss'], '$N_C$', '$N_D$', './', 'scale_wo', method='linear', cmap='cool', vmin=0.1, vmax=0.9)

