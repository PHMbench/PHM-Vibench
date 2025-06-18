import sys
sys.path.append('./')
from model.Signal_processing import WaveFilters as filters
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
import pandas as pd
from post.A1_plot_config import configure_matplotlib
configure_matplotlib(style='ieee', font_lang='en')
import math
from concurrent.futures import ThreadPoolExecutor

markers = ['*','o', 'x', 'v', '^', 's']
colors = ['g','c', 'b', 'm']
lines = ['dotted','-','--',':','-.']
# 假设 filters 类和 args 已经定义

############# 根据给定的滤波器参数绘制 #############

def plot_filter_response(filters, f_cs, f_bs, plot_dir='./plot_dir'):
    
    print(f'Plotting filter response...')
    fig, ax = plt.subplots(figsize=(6, 5))
    markers = ['*','o', 'x', 'v', '^', 's']
    colors = ['g','c', 'b', 'm']
    lines = ['dotted','-','--',':','-.']

    for i, f_c in enumerate(f_cs):
        for j, f_b in enumerate(f_bs):
            filters.f_c.data[:, 0, :] = f_c
            filters.f_b.data[:, 0, :] = f_b
            f = filters.filter_generator([1, 1, 1024])
            ax.plot(filters.omega.cpu().detach().numpy()[0, 0, :], f.cpu().detach().numpy()[0, 0, :],
                    linestyle=lines[j], marker=markers[j], color=colors[i],
                    label=f'$f_c$={f_c:.1f}, $f_b$={f_b:.2f}', alpha=0.5)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    fig.legend(loc='upper center', ncol=4, facecolor='gray')
    plt.savefig(f'{plot_dir}/filter_all4.pdf', dpi=512)
    plt.savefig(f'{plot_dir}/filter_all4.svg', dpi=512)
    plt.savefig(f'{plot_dir}/filter_all4.png', dpi=512)
    plt.show()

############# 根据给定的滤波器参数绘制 #############

def plot_time_domain(filters, f_cs, f_bs, plot_dir='./plot_dir'):
    
    print(f'Plotting time domain response...')
    fig, ax_time = plt.subplots(4, 3, figsize=(8, 10))

    for i, f_c in enumerate(f_cs):
        for j, f_b in enumerate(f_bs):
            filters.f_c.data[:, 0, :] = f_c
            filters.f_b.data[:, 0, :] = f_b
            f = filters.filter_generator([1, 1, 1024])
            time = torch.fft.fftshift(torch.fft.irfft(f), dim=2)
            x = np.linspace(0, 1, time.shape[-1])
            ax_time[i, j].plot(x, time.cpu().detach().numpy()[0, 0, :], linestyle='-', marker=markers[j],
                               color=colors[i], label=f'$f_c$={f_c:.1f}, $f_b$={f_b:.2f}', alpha=0.5)
            ax_time[i, j].set_title(f'$f_c$={f_c:.1f}, $f_b$={f_b:.2f}')
            ax_time[i, j].set_xticks([])
            ax_time[i, j].set_ylabel('Amplitude')

    plt.tight_layout()
    plt.savefig(f'{plot_dir}/filter_time.pdf', dpi=512)
    plt.savefig(f'{plot_dir}/filter_time.png', dpi=512)
    plt.savefig(f'{plot_dir}/filter_time.svg', dpi=512)
    plt.show()


def plot_filter_evolution(params_path = 'post/params.csv',
                          channels = 4, layers = 4, plot_dir='./plot_dir',
                          freq_length = 2049):
    """
    绘制滤波器参数随epoch变化的可视化图。

    Args:
        filters: 滤波器实例。
        params_path (str): 滤波器参数CSV文件路径。
        scales (int): 尺度数量。
        layers (int): 层的数量。
        plot_dir (str): 图表保存目录。
    """
    print(f'Plotting filter evolution from {params_path}...')
    params = pd.read_csv(params_path)
    markers = ['*', 'o', 'x', 'v', '^', 's']
    colors = ['r', 'y', 'g', 'c', 'b', 'm']
    lines = ['dotted', '-', '--', ':', '-.']

    dict_len = len(params.columns) // 2 
    row = 4 # 4
    column = dict_len // row # 12 
    
    fig,axs = plt.subplots(row, column, figsize=(10, 8))
    omega = torch.linspace(0, 0.5, freq_length)
    cmap = cm.get_cmap('cool')
    
    lenth = len(params.iloc[:, 0])
    
    def draw_one(ax, f_c_epochs, f_b_epochs):
        for i, (f_c, f_b) in enumerate(zip(f_c_epochs, f_b_epochs)):
            color = cmap(i / lenth)
            filter = torch.exp(-((omega - f_c) / (2 * f_b)) ** 2)
            
            ax.plot(omega, filter,
                    linestyle=lines[1], color=color, alpha=0.8)
            # break # for test
        ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

            
    # 以下是利用draw_one 的并行处理 # 
    with ThreadPoolExecutor(max_workers=36) as executor:  # max_workers根据你的机器性能调整
        futures = []
        for layer in range(layers):
            for channel in range(channels):
                print(f'layer_{layer}_channel_{channel}')
                f_c_epochs = params.iloc[:, channel * layers + layer * 2]
                f_b_epochs = params.iloc[:, channel * layers + layer * 2 + layers]
                # draw_one(axs[layer, channel], f_c_epochs, f_b_epochs)
                futures.append(executor.submit(draw_one, axs[layer, channel], f_c_epochs, f_b_epochs))
                
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=lenth))
    sm.set_array([])
    # # cbar = plt.colorbar(sm)
    # cbar = fig.colorbar(sm, ax=axs.tolist(), orientation='vertical')
    # # cbar.set_label('Epoch', rotation=270, labelpad=20, fontsize=24)
    # cbar.ax.tick_params(labelsize=24)
    fig.subplots_adjust(right=0.9)  # 为colorbar留出空间
    cax = fig.add_axes([1, 0.05, 0.03, 0.9])  # 添加colorbar的轴位置和大小
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Epoch', rotation=270, labelpad=20, fontsize=24)
    cbar.ax.tick_params(labelsize=24)
    
# 创建额外的轴用于放置大图的横坐标和纵坐标标签
    # ax_big_y.yaxis.set_label_position('left')
    # ax_big_y.xaxis.set_label_position('bottom')
    
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/filters_layer_scale.pdf', transparent=True, dpi=512)
    plt.savefig(f'{plot_dir}/filters_layer_scale.png', transparent=True, dpi=512)
    plt.savefig(f'{plot_dir}/filters_layer_scale.svg', transparent=True, dpi=512)
    plt.plot()
    plt.close()
    
from matplotlib import cm, animation
def animate_filter_evolution(params_path='params.csv', channels=4, layers=4, plot_dir='./plot_dir', freq_length=2049):
    print(f'Animating filter evolution from {params_path}...')
    params = pd.read_csv(params_path)
    
    dict_len = len(params.columns) // 2
    row = layers  # Assuming 'layers' represents rows
    column = channels  # Assuming 'channels' represents columns

    fig, axs = plt.subplots(row, column, figsize=(10, 8))
    omega = torch.linspace(0, 0.5, freq_length)
    cmap = cm.get_cmap('cool')
    
    lenth = len(params.iloc[:, 0])

    def update(epoch):
        print(f'Animating epoch {epoch}...')
        for layer in range(layers):
            for channel in range(channels):
                ax = axs[layer, channel]
                ax.clear()
                
                f_c = params.iloc[epoch, channel * layers + layer * 2]
                f_b = params.iloc[epoch, channel * layers + layer * 2 + layers]
                filter = torch.exp(-((omega - f_c) / (2 * f_b)) ** 2)
                
                ax.plot(omega.numpy(), filter.numpy(), color=cmap(epoch / lenth), alpha=0.8)
                ax.set_title(f'Layer {layer} Channel {channel}')
                ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

        # Update the color bar to match the current epoch
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=lenth))
        sm.set_array([])
        # fig.colorbar(sm, ax=axs.ravel().tolist(), orientation='vertical')

    ani = animation.FuncAnimation(fig, update, frames=lenth, repeat=False)
    
    plt.tight_layout()
    ani.save(f'filter_evolution_animation.gif', writer='imagemagick', fps=10)        
    
    # for l in range(layers):
    #     for s in range(scales):
    #         fig, ax = plt.subplots(figsize=(10, 8))
    #         f_c_epochs = params.iloc[:, s * layers + l * 2 + 2]
    #         f_b_epochs = params.iloc[:, s * layers + l * 2 + 3]

    #         lenth = len(f_c_epochs)
    #         cmap = cm.get_cmap('cool')

    #         for i, (f_c, f_b) in enumerate(zip(f_c_epochs, f_b_epochs)):
    #             color = cmap(i / lenth)

    #             filters.f_c.data[:, 0, :] = f_c
    #             filters.f_b.data[:, 0, :] = f_b

    #             f = filters.filter_generator([1, 1, 1024])
    #             ax.plot(filters.omega.cpu().detach().numpy()[0, 0, :], f.cpu().detach().numpy()[0, 0, :],
    #                     linestyle=lines[1], color=color, alpha=0.8)

    #         ax.set_xlabel('Frequency (Hz)', fontsize=24)
    #         ax.set_ylabel('Amplitude', fontsize=24)
    #         ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    #         sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=lenth))
    #         sm.set_array([])
    #         cbar = plt.colorbar(sm)
    #         cbar.set_label('Epoch', rotation=270, labelpad=20, fontsize=24)
    #         cbar.ax.tick_params(labelsize=24)

