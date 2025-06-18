
import sys
sys.path.append('./')
import torch
from post.A1_plot_config import configure_matplotlib
import matplotlib.pyplot as plt
import numpy as np
configure_matplotlib(style='ieee', font_lang='en')


def compute_frequency_domain(signal, sample_rate):
    n = len(signal)
    fft_values = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(n, 1.0/sample_rate)
    return frequencies, np.abs(fft_values)

def plot_signals(signals, rows, cols, sample_rate, name,time = 1, fre_range=None):
    plt.figure(figsize=(32, 8))  # 调整图像大小，增加宽度

    num_signals = signals.shape[1]
    for i in range(num_signals):
        signal = signals[:, i]
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        length = len(signal)
        t = torch.linspace(0, time,length)
        w = np.linspace(0, 0.5, length // 2 + 1 if length % 2 == 0 else (length + 1) // 2)
        # 计算当前信号应该位于的子图索引
        time_plot_index = i * 2 + 1
        freq_plot_index = i * 2 + 2

        # 绘制时域图
        plt.subplot(rows, cols, time_plot_index)
        plt.plot(t,signal, c='purple')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        # plt.xticks([])
        # plt.yticks([])

        # 计算并绘制频域图
        frequencies, fft_values = compute_frequency_domain(signal, sample_rate)
        
        plt.subplot(rows, cols, freq_plot_index)
        
        if fre_range:
            plt.plot(w[1:fre_range], fft_values[1:fre_range], c='darkblue')
        else:
            plt.plot(w[1:], fft_values[1:], c='darkblue')
            
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        # plt.xticks([])
        # plt.yticks([])

    plt.tight_layout()
    plt.savefig(f'plot/{name}.pdf')
    plt.savefig(f'plot/{name}.png')
    plt.savefig(f'plot/{name}.svg')
    plt.show()
    plt.close()




def plot_layer_outputs(network, data, num_classes, sample_id, sample_rate=4096):
    num_per_class = len(data) // num_classes
    signal = data
    
    with torch.no_grad():
        for idx, layer in enumerate(network.signal_processing_layers):
            signal = layer(signal)
            rows = 4
            cols = signal.shape[-1] * 2 // rows
            for fault in range(num_classes):
                print(f'Plotting layer {idx} output for class {fault}...')
                id_fault = sample_id + int(num_per_class*fault)
                plot_signals(signals=signal[id_fault].detach().cpu().numpy(),
                            rows=rows, cols=cols,
                            sample_rate=sample_rate,
                            name=f'layer_{idx}_sample_3_{id_fault}',
                            fre_range=300)

def plot_final_outputs(signal,num_classes, sample_id, name = '',sample_rate=4096,fre_range = None):
    num_per_class = len(signal) // num_classes
    signal = signal[:,:,:12] # 前12个
    rows = 4
    cols = signal.shape[-1] * 2 // rows
    for fault in range(num_classes):
        print(f'Plotting mdoel {name} output for class {fault}...')
        id_fault = sample_id + int(num_per_class*fault)
        plot_signals(signals=signal[id_fault],
                    rows=rows, cols=cols,
                    sample_rate=sample_rate,
                    name=f'model_{name}sample_{id_fault}',fre_range=fre_range)