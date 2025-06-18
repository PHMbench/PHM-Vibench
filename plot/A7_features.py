from post.A1_plot_config import configure_matplotlib
import matplotlib.pyplot as plt
import numpy as np
configure_matplotlib(style='ieee', font_lang='en')
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.manifold import TSNE
import pandas as pd


def data_to_tsne_dict(feature_dict):
    '''
    function: 
    '''
    tsne_dict = {}
    for model_name, features in feature_dict.items():
        print('model_name:',model_name)
        # assert features is numpy array 如果是tensor需要转换features.numpy()
        assert isinstance(features, np.ndarray)
        # 假设model对象有一个返回特征的方法get_features
        # features = model.get_features()  # 获取模型的特征
        tsne = TSNE(n_components=2, random_state=0, perplexity=20)
        tsne_features = tsne.fit_transform(features)
        tsne_dict[model_name] = tsne_features
    return tsne_dict

def dict_to_figs(tsne_dict, labels,boarder = None, plot_dir='./plot'):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    num_models = len(tsne_dict)

    for model_name, features in tsne_dict.items(): # 嵌套的字典
        if boarder is not None:
            plot_scatter_from_dict(features, labels,boarder, name=model_name, plot_dir=plot_dir)
        else:
            plot_scatter(features, labels, name=model_name,plot_dir = plot_dir)  # 更新调用以移除num_classes



def plot_scatter(features, labels, name='default',plot_dir = './plot'):
    markers = ['o', 'X', '*', 'P', 's', 'D', '<', '>', '^', 'v']
    colors = plt.cm.Set2(np.linspace(0, 1, 8))  # 使用matplotlib的colormap
    print('plot model:',name)

    # 在函数内部计算num_classes
    num_classes = int(labels.max() + 1)

    for i in range(num_classes):
        idxs = labels == i
        plt.scatter(features[idxs, 0], features[idxs, 1],
                   marker=markers[i % len(markers)],
                   color = 'none',
                   label=f'$C_{i}$', alpha=0.6, edgecolors=colors[i % len(colors)], linewidth=0.5)
    
    # plt.set_title(name, fontsize=14)
    plt.legend(loc='best', fontsize='small',frameon=True, framealpha=0.4)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # 美化图例和标题
    # plt.legend(frameon=True, borderpad=1)
    
    plt.tight_layout()
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(f'{plot_dir}/{name}tsne_all_models.png')
    plt.savefig(f'{plot_dir}/{name}tsne_all_models.pdf')
    plt.savefig(f'{plot_dir}/{name}tsne_all_models.svg')
    plt.show()
    plt.close()
    
def generate_labels(num_samples, num_classes):
    """
    根据给定的样本数量和类别数量生成标签。

    Args:
        num_samples (int): 需要生成标签的样本数量。
        num_classes (int): 类别的数量。

    Returns:
        numpy.ndarray: 生成的标签数组。
    """
    
    labels = np.repeat(np.arange(num_classes), num_samples // num_classes)

    return labels

    #TODO source feature,target feature


def plot_scatter_from_dict(features, labels,boarder, name='default',sample = 25,  plot_dir='./plot'):
    # 设置标记和颜色
    markers = ['o', 'X', '*', 'P', 's', 'D', '<', '>', '^', 'v']
    colors = plt.cm.Set2(np.linspace(0, 1, 8))

    # 确保保存目录存在
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plt.figure()
    print('plot model:', name)

    num_classes = labels.max() + 1

    domain_signal = {}
    for idx_board in range(len(boarder) - 1):
        
    # 分别绘制每个领域的散点图
    # for domain, features in features_dict.items():
        for i in range(num_classes): # 
            idxs = (labels == i)  \
            & (boarder[idx_board] <= np.arange(len(labels))) \
            & (np.arange(len(labels)) < boarder[idx_board+1])
            
            # idxs = idxs and [boarder[idx_board]:boarder[idx_board+1]]
            # idxs = idxs[boarder[idx_board]:boarder[idx_board+1]]
            
            feature_x = features[idxs, 0][:sample] # [boarder[idx_board]:boarder[idx_board+1]]
            feature_y = features[idxs, 1][:sample] # [boarder[idx_board]:boarder[idx_board+1]]
            
            plt.scatter(feature_x, feature_y,
                        marker=markers[i % len(markers)],
                        color = 'none',
                        label=f'$D_{idx_board}C_{i}$',
                        alpha=0.6,
                        edgecolors=colors[idx_board % len(colors)],)

    plt.legend(loc='best', fontsize='small', frameon=True, framealpha=0.4)
    plt.title(name)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    # 保存图形
    plt.savefig(os.path.join(plot_dir, f'{name}_scatter_gen.png'))
    plt.savefig(os.path.join(plot_dir, f'{name}_scatter_gen.pdf'))
    plt.savefig(os.path.join(plot_dir, f'{name}_scatter_gen.svg'))
    plt.show()
    plt.close()
#%%

def plot_scatter_from_dict_with_domain(features, labels, conditions, name='default', sample=25, plot_dir='./plot'):
    # 设置标记和颜色
    markers = ['o', 'X', '*', 'P', 's', 'D', '<', '>', '^', 'v']
    colors = plt.cm.Set2(np.linspace(0, 1, len(np.unique(labels))))

    # 确保保存目录存在
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plt.figure()
    print('plot model:', name)

    num_classes = int(labels.max() + 1)
    num_conditions = int(conditions.max() + 1)

    # 分别绘制每个类别和条件的散点图
    for i in range(num_classes):
        for j in range(num_conditions):
            
            idxs = (labels == i) & (conditions == j)

            feature_x = features[idxs, 0][:sample]
            feature_y = features[idxs, 1][:sample]

            plt.scatter(feature_x, feature_y,
                        marker=markers[i % len(markers)],
                        color=colors[j % len(colors)],
                        label=f'$C_{i}D_{j}$',
                        alpha=0.8,
                        facecolors='none',  # 使标记只显示边缘
                        edgecolors=colors[j % len(colors)]                    
                        )

    plt.legend(loc='best', fontsize='small', frameon=True, framealpha=0.4)
    # plt.title(name)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    # 保存图形
    plt.savefig(os.path.join(plot_dir, f'{name}_scatter_gen.png'),dpi = 512, transparent=True)
    plt.savefig(os.path.join(plot_dir, f'{name}_scatter_gen.pdf'), transparent=True)
    plt.savefig(os.path.join(plot_dir, f'{name}_scatter_gen.svg'), transparent=True)
    plt.show()
    plt.close()

############################################ signal features

def calculate_cosine_similarity_by_feature(data, labels,plot_dir = './plot'):
    unique_labels = np.unique(labels)
    n_features = data.shape[1]
    feature_similarities = {f'f{i}': [] for i in range(n_features)}
    feature_similarities['sum'] = []

    for feature_idx in range(n_features):
        sum_similarity = 0
        for i in unique_labels:
            for j in unique_labels:
                if i != j:
                    # 提取每个类别对应的特征
                    feature_i = data[labels == i, feature_idx].reshape(-1, 1)
                    feature_j = data[labels == j, feature_idx].reshape(-1, 1)
                    # 计算当前特征的余弦相似度
                    similarity = cosine_similarity(feature_i, feature_j).flatten().mean()
                    feature_similarities[f'f{feature_idx}'].append(similarity)
                    sum_similarity += similarity
        feature_similarities['sum'].append(sum_similarity)
        
    similarity_df = pd.DataFrame(feature_similarities)
    similarity_df.to_csv(plot_dir + '/feature_similarities.csv', index=False)

    return feature_similarities

def select_topk_features(similarity_df, k):
    # 对求和的相似度进行排序，取topk个
    sorted_indices = similarity_df['sum'].argsort()[:k]
    topk_feature_names = similarity_df.columns[sorted_indices]
    return topk_feature_names

def plot_mean_and_std_scatter(data, labels, name='TSPN_mean_std'):
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    n_features = data.shape[1]

    # 准备绘图
    markers = ['o', '^', 's', 'd', '*', '+']
    colors = ['darkviolet','blue','r','green', 'y', 'g', 'c', 'b', 'm']

    plt.figure(figsize=(14, 6))

    for i, label in enumerate(unique_labels):
        label_data = data[labels == label]
        mean_values = np.mean(label_data, axis=0)
        std_values = np.std(label_data, axis=0)
        
        plt.errorbar(range(n_features), mean_values, yerr=std_values, fmt=markers[i % len(markers)],markersize = 16,
                     color=colors[i % len(colors)], ecolor=colors[i % len(colors)], 
                     elinewidth=2, capsize=4, label=f'Class {int(label)}')

    plt.title('Feature differences between classes')
    plt.xlabel('Features')
    plt.ylabel('Mean value')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'plot/{name}.pdf')  # 保存图像为PDF文件
    plt.savefig(f'plot/{name}.png')  # 保存图像为PNG文件
    plt.show()
    
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_top_k_features_by_class_similarity(data, labels,name = 'default', k=10):
    unique_labels = np.unique(labels)
    n_features = data.shape[1]
    feature_similarities = {}
    # feature_similarity_sum = np.zeros(n_features)
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i >= j:  # 避免重复计算相同的类别组合或自身
                continue
            feature_similarities[f's{i}_{j}'] = []
    # feature_similarities = {f's{i}_{j}': [] for i in range(unique_labels) for j in range(unique_labels)}
    feature_similarities['sum'] = []
    
    # 遍历每一对不同的类别组合
    for feature_idx in range(n_features):
        sum_similarity = 0
        for i, label_i in enumerate(unique_labels):
            for j, label_j in enumerate(unique_labels):
                if i >= j:  # 避免重复计算相同的类别组合或自身
                    continue

                feature_i = data[labels == i, feature_idx].reshape(1, -1)
                feature_j = data[labels == j, feature_idx].reshape(1, -1)
                
                # 计算两个类别特征平均向量之间的余弦相似度
                similarity = cosine_similarity(feature_i, feature_j)[0][0]
                

                feature_similarities[f's{i}_{j}'].append(similarity)
                sum_similarity += abs(similarity)
        feature_similarities['sum'].append(sum_similarity)
        
    # 对特征的相似度总和进行排序，相似度越低表示特征越有可能区分不同类别
    sorted_indices = np.argsort(feature_similarities['sum'])

    # 选择相似度总和最低的 top-k 特征索引
    topk_indices = sorted_indices[:k]

    # 提取 top-k 特征
    topk_features = data[:, topk_indices]
    
    pd.DataFrame(feature_similarities).to_csv(f'plot/{name}_feature_similarities.csv')

    # 返回 top-k 特征的索引及其对应的相似度列表
    return topk_features, topk_indices

def data_to_selected_dict(feature_dict,labels,k = 2):
    selected_f_dict = {}
    for model_name, features in feature_dict.items():
        # assert features is numpy array 如果是tensor需要转换features.numpy()
        assert isinstance(features, np.ndarray)
        # 假设model对象有一个返回特征的方法get_features
        # features = model.get_features()  # 获取模型的特征
        selected_f = find_top_k_features_by_class_similarity(features, labels, name = model_name, k=k)[0]
        selected_f_dict[model_name] = selected_f
    return selected_f_dict


def plot_mean_and_std_scatter(data, labels, name='TSPN_mean_std'):
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    n_features = data.shape[1]

    data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-6)
    
    # 准备绘图
    markers = ['o', '^', 's', 'd', '*', '+']
    colors = ['darkviolet','blue','r','green', 'y', 'g', 'c', 'b', 'm']

    plt.figure(figsize=(14, 6))

    for i, label in enumerate(unique_labels):
        label_data = data[labels == label]
        mean_values = np.mean(label_data, axis=0)
        std_values = np.std(label_data, axis=0)
        
        plt.errorbar(range(n_features), mean_values, yerr=std_values, fmt=markers[i % len(markers)],markersize = 16,
                     color=colors[i % len(colors)], ecolor=colors[i % len(colors)], 
                     elinewidth=2, capsize=4, label=f'Class {int(label)}')

    plt.title('Feature differences between classes')
    plt.xlabel('Features')
    plt.ylabel('Mean value')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'plot/{name}.pdf')  # 保存图像为PDF文件
    plt.savefig(f'plot/{name}.png')  # 保存图像为PNG文件
    plt.show()
