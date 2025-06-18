
from post.A1_plot_config import configure_matplotlib
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import pandas as pd
from openpyxl import Workbook
from model.TSPN import SignalProcessingLayer, FeatureExtractorlayer,Classifier
# 获取Topk个索引

def get_top_k_weights(idx, layer,k = 10,weight_flag = 'weight_connection'):
    
    """
    获取权重的前 k 个值 get_top_k_weights
    """
    
    print('layer:',idx)
    if weight_flag == 'weight_connection':
        weight = layer.weight_connection.weight # .detach().cpu().numpy()
    elif weight_flag == 'skip_connection':
        layer.skip_connection.weight.data = F.softmax((5.0 / 0.2) *  # 0.09 / 0.2
                                                    layer.skip_connection.weight.data, dim=0)
        weight = layer.skip_connection.weight
    elif weight_flag == 'clf':
        weight = layer.weight
        
    # 计算权重的绝对值
    abs_weight = torch.abs(weight)

    # 将权重矩阵扁平化
    flat_abs_weights = abs_weight.flatten()

    # 使用 topk 方法找到扁平化权重中最大的 k 个元素及其索引
    values, flat_indices = flat_abs_weights.topk(k, largest=True)

    # 如果需要，将扁平化后的索引转换回原始矩阵的二维索引
    row_indices, col_indices = np.unravel_index(flat_indices.cpu().numpy(), abs_weight.shape)
    result_list = []

    # 在循环中将每个元素的信息添加到列表中
    for i in range(k):
        result_list.append((row_indices[i], col_indices[i], values[i].item()))
    return result_list


import networkx as nx

def signal_weight(model,k=20,weight_flag = 'weight_connection'):
    """
    每一层的权重提取信号权重提取 signal_weight
    
    """
    layer_top_weight = {}
    for idx, layer in enumerate(model.signal_processing_layers):
        result_list = get_top_k_weights(idx, layer,k=k,weight_flag = weight_flag)
        layer_top_weight[f'layer:{idx}'] = result_list
        print(result_list)
    return layer_top_weight
        
def draw_network_structure(layer_top_weight,precision = 6,
                           save_path='./plot',
                           filter_flag = False,
                           name = 'network_structure2'):
    """
    绘制网络结构图。
    
    参数:
    - layer_top_weight: 包含网络层和权重数据的字典。
    - save_path: 图片保存路径，不包含文件后缀。
    """
    G = nx.DiGraph()

    # 添加边
    for idx, (layer, tuples) in enumerate(layer_top_weight.items()):
        for target_idx, source_idx, weight in tuples:
            if filter_flag:
                if weight < 1e-4:  # 裁剪掉权重小于1e-4的边
                    continue                
            source_node = f'$s^{idx}_{{{source_idx}}}$'
            target_node = f'$s^{idx+1}_{{{target_idx}}}$'
            weight_ = round(weight, precision)
            G.add_edge(source_node, target_node, weight=weight_)
            
    if filter_flag:
        G = remove_edges_below_threshold(G)
        G = remove_unconnected_nodes(G)
        
    # 为了美化图形，为每一层的节点分配垂直位置
    pos = {}
    layer_levels = {}
    for node in sorted(G.nodes): # TODO number order
        layer = parse_layer(node)
        if layer not in layer_levels:
            layer_levels[layer] = 0
        pos[node] = (layer_levels[layer], -layer)
        layer_levels[layer] += 1


###############################################


    # 绘图
    # plt.figure(figsize=(12, 8))
    # nx.draw(G, pos, with_labels=True, node_size=2600, node_color="cornflowerblue", font_size=40, font_weight="bold", 
    #         edge_color="gray", width=2, arrowstyle="->", arrowsize=10)
    # edge_labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='yellowgreen', font_size=15)

# # 绘图 改颜色
#     plt.figure(figsize=(12, 8))
#     edges = G.edges(data=True)
#     weights = [d['weight'] for (u, v, d) in edges]
#     edge_colors = [plt.cm.Blues(weight) for weight in weights]

#     nx.draw(G, pos, with_labels=True, node_size=2600, node_color="cornflowerblue", 
#             font_size=40, font_weight="bold", edge_color=edge_colors, width=2, 
#             arrowstyle="->", arrowsize=10, alpha=0.7)
#     edge_labels = nx.get_edge_attributes(G, 'weight')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='yellowgreen', font_size=15)

## 改透明度
    # 绘图
    
    plt.figure(figsize=(12, 8))
    edges = G.edges(data=True)
    edge_weights = [d['weight'] for _, _, d in edges]
    max_weight = max(edge_weights) if edge_weights else 1
    min_weight = min(edge_weights) if edge_weights else 0.000001
    if max_weight != min_weight:
        edge_alphas = [(weight - min_weight) / (max_weight - min_weight) for weight in edge_weights]
    else:
        edge_alphas = [1] * len(edge_weights)
    edge_colors = [(0, 0, 0, alpha) for alpha in edge_alphas]

    nx.draw(G, pos, with_labels=True, node_size=2600, node_color="cornflowerblue", font_size=40, font_weight="bold", 
            edge_color=edge_colors, width=2, arrowstyle="->", arrowsize=10)
    
    # edge_labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='yellowgreen', font_size=15)
    

    plt.axis('off')  # 隐藏坐标轴
    plt.savefig(f'{save_path}/filter_flag_{filter_flag}{name}.png')
    plt.savefig(f'{save_path}/filter_flag_{filter_flag}{name}.svg')
    plt.show()
    
def parse_layer(node):
    node_plain = node.replace('$', '').replace('{', '').replace('}', '')   
    layer, idx = node_plain.split('^')[1].split('_')
    layer = int(layer)
    idx = int(idx)
    return layer

def remove_edges_below_threshold(G, threshold=1e-4):
    """移除权重低于阈值的边。"""
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < threshold]
    G.remove_edges_from(edges_to_remove)
    return G

def remove_unconnected_nodes(G):
    """移除没有入度或出度的节点，对于第一层和最后一层节点有特殊处理。"""
    removed = True
    while removed:
        removed = False
        nodes_to_remove = [node for node in G.nodes if G.in_degree(node) == 0 and G.out_degree(node) == 0]
        
        if nodes_to_remove:
            G.remove_nodes_from(nodes_to_remove)
            removed = True

        # 对于中间层节点，检查并移除那些没有入度或出度的节点
        max_layer_idx = max([parse_layer(node) for node in G.nodes])
        for node in list(G.nodes):
            layer = parse_layer(node)
            if (G.in_degree(node) == 0 and layer != 0) or (G.out_degree(node) == 0 and layer != max_layer_idx):
                G.remove_node(node)
                removed = True
    return G    


import matplotlib.pyplot as plt
import seaborn as sns
import torch


def extract_weight_matrices(model, k=20, weight_flag='weight_connection'):
    """
    提取模型中前k个信号处理层的权重矩阵。
    
    参数:
    - model: 包含信号处理层的模型（nn.Module的实例）。
    - k: 提取的层的数量。如果模型层数超过k，则只提取前k层。
    - weight_flag: 权重属性的名称，默认为'weight_connection'。
    
    返回:
    - weights: 一个列表，包含提取的权重矩阵（NumPy数组）。
    """
    weights = []
    for idx, layer in enumerate(model):
        if idx >= k:
            break  # 只处理前k层
        
        if not hasattr(layer, weight_flag):
            print(f"层 {idx} 不包含属性 '{weight_flag}'，跳过。")
            continue
        
        weight_tensor = getattr(layer, weight_flag).weight.abs()  # 获取权重张量并取绝对值
        weight_data = weight_tensor.detach().cpu().numpy()  # 转为NumPy数组
        
        if len(weight_data.shape) > 2:
            weight_data = weight_data.squeeze()  # 如果权重是多维的，尝试去掉单维度
        
        weights.append((idx, weight_data))
    
    return weights

def extract_classifier_linear_weights(clf_module):
    """
    从分类器模块中提取所有线性层的权重矩阵。
    
    参数:
    - clf_module: 分类器模块（例如 model.network.clf）。
    
    返回:
    - linear_weights: 一个字典，键为线性层的名称或索引，值为对应的权重矩阵（NumPy数组）。
    """
    linear_weights = {}
    for idx, layer in enumerate(clf_module.children()):
        if isinstance(layer, nn.Linear):
            weight_tensor = layer.weight.detach().cpu().numpy()
        if len(weight_tensor.shape) > 2:
            weight_tensor = weight_tensor.squeeze()  # 如果权重是多维的，尝试去掉单维度            
        linear_weights[f'linear_{idx}'] = weight_tensor

    return linear_weights

def plot_weight_heatmap(weight_data, path='./plot', name='', idx=0, weight_flag='weight_connection'):
    """
    绘制并保存权重矩阵的热力图。
    
    参数:
    - weight_data: 要绘制的权重矩阵（NumPy数组）。
    - path: 图片保存路径，默认为'./plot'。
    - name: 图片保存的基础名字。
    - idx: 层的索引，用于在标题和文件名中标识。
    - weight_flag: 权重属性的名称，默认为'weight_connection'。
    """
    # 确保保存路径存在
    os.makedirs(path, exist_ok=True)
    
    plt.figure(figsize=(10, 8))  # 可以根据需要调整图形大小
    sns.heatmap(weight_data, cmap='BuPu', annot=False, fmt="f",
                linewidths=.5, xticklabels='', yticklabels='')  # 设置色彩映射为BuPu
    plt.title(f'Layer {idx} {weight_flag} Heatmap')  # 标题可以包含层的索引
    plt.tight_layout()
    
    # 保存为PDF和SVG
    pdf_path = os.path.join(path, f'{weight_flag}{name}_layer{idx}.pdf')
    svg_path = os.path.join(path, f'{weight_flag}{name}_layer{idx}.svg')
    plt.savefig(pdf_path, dpi=256)
    plt.savefig(svg_path, dpi=256)
    plt.close()  # 关闭当前图形以释放内存

def signal_vis_weight(model, path='./plot', name='', k=20, weight_flag='weight_connection'):
    """
    绘制模型信号处理层的权重热力图。 TODO 由上述两个函数替换
    
    参数:
    - model: 包含信号处理层的模型。
    - path: 图片保存路径。
    - name: 图片保存的基础名字。
    - k: 可视化的层的数量（如果模型层数超过此数，则只显示前k层）。
    - weight_flag: 权重属性的名称，默认为'weight_connection'。
    """
    for idx, layer in enumerate(model):
        if idx >= k:
            break  # 只处理前k层
        
        weight_tensor = getattr(layer, weight_flag).weight.abs()  # 获取权重张量
        weight_data = weight_tensor.detach().cpu().numpy()  # 转为NumPy数组
        
        if len(weight_data.shape) > 2:
            weight_data = weight_data.squeeze()  # 如果权重是多维的，尝试去掉单维度
        
        # 绘制热力图
        plt.figure(figsize=(10, 8))  # 可以根据需要调整图形大小
        sns.heatmap(weight_data, cmap='BuPu', annot=False, fmt="f",
                    linewidths=.5, xticklabels='', yticklabels='')  # 设置色彩映射为PuBu
        plt.title(f'Layer {idx} Weight Heatmap')  # 标题可以包含层的索引
        plt.savefig(path + f'/{weight_flag}{name}_layer{idx}.pdf', dpi=256)  # 保存为PDF
        plt.savefig(path + f'/{weight_flag}{name}_layer{idx}.svg', dpi=256) 
        plt.show()  # 显示图形


def parse_attention(noisy_data,model):
    
    model = model.cuda()
    model(torch.tensor(noisy_data).float().cuda())

    model = model.network
    SP_attentions = []
    for layer in model.signal_processing_layers:
        SP_attentions.append(layer.channel_attention.gate.squeeze().detach().cpu().numpy())
    FE_attention = model.feature_extractor_layers.FEAttention.gate.squeeze().detach().cpu().numpy()
    model = model.cpu()
    torch.cuda.empty_cache()
    
    return SP_attentions,FE_attention

def parse_input(noisy_data,model):
    
    model = model.cuda()
    y = model(torch.tensor(noisy_data).float().cuda())

    model = model.network
    SP_xs = []
    for layer in model.signal_processing_layers:
        SP_xs.append(layer.channel_attention.x.squeeze().detach().cpu().numpy())
    FE_x= model.feature_extractor_layers.FEAttention.x.squeeze().detach().cpu().numpy()
    model = model.cpu()
    torch.cuda.empty_cache()
    
    return SP_xs,FE_x,y.detach().cpu().numpy()

def parse_attention_batch(noisy_data, model, batch_size):
    model = model.cuda()
    num_batches = len(noisy_data) // batch_size + (1 if len(noisy_data) % batch_size != 0 else 0)
    
    all_SP_attentions = []
    all_FE_attentions = []
    with torch.no_grad():
        for i in range(num_batches):
            batch_data = noisy_data[i * batch_size:(i + 1) * batch_size]
            batch_data = torch.tensor(batch_data).float().cuda()
            
            SP_attentions, FE_attention = parse_attention(batch_data, model)


            if all_SP_attentions == []:
                all_SP_attentions = [[] for _ in range(len(SP_attentions))]
            
            for j, sp_attention in enumerate(SP_attentions):
                all_SP_attentions[j].append([sp_attention])
                            
            all_SP_attentions = [[np.concatenate(attentions, axis=0)] for attentions in all_SP_attentions]
            all_FE_attentions.append(FE_attention)
            
            # 清空缓存
            del batch_data
            torch.cuda.empty_cache()
    # 在第0维度拼接
    
    all_SP_attentions = np.concatenate(all_SP_attentions, axis=0)
    all_SP_attentions = all_SP_attentions.reshape(len(SP_attentions), -1, all_SP_attentions.shape[-1])
    all_FE_attentions = np.concatenate(all_FE_attentions, axis=0)    
    return all_SP_attentions, all_FE_attentions

def visualize_Attention(sparse_matrix, labels, channel_groups, path='./plot', name=''):
    plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体为新罗马

    # 设置绘图风格
    sns.set_theme(style="whitegrid", font='Times New Roman', font_scale=1.4)
    sns.set_palette("Set2")

    # 创建一个大的绘图区域，包含多个子图
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # 第一个子图：热力图
    sns.heatmap(sparse_matrix, ax=axs[0, 0], cmap="GnBu", cbar_kws={'label': 'Weight'}, alpha=0.8)  # jet cmap="jet",
    # axs[0, 0].set_title('Sparse Attention Matrix')
    axs[0, 0].set_xlabel('Channel')
    axs[0, 0].set_ylabel('Sample')

    # 第二个子图：根据channel_groups的权重求和
    summed_weights = np.array([sparse_matrix[:, group].sum(axis=1) for group in channel_groups]).T
    sns.heatmap(summed_weights, ax=axs[0, 1], cmap="BuGn", cbar_kws={'label': 'Weight'}, alpha=0.8)
    # axs[0, 1].set_title('Summed Weights by Channel Group')
    axs[0, 1].set_xlabel('Operator Group')
    axs[0, 1].set_ylabel('Sample')

    # 第三个子图：根据类别绘制累积的样本权重分布
    unique_categories = np.unique(labels)
    category_sums = np.array([sparse_matrix[labels == cat].sum(axis=0) for cat in unique_categories])
    sns.heatmap(category_sums, ax=axs[1, 0], cmap="BuPu", cbar_kws={'label': 'Weight'}, alpha=0.8)
    # axs[1, 0].set_title('Summed Weights by Category')
    axs[1, 0].set_xlabel('Channel')
    axs[1, 0].set_ylabel('Label')

    # 第四个子图：标签和通道种类的非零元素数量分布
    label_channel_group_counts = np.zeros((len(unique_categories), len(channel_groups)))
    for label_idx, label in enumerate(unique_categories):
        label_samples = sparse_matrix[labels == label]
        for group_idx, group in enumerate(channel_groups):
            label_channel_group_counts[label_idx, group_idx] = np.count_nonzero(label_samples[:, group])

    sns.heatmap(label_channel_group_counts, ax=axs[1, 1], annot=True, fmt=".0f",cmap="OrRd", cbar_kws={'label': 'Count'}, alpha=0.8)
    # axs[1, 1].set_title('Counts by Label and Channel Group')
    axs[1, 1].set_xlabel('Operator Group')
    axs[1, 1].set_ylabel('Label')

    # 调整布局，使子图不重叠
    plt.tight_layout()

    # # 显示图形
    plt.savefig(path + f'/Attention{name}.pdf', dpi=256)  # 保存为PDF
    plt.savefig(path + f'/Attention{name}.svg', dpi=256)
    plt.show()
    
def generate_channel_groups(num_features, scale, name):
    """
    Generate channel groups based on the feature names, scale, and name.
    
    Parameters:
    - feature_names: List of feature names.
    - scale: The scale parameter.
    - name: The name parameter to determine the generation rule.
    
    Returns:
    - channel_groups: List of channel groups.
    """

    channel_groups = []

    if "FE" in name:
        # Rule when name does not contain "FE"
        for i in range(num_features):
            group = [i]
            for j in range(1, scale):
                group.append(i + j * num_features)
            channel_groups.append(group)
    else:
        # Rule when name contains "FE"
        for i in range(num_features):
            feature_list = []
            for j in range(scale):
                feature_list.append(i + num_features * j)
            channel_groups.append(feature_list)
    
    return channel_groups  



def get_all_layers(module, layers=None):
    """
    递归收集所有子模块。
    
    Args:
        module (nn.Module): 要遍历的主模块。
        layers (list, optional): 用于存储子模块的列表。默认为 None。
        
    Returns:
        list: 所有子模块的列表。
    """
    if layers is None:
        layers = []
    for child in module.children():
        # 如果子模块没有更深层的子模块，则直接添加
        if not list(child.children()):
            layers.append(child)
        else:
            # 否则，递归调用自身
            get_all_layers(child, layers)
    return layers

# def save_linear_weights_to_excel(model, filename):
#     """
#     保存模型中所有 Linear 层的权重到 Excel 文件中，每个 Linear 层的权重保存在不同的工作表中。
    
#     Args:
#         model (nn.Module): 要处理的模型。
#         filename (str): 保存的 Excel 文件名（包括路径）。
#     """
#     # 获取所有子模块
#     layers = get_all_layers(model)
#     # 筛选出 Linear 层
#     linear_layers = [layer for layer in layers if isinstance(layer, nn.Linear)]
    
#     if not linear_layers:
#         print("模型中没有找到 Linear 层。")
#         return
    
#     # 使用 Pandas 的 ExcelWriter 保存到 Excel
#     with pd.ExcelWriter(filename, engine='openpyxl') as writer:
#         for idx, layer in enumerate(linear_layers):
#             # 尝试获取模块的名称
#             module_name = None
#             for name, mod in model.named_modules():
#                 if mod is layer:
#                     module_name = name
#                     break
#             if module_name is None:
#                 module_name = f'Linear_{idx+1}'
#             # Excel 工作表名称限制为 31 个字符
#             sheet_name = module_name[:31]
#             # 获取权重数据
#             weight = layer.weight.data.cpu().numpy()
#             # 转换为 DataFrame
#             df = pd.DataFrame(weight)
#             # 将 DataFrame 写入 Excel 工作表
#             df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
#     print(f"所有 Linear 层的权重已保存到 {filename}")

def save_linear_weights_to_excel(model, filename):
    """
    保存模型中所有 Linear 层的权重到 Excel 文件中，每个 Linear 层的权重保存在不同的工作表中。
    在每个工作表中，第一列为操作符名称，第一行为通道编号。
    
    Args:
        model (nn.Module): 要处理的模型。
        filename (str): 保存的 Excel 文件名（包括路径）。
    """
    # 获取所有子模块
    layers = get_all_layers(model)
    # 筛选出 Linear 层
    linear_layers = [layer for layer in layers if isinstance(layer, nn.Linear)]
    
    if not linear_layers:
        print("模型中没有找到 Linear 层。")
        return
    
    # 使用 Pandas 的 ExcelWriter 保存到 Excel
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for idx, layer in enumerate(linear_layers):
            # 尝试获取模块的名称和路径
            module_name = None
            for name, mod in model.named_modules():
                if mod is layer:
                    module_name = name
                    break
            if module_name is None:
                module_name = f'Linear_{idx+1}'
            
            # Excel 工作表名称限制为 31 个字符
            sheet_name = module_name[:31]
            
            # Determine the context of the Linear layer
            parent_module = None
            for name, mod in model.named_modules():
                if layer in list(mod.children()):
                    parent_module = mod
                    break
            
            operator_names = []
            if isinstance(parent_module, SignalProcessingLayer):
                # 获取 signal_processing_modules 的名称
                modules = list(parent_module.signal_processing_modules.keys())
                num_modules = len(modules)
                out_features = layer.weight.data.size(0)
                channels_per_module = out_features // num_modules
                # 如果无法均分，分配多一些给前面的模块
                counts = [channels_per_module + 1 if i < out_features % num_modules else channels_per_module for i in range(num_modules)]
                for module, count in zip(modules, counts):
                    operator_names.extend([module] * count)
            elif isinstance(parent_module, FeatureExtractorlayer):
                # 获取 feature_extractor_modules 的名称
                modules = list(parent_module.feature_extractor_modules.keys())
                num_modules = len(modules)
                out_features = layer.weight.data.size(0)
                channels_per_module = out_features // num_modules
                counts = [channels_per_module + 1 if i < out_features % num_modules else channels_per_module for i in range(num_modules)]
                for module, count in zip(modules, counts):
                    operator_names.extend([module] * count)
            elif isinstance(parent_module, Classifier):
                # 使用 feature1, feature2, ... 作为操作符名称
                out_features = layer.weight.data.size(0)
                operator_names = [f'feature{i}' for i in range(out_features)]
            else:
                # 其他 Linear 层，使用默认名称
                out_features = layer.weight.data.size(0)
                operator_names = [f'Linear_{idx}_out_{i}' for i in range(out_features)]
            
            in_features = layer.weight.data.size(1)
            weight = layer.weight.data.cpu().numpy()
            
            # 检查 operator_names 是否与 out_features 匹配
            if len(operator_names) != weight.shape[0]:
                print(f"警告: 在模块 '{module_name}' 中，操作符名称数量 ({len(operator_names)}) 与输出特征数量 ({weight.shape[0]}) 不匹配。将使用默认名称。")
                operator_names = [f'out_{i}' for i in range(weight.shape[0])]
            
            # 创建 DataFrame
            df = pd.DataFrame(weight)
            # 添加操作符名称作为第一列
            df.insert(0, 'Operator', operator_names)
            # 创建通道编号列表
            channel_numbers = [f'Ch_{i}' for i in range(weight.shape[1])]
            # 设置第一行为通道编号
            df.columns = ['Operator'] + channel_numbers
            
            # 将 DataFrame 写入 Excel 工作表
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"所有 Linear 层的权重已保存到 {filename}")