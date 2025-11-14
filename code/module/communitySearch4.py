import numpy as np
import torch
import dgl
import scipy.sparse as sp
from collections import defaultdict, deque
import time
import json
import igraph as ig
from typing import Tuple
import os.path as osp



def dgl_to_igraph(g: dgl.DGLGraph) -> ig.Graph:
    """
    将 DGL 无向图转换为 igraph 图
    """
    src, dst = g.edges()
    edges = list(zip(src.numpy(), dst.numpy()))
    n_nodes = g.num_nodes()
    ig_g = ig.Graph()
    ig_g.add_vertices(n_nodes)
    ig_g.add_edges(edges)
    ig_g = ig_g.as_undirected()
    return ig_g

def community_metrics(g_dgl: dgl.DGLGraph, community_nodes: np.ndarray) -> Tuple[float, int, float]:

    if community_nodes.ndim != 1:
        raise ValueError("community_nodes must be a 1D ndarray of integers.")

    community_nodes_list = community_nodes.tolist()
    node_set = set(community_nodes_list)

    # 转换为 igraph 图
    ig_g = dgl_to_igraph(g_dgl)

    # 子图
    subgraph = ig_g.subgraph(community_nodes_list)

    # 1. 密度
    n = len(community_nodes_list)
    m = subgraph.ecount()
    density = 0 if n <= 1 else 2 * m / (n * (n - 1))

    # 2. 直径
    diameter = 0 if n == 0 or m == 0 else subgraph.diameter(directed=False, unconn=True)

    # 3. 传导性
    boundary_edges = 0
    vol_in = 0
    vol_out = 0
    for v in range(ig_g.vcount()):
        deg = ig_g.degree(v)
        if v in node_set:
            vol_in += deg
            for nbr in ig_g.neighbors(v):
                if nbr not in node_set:
                    boundary_edges += 1
        else:
            vol_out += deg
    conductance = boundary_edges / min(vol_in, vol_out) if min(vol_in, vol_out) > 0 else 0

    return density, diameter, conductance

def batch_compute_similarity(embeddings, query_idx):

    query_emb = embeddings[query_idx]  # 形状[D]

    # 矩阵乘法计算点积（形状[N]）
    dot_product = torch.matmul(embeddings, query_emb)

    # 计算L2范数（形状均为标量扩展为[N]）
    norm_emb = torch.norm(embeddings, dim=1)  # 形状[N]
    norm_query = torch.norm(query_emb)  # 标量
    norms = norm_emb * norm_query + 1e-8  # 加平滑项防除零

    return dot_product / norms  # 形状[N]


def optimized_community_search(query_node, embeddings, graph, w=0.5, device='cuda', max_iter=10000):

    # 设备配置（自动检测GPU可用性）
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # 数据转换到设备（重要优化点：减少CPU-GPU数据传输）
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    graph = graph.to(device)

    # 批量计算相似度得分（形状[N]）
    sim_scores = batch_compute_similarity(emb_tensor, query_node)
    avg_score = torch.mean(sim_scores)  # 平均得分用于密度计算

    # 初始化数据结构（使用布尔掩码代替集合操作）
    visited = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=device)
    visited[query_node] = True  # 初始包含查询节点

    # 邻居节点管理（使用张量代替集合提升性能）
    neighbors = graph.successors(query_node).unique()  # 形状[S], S为邻居数

    # 最佳社区追踪
    best_density = -torch.inf
    best_mask = visited.clone()

    # 贪心扩展循环
    for _ in range(max_iter):
        # 筛选未访问的有效邻居（形状[K], K <= S）
        valid_mask = ~visited[neighbors]
        valid_neighbors = neighbors[valid_mask]

        if valid_neighbors.numel() == 0:
            break  # 无有效邻居时终止

        # 选择最高得分节点（O(1)复杂度）
        neighbor_scores = sim_scores[valid_neighbors]  # 形状[K]
        best_idx = torch.argmax(neighbor_scores)
        best_node = valid_neighbors[best_idx]

        # 更新访问状态（原地操作减少内存分配）
        visited[best_node] = True

        # 扩展新邻居（批量操作提升性能）
        new_neighbors = graph.successors(best_node).unique()
        neighbors = torch.cat([neighbors, new_neighbors]).unique()

        # 动态密度计算（关键公式实现）
        current_nodes = visited.nonzero().squeeze()
        current_scores = sim_scores[current_nodes]
        current_density = (current_scores.sum() - len(current_scores) * avg_score) / (len(current_scores) ** w)

        # 更新最佳社区
        if current_density > best_density:
            best_density = current_density
            best_mask = visited.clone()
        else:
            break  # 密度不再增加时提前终止

    # 结果转换回CPU（避免设备不一致问题）
    result_nodes = best_mask.nonzero().squeeze().cpu().numpy()
    print(f"搜索出的社区大小={len(result_nodes)}\n")
    return result_nodes, len(result_nodes)


def vectorized_evaluation(pred_nodes, gt_nodes, total_nodes):

    # 转换为布尔掩码（形状[N]）
    pred_mask = torch.zeros(total_nodes, dtype=torch.bool)
    pred_mask[pred_nodes] = True
    gt_mask = torch.zeros(total_nodes, dtype=torch.bool)
    gt_mask[gt_nodes] = True

    # 集合运算的向量化实现
    intersection = torch.logical_and(pred_mask, gt_mask).sum().item()
    union = torch.logical_or(pred_mask, gt_mask).sum().item()

    # 指标计算
    precision = intersection / max(pred_mask.sum().item(), 1e-8)
    recall = intersection / max(gt_mask.sum().item(), 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = intersection / max(union, 1e-8)

    return (precision, recall, f1, iou)


def search_com_optimized(path, g, w=0.1):

    with open(f"{path}/rel_communities.json", 'r') as f:
        community_data = json.load(f)

    # 创建快速查找结构（节点到社区的映射）
    node_to_community = {
        int(node): list(map(int, members))
        for node, members in community_data["communities"].items()
    }

    # 节点嵌入（形状[N, D]）
    embeddings = np.load(f"{path}/PreEmb.npy")
    print("embeddings shape:", embeddings.shape)
    # 查询节点列表（形状[Q]）
    query_nodes = np.loadtxt(f"{path}/query4.txt", dtype=int)

    # 验证查询节点存在性
    missing_nodes = [n for n in query_nodes if n not in node_to_community]
    # if missing_nodes:
    #     raise ValueError(f"查询节点 {missing_nodes} 在社区文件中不存在")
    # 输出缺失的节点
    if missing_nodes:
        print(f"以下查询节点在社区文件中不存在，已移除: {missing_nodes}")

    # 移除不存在的查询节点
    query_nodes = [n for n in query_nodes if n in node_to_community]

    # 图构建 ----------------------------------------------------
    graph = g

    metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'iou': [],
        'time': []
    }

    query_com_len = 0
    # 主查询循环 -------------------------------------------------
    for q_node in query_nodes:
        start_time = time.time()

        # 执行优化后的社区搜索
        community, com_len = optimized_community_search(
            query_node=q_node,
            embeddings=embeddings,
            graph=graph,
            w=w
        )
        query_com_len += com_len
        # 获取真实标签节点
        # gt_nodes = label_to_nodes[labels[q_node]]
        gt_nodes = node_to_community[q_node]

        # 计算评估指标
        precision, recall, f1, iou = vectorized_evaluation(
            pred_nodes=community,
            gt_nodes=gt_nodes,
            total_nodes=graph.num_nodes()
        )
        # density1, diameter1, conductance1 = community_metrics(graph, community)


        # 记录结果
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        metrics['iou'].append(iou)

        metrics['time'].append(time.time() - start_time)

    print(f"搜索出的社区的平均大小={query_com_len / len(query_nodes)}\n")

    # 结果保存 --------------------------------------------------
    result_file = f"{path}/result.txt"
    with open(result_file, 'w') as f:
        # 写入表头
        f.write("Precision\tRecall\tF1\tIoU\tTime(ms)\n")
        # 逐行写入数据
        for p, r, f1, i, t in zip(metrics['precision'], metrics['recall'],
                                  metrics['f1'], metrics['iou'], metrics['time']):
            f.write(f"{p:.4f}\t{r:.4f}\t{f1:.4f}\t{i:.4f}\t{t * 1000:.2f}\n")

        # 计算并写入平均值
        avg_p = np.mean(metrics['precision'])
        avg_r = np.mean(metrics['recall'])
        avg_f1 = np.mean(metrics['f1'])
        avg_iou = np.mean(metrics['iou'])
        avg_time = np.mean(metrics['time'])

        f.write(f"\nAverages:\n{avg_p:.4f}\t{avg_r:.4f}\t{avg_f1:.4f}\t{avg_iou:.4f}\t{avg_time * 1000:.2f}")

    return metrics


# 兼容原接口
def search_com4(path, w, g):
    g = g.to('cpu')
    return search_com_optimized(path, g, w)

