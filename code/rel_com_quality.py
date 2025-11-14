import numpy as np
import torch
import dgl
import scipy.sparse as sp
import igraph as ig
from typing import Dict, Tuple, List
from collections import defaultdict

def build_undirected_graph(meta_path_file):

    adj = sp.load_npz(meta_path_file)
    src_nodes = torch.tensor(adj.row, dtype=torch.int32)
    dst_nodes = torch.tensor(adj.col, dtype=torch.int32)
    g = dgl.graph((src_nodes, dst_nodes), idtype=torch.int32)
    g = dgl.add_reverse_edges(g)  # 无向图需要双向边
    g = dgl.remove_self_loop(g)     # 去除自环
    return g

def evaluate_community_metrics_from_labels_and_queries(
    g_dgl: dgl.DGLGraph,
    labels_path: str,
    query_path: str
) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
    # 加载数据
    labels = np.load(labels_path)  # labels[i] = community_id
    query_nodes = np.loadtxt(query_path, dtype=int).tolist()

    # 构造社区字典 {community_id: [node_ids]}
    community_dict = defaultdict(list)
    for node_id, com_id in enumerate(labels):
        community_dict[com_id].append(node_id)

    # 构建 igraph 图
    src, dst = g_dgl.edges()
    edges = list(zip(src.numpy(), dst.numpy()))
    ig_g = ig.Graph()
    ig_g.add_vertices(g_dgl.num_nodes())
    ig_g.add_edges(edges)
    ig_g = ig_g.as_undirected()

    # 计算所有社区指标
    community_metrics_dict = {}
    for com_id, nodes in community_dict.items():
        subgraph = ig_g.subgraph(nodes)
        n = len(nodes)
        m = subgraph.ecount()

        density = 0 if n <= 1 else 2 * m / (n * (n - 1))
        diameter = 0 if n == 0 or m == 0 else subgraph.diameter(directed=False, unconn=True)

        node_set = set(nodes)
        boundary_edges = 0
        vol_in, vol_out = 0, 0
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

        community_metrics_dict[com_id] = {
            'density': density,
            'diameter': diameter,
            'conductance': conductance
        }

    # 查询节点所在社区的平均指标
    query_com_ids = set(labels[q] for q in query_nodes)
    selected_metrics = [community_metrics_dict[cid] for cid in query_com_ids]

    avg_density = np.mean([m['density'] for m in selected_metrics])
    avg_diameter = np.mean([m['diameter'] for m in selected_metrics])
    avg_conductance = np.mean([m['conductance'] for m in selected_metrics])

    query_avg_metrics = {
        'avg_density': avg_density,
        'avg_diameter': avg_diameter,
        'avg_conductance': avg_conductance
    }

    return community_metrics_dict, query_avg_metrics


dataset = 'self_imdb'
g = build_undirected_graph('./embeds/' + dataset + '/final_meta_path.npz')
labels_path = './embeds/' + dataset + '/labels.npy'
query_path = './embeds/' + dataset + '/query.txt'

all_metrics, query_avg = evaluate_community_metrics_from_labels_and_queries(g, labels_path, query_path)

print("查询节点对应社区的平均指标:")
for k, v in query_avg.items():
    print(f"{k}: {v:.4f}")