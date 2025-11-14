import numpy as np
import scipy.sparse as sp
import torch as th
import dgl
import os.path as osp

from dgl import DGLGraph
from sklearn.preprocessing import OneHotEncoder
import pickle
from typing import List, Tuple, Dict, Union
import json

def sp_to_spt(mat):
    coo = mat.tocoo()                           # 将稀疏矩阵转换为 COO 格式（坐标格式）
    values = coo.data                           # 提取 COO 格式中的数据值
    indices = np.vstack((coo.row, coo.col))

    i = th.LongTensor(indices)
    v = th.FloatTensor(values)
    shape = coo.shape
    # 创建并返回一个 PyTorch 稀疏张量，指定索引、值和形状
    return th.sparse.FloatTensor(i, v, th.Size(shape))

def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return th.from_numpy(mat).type(th.FloatTensor)
    return sp_to_spt(mat)

def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

def preprocess_features(features):

    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def load_acm(type_num, device):
    path = "../data/acm/"
    adj = sp.load_npz(osp.join(path, 'adj.npz'))
    # 构建图
    g = dgl.DGLGraph(adj + (adj.T))  # 保证无向图  adj csr
    g = dgl.remove_self_loop(g)  # 去除自环
    g = dgl.add_self_loop(g)  # 添加自环
    g = g.to(device)  # 将图结构移至指定设备

    feat_p = sp.load_npz(osp.join(path, "p_feat.npz")).astype("float32")
    feat_p = th.FloatTensor(preprocess_features(feat_p)).to(device)
    feat_a = sp.load_npz(osp.join(path, "a_feat.npz")).astype("float32")
    feat_a = th.FloatTensor(preprocess_features(feat_a)).to(device)
    feat_s = th.eye(type_num[2], dtype=th.float32, device=device)
    # 特征矩阵列表 (按类型顺序存储)
    features_list = [feat_p, feat_a, feat_s]

    # 加载元路径内容邻接矩阵 (scipy csr_matrix -> torch sparse tensor)
    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")
    meta_path_list = [pap, psp]

    # 存储每种节点特征的维数
    in_dims = [features.shape[1] for features in features_list]

    # 加载边类型的映射，edge2type 是一个字典，映射每条边到一个类型(1,2) -> 类型1
    with open(osp.join(path, 'edge2type.pickle'), 'rb') as f:
        edge2type = pickle.load(f)

    print("边的数量1:" + str(len(edge2type)))
    print("边的数量2:" + str(g.number_of_edges()))
    # 确保边的类型映射字典中的条目数量等于图中边的数量
    assert len(edge2type) == g.number_of_edges()

    e_feat = []
    # 构建边的特征列表，每条边的特征由 edge2type 提供
    for u, v in zip(*g.edges()):
        u = u.item()  # 转换为 Python 标量
        v = v.item()
        e_feat.append(edge2type[(u, v)])

    e_feat = th.tensor(e_feat, dtype=th.long).to(device)
    node_types = np.load(osp.join(path, 'node_types.npy'))
    id2type = {i: val for i, val in enumerate(node_types)}
    return features_list, meta_path_list, g, in_dims, edge2type, e_feat, id2type

def load_dblp(type_num: List[int], device: th.device) -> Tuple[
    List[th.Tensor],          # 元路径内容邻接关系列表 (torch sparse tensors)
    List[th.Tensor],          # 特征矩阵列表 (dense tensors)
    List[sp.csr_matrix],      # 元路径矩阵列表 (scipy sparse matrices)
    DGLGraph,                 # DGL图对象
    List[int],                # 各节点类型输入维度
    Dict[Tuple[int, int], int],  # 边类型映射字典
    th.Tensor,                # 边类型特征张量
    Dict[int, int]            # 节点id到类型映射
]:
    path = "../data/dblp/"

    adj = sp.load_npz(osp.join(path, 'adj.npz'))
    # 构建图
    g = dgl.DGLGraph(adj + (adj.T))  # 保证无向图  adj csr
    g = dgl.remove_self_loop(g)  # 去除自环
    g = dgl.add_self_loop(g)  # 添加自环
    g = g.to(device)  # 将图结构移至指定设备

    # 作者特征 (scipy csr_matrix -> torch dense tensor)
    feat_a = sp.load_npz(osp.join(path, "a_feat.npz")).astype("float32")
    feat_a = th.FloatTensor(preprocess_features(feat_a)).to(device)  # shape: [num_authors, feat_dim]

    # 论文特征 (scipy csr_matrix -> torch dense tensor)
    feat_p = sp.load_npz(osp.join(path, "p_feat.npz")).astype("float32")
    feat_p = th.FloatTensor(preprocess_features(feat_p)).to(device)  # shape: [num_papers, feat_dim]

    # 会议特征 (单位矩阵 -> torch dense tensor)
    feat_c = th.eye(type_num[3], dtype=th.float32, device=device)  # shape: [num_confs, num_confs]

    # 术语特征 (numpy数组 -> torch dense tensor)
    feat_t = np.load(path+"t_feat.npz")
    feat_t = th.FloatTensor(feat_t).to(device)  # shape: [num_terms, feat_dim]

    # 特征矩阵列表 (按类型顺序存储)
    features_list = [feat_a, feat_p, feat_t, feat_c]

    apa = sp.load_npz(osp.join(path, "apa.npz"))  # shape: [num_authors, num_authors]
    apcpa = sp.load_npz(osp.join(path, "apcpa.npz"))  # shape: [num_authors, num_authors]
    aptpa = sp.load_npz(osp.join(path, "aptpa.npz"))  # shape: [num_authors, num_authors]
    meta_path_list = [apa, apcpa, aptpa]

    # 存储每种节点特征的维数
    in_dims = [features.shape[1] for features in features_list]

    # 加载边类型的映射，edge2type 是一个字典，映射每条边到一个类型(1,2) -> 类型1
    with open(osp.join(path, 'edge2type.pickle'), 'rb') as f:
        edge2type = pickle.load(f)

    print("边的数量1:" + str(len(edge2type)))
    print("边的数量2:" + str(g.number_of_edges()))
    # 确保边的类型映射字典中的条目数量等于图中边的数量
    assert len(edge2type) == g.number_of_edges()

    e_feat = []
    # 构建边的特征列表，每条边的特征由 edge2type 提供
    for u, v in zip(*g.edges()):
        u = u.item()  # 转换为 Python 标量
        v = v.item()
        e_feat.append(edge2type[(u, v)])

    # 将边特征转换为 PyTorch 张量，并将其传递到指定的设备
    e_feat = th.tensor(e_feat, dtype=th.long).to(device)
    # 加载节点类型的数组，每个节点都有一个类型
    node_types = np.load(osp.join(path, 'node_types.npy'))
    # 节点id到节点type的映射
    id2type = {i: val for i, val in enumerate(node_types)}

    return features_list, meta_path_list, g, in_dims, edge2type, e_feat, id2type



def load_imdb(type_num, device):
    # m a d k
    # 4275 5432 2083 7313
    path = "../data/imdb/"

    adj = sp.load_npz(osp.join(path, 'adj.npz'))
    # 构建图
    g = dgl.DGLGraph(adj + (adj.T))  # 保证无向图  adj csr
    g = dgl.remove_self_loop(g)  # 去除自环
    g = dgl.add_self_loop(g)  # 添加自环
    g = g.to(device)  # 将图结构移至指定设备

    feat_m = sp.load_npz(osp.join(path, "m_feat.npz")).astype("float32")
    feat_m = th.FloatTensor(preprocess_features(feat_m)).to(device)
    feat_a = th.eye(type_num[1], dtype=th.float32, device=device)
    feat_d = th.eye(type_num[2], dtype=th.float32, device=device)
    feat_k = th.eye(type_num[3], dtype=th.float32, device=device)
    # 特征矩阵列表 (按类型顺序存储)
    features_list = [feat_m, feat_a, feat_d, feat_k]
    mam = sp.load_npz(path + 'mam.npz')
    mdm = sp.load_npz(path + 'mdm.npz')
    mkm = sp.load_npz(path + 'mkm.npz')
    meta_path_list = [mam, mdm, mkm]

    # 存储每种节点特征的维数
    in_dims = [features.shape[1] for features in features_list]

    # 加载边类型的映射，edge2type 是一个字典，映射每条边到一个类型(1,2) -> 类型1
    with open(osp.join(path, 'edge2type.pickle'), 'rb') as f:
        edge2type = pickle.load(f)

    print("边的数量1:" + str(len(edge2type)))
    print("边的数量2:" + str(g.number_of_edges()))
    # 确保边的类型映射字典中的条目数量等于图中边的数量
    assert len(edge2type) == g.number_of_edges()

    e_feat = []
    # 构建边的特征列表，每条边的特征由 edge2type 提供
    for u, v in zip(*g.edges()):
        u = u.item()  # 转换为 Python 标量
        v = v.item()
        e_feat.append(edge2type[(u, v)])

    # 将边特征转换为 PyTorch 张量，并将其传递到指定的设备
    e_feat = th.tensor(e_feat, dtype=th.long).to(device)
    # 加载节点类型的数组，每个节点都有一个类型
    node_types = np.load(osp.join(path, 'node_types.npy'))
    # 节点id到节点type的映射
    id2type = {i: val for i, val in enumerate(node_types)}

    return features_list, meta_path_list, g, in_dims, edge2type, e_feat, id2type


def load_self_imdb(type_num, device):
    # m a d k
    # 4275 5432 2083 7313
    path = "../data/self_imdb/"

    adj = sp.load_npz(osp.join(path, 'adj.npz'))
    # 构建图
    g = dgl.DGLGraph(adj + (adj.T))  # 保证无向图  adj csr
    g = dgl.remove_self_loop(g)  # 去除自环
    g = dgl.add_self_loop(g)  # 添加自环
    g = g.to(device)  # 将图结构移至指定设备

    feat_m = sp.load_npz(osp.join(path, "m_feat.npz")).astype("float32")
    feat_m = th.FloatTensor(preprocess_features(feat_m)).to(device)
    feat_a = th.eye(type_num[1], dtype=th.float32, device=device)
    feat_d = th.eye(type_num[2], dtype=th.float32, device=device)
    # 特征矩阵列表 (按类型顺序存储)
    features_list = [feat_m, feat_a, feat_d]

    mam = sp.load_npz(path + 'mam.npz')
    mdm = sp.load_npz(path + 'mdm.npz')
    meta_path_list = [mam, mdm]

    # 存储每种节点特征的维数
    in_dims = [features.shape[1] for features in features_list]

    # 加载边类型的映射，edge2type 是一个字典，映射每条边到一个类型(1,2) -> 类型1
    with open(osp.join(path, 'edge2type.pickle'), 'rb') as f:
        edge2type = pickle.load(f)

    print("边的数量1:" + str(len(edge2type)))
    print("边的数量2:" + str(g.number_of_edges()))
    # 确保边的类型映射字典中的条目数量等于图中边的数量
    assert len(edge2type) == g.number_of_edges()

    e_feat = []
    # 构建边的特征列表，每条边的特征由 edge2type 提供
    for u, v in zip(*g.edges()):
        u = u.item()  # 转换为 Python 标量
        v = v.item()
        e_feat.append(edge2type[(u, v)])

    # 将边特征转换为 PyTorch 张量，并将其传递到指定的设备
    e_feat = th.tensor(e_feat, dtype=th.long).to(device)
    # 加载节点类型的数组，每个节点都有一个类型
    node_types = np.load(osp.join(path, 'node_types.npy'))
    # 节点id到节点type的映射
    id2type = {i: val for i, val in enumerate(node_types)}

    return features_list, meta_path_list, g, in_dims, edge2type, e_feat, id2type

def load_freebase(type_num, device):
    # m a d w
    # 3492, 33401, 2502, 4459
    path = "../data/freebase/"

    adj = sp.load_npz(osp.join(path, 'adj.npz'))
    # 构建图
    g = dgl.DGLGraph(adj + (adj.T))  # 保证无向图  adj csr
    g = dgl.remove_self_loop(g)  # 去除自环
    g = dgl.add_self_loop(g)  # 添加自环
    g = g.to(device)  # 将图结构移至指定设备

    feat_m = th.eye(type_num[0], dtype=th.float32, device=device)
    feat_a = th.eye(type_num[1], dtype=th.float32, device=device)
    feat_d = th.eye(type_num[2], dtype=th.float32, device=device)
    feat_w = th.eye(type_num[3], dtype=th.float32, device=device)
    # 特征矩阵列表 (按类型顺序存储)
    features_list = [feat_m, feat_a, feat_d, feat_w]

    mam = sp.load_npz(path + 'mam.npz')
    mdm = sp.load_npz(path + 'mdm.npz')
    mwm = sp.load_npz(path + 'mwm.npz')
    meta_path_list = [mam, mdm, mwm]

    # 存储每种节点特征的维数
    in_dims = [features.shape[1] for features in features_list]

    with open(osp.join(path, 'edge2type.pickle'), 'rb') as f:
        edge2type = pickle.load(f)

    print("边的数量1:" + str(len(edge2type)))
    print("边的数量2:" + str(g.number_of_edges()))
    # 确保边的类型映射字典中的条目数量等于图中边的数量
    assert len(edge2type) == g.number_of_edges()

    e_feat = []
    # 构建边的特征列表，每条边的特征由 edge2type 提供
    for u, v in zip(*g.edges()):
        u = u.item()  # 转换为 Python 标量
        v = v.item()
        e_feat.append(edge2type[(u, v)])

    # 将边特征转换为 PyTorch 张量，并将其传递到指定的设备
    e_feat = th.tensor(e_feat, dtype=th.long).to(device)
    # 加载节点类型的数组，每个节点都有一个类型
    node_types = np.load(osp.join(path, 'node_types.npy'))
    # 节点id到节点type的映射
    id2type = {i: val for i, val in enumerate(node_types)}

    return features_list, meta_path_list, g, in_dims, edge2type, e_feat, id2type

def load_data(dataset, type_num, device):

    if dataset == "acm":
        feats, adjs, g, in_dims, edge2type, e_feat, id2type = load_acm(type_num, device)
    elif dataset == "dblp":
        feats, adjs, g, in_dims, edge2type, e_feat, id2type = load_dblp(type_num, device)
    elif dataset == "imdb":
        feats, adjs, g, in_dims, edge2type, e_feat, id2type = load_imdb(type_num, device)
    elif dataset == "self_imdb":
        feats, adjs, g, in_dims, edge2type, e_feat, id2type = load_self_imdb(type_num, device)
    elif dataset == "freebase":
        feats, adjs, g, in_dims, edge2type, e_feat, id2type = load_freebase(type_num, device)
    return feats, adjs, g, in_dims, edge2type, e_feat, id2type
