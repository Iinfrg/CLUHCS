
import numpy as np
import scipy.sparse as sp
import random
import torch
from heapq import nlargest

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def normalize_adj(adj):

    adj = sp.coo_matrix(adj)
    rowsum = adj.sum(axis=1)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_norm_coo = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo().todense()

    adj_torch = torch.from_numpy(adj_norm_coo).float()
    if torch.cuda.is_available():
        adj_torch = adj_torch.cuda()
    return adj_torch


def preprocess_adj(adjs):
    """对多个邻接矩阵执行标准预处理: 去自环 -> 重建自环 -> 对称归一化

    1. 去自环 -> 2. 重建自环 -> 3. 对称归一化
    典型应用: 图神经网络的标准预处理流程
    """
    processed_adjs = []  # 存储处理后的邻接矩阵容器

    for adj in adjs:
        # 1. 移除原始自环边(确保后续添加的是标准化自环)
        adj_no_selfloop = adj - sp.dia_matrix(
            (adj.diagonal()[np.newaxis, :], [0]),
            shape=adj.shape
        )
        adj_no_selfloop.eliminate_zeros()  # 压缩存储格式

        # 2. 重建带自环的邻接矩阵
        # 直接添加单位矩阵(确保每个节点有标准化自连接)
        adj_with_selfloop = adj_no_selfloop + sp.eye(adj.shape[0])

        # 3. 对称归一化处理(D^{-1/2} A D^{-1/2})
        adj_normalized = normalize_adj(adj_with_selfloop)

        processed_adjs.append(adj_normalized)

    return processed_adjs

def mask_feature(feat, adj_mask):
    feats_coo = sp.coo_matrix(feat)
    feats_num = feats_coo.getnnz()
    feats_idx = [i for i in range(feats_num)]
    mask_num = int(feats_num * adj_mask)
    mask_idx = random.sample(feats_idx, mask_num)
    feats_data = feats_coo.data
    for j in mask_idx:
        feats_data[j] = 0
    mask_feats = torch.sparse.FloatTensor(torch.LongTensor([feats_coo.row.tolist(), feats_coo.col.tolist()]), torch.FloatTensor(feats_data.astype(np.float)))

    if torch.cuda.is_available():
        mask_feats = mask_feats.cuda()
    return mask_feats


def pathsim(adjs, u):
    # 阶段1: 格式统一处理
    # 将所有输入矩阵强制转换为COO格式（与原始top_adjs格式对齐）
    adjs = [adj.tocoo() for adj in adjs]

    # 阶段2: 构建正样本邻接矩阵
    combined_sim_matrix = np.zeros(adjs[0].shape)
    for adj in adjs:
        A = adj.toarray()  # 转换为密集矩阵计算相似度
        rows, cols = adj.nonzero()

        # 计算路径相似度
        values = []
        for i, j in zip(rows, cols):
            values.append(2 * A[i, j] / (A[i, i] + A[j, j]))

        # 累加到全局相似度矩阵
        combined_sim_matrix += sp.coo_matrix(
            (values, (rows, cols)),
            shape=A.shape
        ).toarray()

    # 计算平均相似度
    combined_sim_matrix /= len(adjs)

    # 阶段3: 生成正样本矩阵
    # 生成满足条件的行列索引（i≠j）
    rows, cols = np.where(combined_sim_matrix >= u)
    # 过滤自环（i == j的边）
    non_self_mask = (rows != cols)  # 生成布尔掩码，True表示非自环
    rows_filtered = rows[non_self_mask]
    cols_filtered = cols[non_self_mask]

    # 构建正样本邻接矩阵（权重为1表示正样本关系）
    pos_adj = sp.coo_matrix(
        (np.ones_like(rows_filtered), (rows_filtered, cols_filtered)),
        shape=combined_sim_matrix.shape,
        dtype=np.int32
    )

    return adjs, pos_adj
def pathsim(adjs, u):

    adjs = [adj.tocoo() for adj in adjs]

    combined_sim_matrix = np.zeros(adjs[0].shape)

    for adj in adjs:
        A = adj.toarray()  # 转换为密集矩阵计算相似度
        rows, cols = adj.nonzero()

        # 计算路径相似度
        values = []
        for i, j in zip(rows, cols):
            values.append(2 * A[i, j] / (A[i, i] + A[j, j]))

        # 累加到全局相似度矩阵
        combined_sim_matrix += sp.coo_matrix(
            (values, (rows, cols)),
            shape=A.shape
        ).toarray()

    # 计算平均相似度
    combined_sim_matrix /= len(adjs)

    rows, cols = np.where(combined_sim_matrix >= u)
    # 过滤自环（i == j的边）
    non_self_mask = (rows != cols)  # 生成布尔掩码，True表示非自环
    rows_filtered = rows[non_self_mask]
    cols_filtered = cols[non_self_mask]

    # 构建正样本邻接矩阵（权重为1表示正样本关系）
    pos_adj = sp.coo_matrix(
        (np.ones_like(rows_filtered), (rows_filtered, cols_filtered)),
        shape=combined_sim_matrix.shape,
        dtype=np.int32
    )

    return adjs, pos_adj

def xiaorong_pathsim(adjs, u):

    adjs = [adj.tocoo() for adj in adjs]

    all_rows = np.concatenate([adj.row for adj in adjs])
    all_cols = np.concatenate([adj.col for adj in adjs])

    coords = np.column_stack((all_rows, all_cols))
    unique_coords = np.unique(coords, axis=0)
    non_self_mask = unique_coords[:, 0] != unique_coords[:, 1]
    filtered_coords = unique_coords[non_self_mask]

    exist_adj = sp.coo_matrix(
        (np.ones(len(filtered_coords), dtype=np.int32),
         (filtered_coords[:, 0], filtered_coords[:, 1])),
        shape=adjs[0].shape,
        dtype=np.int32
    )

    return adjs, exist_adj


def xiaorong_pathsim_zero(adjs, u):

    adjs = [adj.tocoo() for adj in adjs]

    exist_adj = sp.coo_matrix(
        (np.zeros(0, dtype=np.int32),  # 空数据数组（全零）
         (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32))),  # 空行列索引
        shape=adjs[0].shape,
        dtype=np.int32
    )

    return adjs, exist_adj