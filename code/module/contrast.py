import torch
import torch.nn.functional as F  # 包含激活函数和相似度计算

# 稀疏矩阵处理（用于pos_adj参数）
from scipy.sparse import coo_matrix

# 高效张量散射操作（处理节点分组计算）
from torch_scatter import scatter_mean

# 类型标注支持（可选但推荐）
from typing import Optional, Tuple

def total_loss(z1: torch.Tensor,
               z2: torch.Tensor,
               pos_adj: coo_matrix,
               tau: float = 0.8,
               lambda_: float = 1,
               alpha: float = 1) -> torch.Tensor:
    # 转换正样本索引到PyTorch张量
    rows = torch.tensor(pos_adj.row, dtype=torch.long, device=z1.device)
    cols = torch.tensor(pos_adj.col, dtype=torch.long, device=z1.device)
    N = z1.size(0)

    # ========== 视图内对比损失 ========== #
    def intra_view_loss(z_src: torch.Tensor, z_tar: torch.Tensor) -> torch.Tensor:
        """计算视图内对比损失，对应公式(3-16)/(3-17)"""
        # 相似度矩阵 [N, N]
        sim = torch.mm(z_src, z_tar.t()) / tau
        # 分母对数项 log(Σ_t exp(sim[i][t])) [N]
        log_denominator = torch.logsumexp(sim, dim=1)
        # 分子项 sim[i][j] [num_edges]
        nume = sim[rows, cols]
        # 单个损失项 log_denominator[i] - nume[i][j] [num_edges]
        losses = -(nume - log_denominator[rows])
        # 按节点i分组求均值后求和 [N] → 标量
        return scatter_mean(losses, rows, dim_size=N).mean()

    # 关系视图内和元路径视图内损失
    L_intra_sv = intra_view_loss(z1, z1)
    L_intra_hv = intra_view_loss(z2, z2)
    L_intra = 0.5 * (L_intra_sv + L_intra_hv)

    # ========== 视图间对比损失 ========== #
    def inter_view_loss(z_src: torch.Tensor, z_tar: torch.Tensor) -> torch.Tensor:
        """计算视图间对比损失，对应公式(3-19)/(3-20)"""
        # 与intra_view_loss逻辑一致，但z_src和z_tar为不同视图
        sim = torch.mm(z_src, z_tar.t()) / tau
        log_denominator = torch.logsumexp(sim, dim=1)
        nume = sim[rows, cols]
        losses = -(nume - log_denominator[rows])
        return scatter_mean(losses, rows, dim_size=N).mean()

    # 跨视图双向损失
    L_inter_sv_hv = inter_view_loss(z1, z2)
    L_inter_hv_sv = inter_view_loss(z2, z1)
    L_inter = 0.5 * (L_inter_sv_hv + L_inter_hv_sv)

    # ========== 无监督对比损失 ========== #
    def unsupervised_loss(z_src: torch.Tensor, z_tar: torch.Tensor) -> torch.Tensor:
        """计算跨视图自身对比损失，对应公式(3-22)"""
        sim = torch.mm(z_src, z_tar.t()) / tau
        log_denominator = torch.logsumexp(sim, dim=1)
        # 对角线元素即 sim(z_i^{sv}, z_i^{hv})
        diag = torch.diag(sim)
        # 损失项 log_denominator[i] - diag[i]
        losses = log_denominator - diag
        return losses.mean()

    # 自身对比双向损失
    L_u_sv_hv = unsupervised_loss(z1, z2)
    L_u_hv_sv = unsupervised_loss(z2, z1)
    L_u = 0.5 * (L_u_sv_hv + L_u_hv_sv)

    # ========== 总损失合成 ========== #
    L_s = L_intra + lambda_ * L_inter
    return L_s + alpha * L_u

def xiaorong_total_loss(z1: torch.Tensor,
               pos_adj: coo_matrix,
               tau: float = 0.8) -> torch.Tensor:
    """
    仅保留视图内对比损失的简化版本
    :param z1: torch.Tensor - 单视图节点嵌入矩阵 [N, D]
    :param pos_adj: coo_matrix - 正样本邻接矩阵
    :param tau: float - 温度系数
    :return: torch.Tensor - 总损失值
    """
    # 转换正样本索引到PyTorch张量
    rows = torch.tensor(pos_adj.row, dtype=torch.long, device=z1.device)
    cols = torch.tensor(pos_adj.col, dtype=torch.long, device=z1.device)
    N = z1.size(0)

    # ========== 视图内对比损失 ========== #
    def intra_view_loss(z_src: torch.Tensor, z_tar: torch.Tensor) -> torch.Tensor:
        """计算视图内对比损失"""
        # 相似度矩阵 [N, N]
        sim = torch.mm(z_src, z_tar.t()) / tau
        # 分母对数项 log(Σ_t exp(sim[i][t])) [N]
        log_denominator = torch.logsumexp(sim, dim=1)
        # 分子项 sim[i][j] [num_edges]
        nume = sim[rows, cols]
        # 单个损失项 log_denominator[i] - nume[i][j] [num_edges]
        losses = -(nume - log_denominator[rows])
        # 按节点i分组求均值后求和 [N] → 标量
        return scatter_mean(losses, rows, dim_size=N).mean()

    # 仅计算单视图的视图内损失
    L_intra = intra_view_loss(z1, z1)

    return L_intra