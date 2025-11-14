
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from typing import List, Optional, Tuple, Union
from .preprocess import *
from .encoder import *
from .contrast import *
from .conv import *
import dgl
from dgl import DGLGraph
from .transformer_model import TransformerModel

def re_featuresv2(adj, features, K):

    N, d = features.shape
    nodes_features = torch.empty((N, 1, K + 1, d), device=features.device)

    nodes_features[:, 0, 0, :] = features

    x = features.clone()
    for i in range(K):
        x = torch.matmul(adj, x)  # 矩阵乘法传播特征
        nodes_features[:, 0, i + 1, :] = x  # 存储当前 hop 的特征

    return nodes_features.squeeze(1)


class Attentionl(nn.Module):
    def __init__(self, sub_num: int, hidden_dim: int, attn_drop: float = 0.5):

        super().__init__()
        self.d = hidden_dim
        self.P = sub_num

        # 为每个元路径初始化特征变换器
        self.path_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d, self.d),  # W_phi
                nn.Tanh()  # 激活函数
            ) for _ in range(self.P)
        ])

        # 每个元路径的重要性参数 δ_phi
        self.deltas = nn.ParameterList([
            nn.Parameter(torch.randn(self.d)) for _ in range(self.P)
        ])

        # 正则化组件
        self.dropout = nn.Dropout(attn_drop)

    def forward(self, embeds: list):

        assert len(embeds) == self.P, "输入元路径数量与sub_num不匹配"
        n_nodes = embeds[0].shape[0]

        scores = []
        for i in range(self.P):
            transformed = self.path_transforms[i](embeds[i])

            score = transformed @ self.deltas[i]  # (n, d) × (d,) → (n,)
            scores.append(score.unsqueeze(1))  # 转为列向量 (n, 1)

        score_matrix = torch.cat(scores, dim=1)
        score_matrix = self.dropout(score_matrix)

        alpha = F.softmax(score_matrix, dim=1)

        fused = sum(embeds[i] * alpha[:, i].unsqueeze(1) for i in range(self.P))
        return fused


class CLUHCS(nn.Module):
    def __init__(
            self,
            feats_dim_list: List[int],  # 各节点类型的特征维度列表 [作者dim, 论文dim, ...]
            sub_num: int,  # 元路径数量
            hidden_dim: int,  # 特征映射后的统一维度
            embed_dim: int,  # 最终节点嵌入维度
            tau: float,  # 对比损失温度系数（主）
            tau2: float,  # 对比损失温度系数（辅助）
            adjs: List[torch.Tensor],  # 元路径邻接矩阵列表（已归一化，形状为[N,N]）
            lam_proto: float,  # 原型对比损失权重
            dropout: float,  # Dropout概率（0表示禁用）
            dataset: str,  # 数据集名称（'dblp'或其他）
            pos_adj: 'scipy.sparse.coo_matrix',  # 正样本邻接矩阵（Scipy COO格式）
            g: DGLGraph,        # DGL图G，里面有邻接矩阵
            # 图编码器新增参数（需从外部传入）
            e_feat: torch.Tensor, # 一维边特征
            num_layers: int,  # 图编码器层数
            heads: int,  # 注意力头数
            edge_dim: int,  # 边特征维度
            num_etypes: int,  # 边类型总数
            feat_drop: float = 0.0,  # 特征Dropout概率
            attn_drop: float = 0.0,  # 注意力Dropout概率
            negative_slope: float = 0.2,  # LeakyReLU负斜率
            alpha: float = 0.2,  # GAT计算中的alpha参数
            use_batchnorm: bool = False,  # 是否使用BatchNorm
            # Transformer新增参数（需从外部传入）
            t_hops: int = 2,        # 处理的最大跳数
            t_num_heads: int = 8,   # Transformer中多头自主注意力的层数
            t_n_layers: int = 4     # Transformer的层数
    ):

        super(CLUHCS, self).__init__()
        self.sub_num = sub_num
        self.dataset = dataset
        self.tau = tau
        self.tau2 = tau2
        self.lam_proto = lam_proto
        self.pos_adj = pos_adj
        self.num_layers = num_layers
        self.e_feat = e_feat
        self.g = g
        self.heads = heads
        self.t_hops = t_hops
        self.t_num_heads = t_num_heads
        self.t_n_layers = t_n_layers
        self.hidden_dim = hidden_dim

        self.adj = torch.stack(adjs).mean(axis=0)  # shape: [N, N]

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        agg_num = 7 if dataset == 'dblp' else sub_num

        self.fc_list = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim, bias=True)
            for in_dim in feats_dim_list  # 输入维度对应各节点类型
        ])  # 输出维度统一为hidden_dim


        self.agg = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(agg_num)
        ])


        self.project = nn.Linear(embed_dim, embed_dim)  # 投影到对比空间

        # 每个元路径对应一个Transformer编码器
        self.transformer_list = nn.ModuleList([
            TransformerModel(
                hops=t_hops,  # 处理的最大跳数
                input_dim=hidden_dim,  # 输入特征维度，现在的t_input_dim指的是目标类型的输入时的最初维度
                n_layers=t_n_layers,  # Transformer层数           # 基本5左右
                num_heads=t_num_heads,  # 多头注意力头数
                hidden_dim=self.hidden_dim,  # 隐藏层维度
                dropout_rate=dropout,  # Dropout率
                attention_dropout_rate=dropout
            ) for _ in range(sub_num)  # sub_num个元路径对应sub_num个Transformer
        ])

        # 语义注意力层（融合不同元路径的表示）
        self.sematic_attention = Attentionl(
            sub_num=sub_num,
            hidden_dim=self.hidden_dim,
            attn_drop=dropout
        )

        self.graph_gat_layers = nn.ModuleList()
        self.activation = F.elu  # 激活函数

        for layer in range(0, num_layers):
            self.graph_gat_layers.append(
                myGATConv(
                    in_feats=hidden_dim,
                    out_feats=hidden_dim,
                    num_heads=self.heads,
                    num_etypes=num_etypes,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    alpha=alpha
                )
            )
        
    def forward(self, feats, adjs_norm):

        # 投影
        projection_feat_list = self.projection(feats)

        # K-hop传播后的特征 每个张量的形状是(N, K+1, d)
        multi_hop_features = [re_featuresv2(adj_n, projection_feat_list[0], self.t_hops) for adj_n in adjs_norm]

        z_mp_list = []
        for i in range(len(multi_hop_features)):  # 遍历每个元路径
            # 通过对应Transformer并调整维度
            trans_out = self.transformer_list[i](multi_hop_features[i])
            # proj_out = self.att_embeddings_proj(trans_out)
            z_mp_list.append(trans_out)

        # 语义注意力融合（加权平均不同元路径的表示）
        z_transformer = self.sematic_attention(z_mp_list)

        x = torch.cat(projection_feat_list, 0)  # 拼接    所有类型节点的数量 * 隐藏层维度
        node_sum = x.size(0)        # 节点数量
        for l in range(self.num_layers):
            x = self.graph_gat_layers[l](self.g, x, self.e_feat)
            z = x.view(node_sum, -1)    # 图编码器的最终嵌入
            x = self.dropout(z)

        z1 = z[:feats[0].shape[0], :]           # 主节点特征

        self.z1_pos1 = z1
        loss = self.contrast_loss(z1, z_transformer)
        print('total loss: %f' % loss)
        return loss

    def contrast_loss(self, z1, z_transformer):

        # 投影到对比空间
        z2 = F.tanh(self.project(z1))
        # L2归一化（用于余弦相似度计算）
        z2 = F.normalize(z2, dim=1)

        # 投影到对比空间
        z_transformer_2 = F.tanh(self.project(z_transformer))
        # L2归一化（用于余弦相似度计算）
        z_transformer_2 = F.normalize(z_transformer_2, dim=1)

        # 总损失函数
        loss = total_loss(z2, z_transformer_2, self.pos_adj, self.tau)
        # loss = xiaorong_total_loss( z_transformer_2, self.pos_adj, self.tau)
        self.z_pos = z_transformer

        # 消融元路径视图
        # loss = xiaorong_total_loss(z1, self.pos_adj, self.tau)
        # self.z_pos = z1

        # 消融关系视图
        # loss = xiaorong_total_loss(z_transformer_2, self.pos_adj, self.tau)
        # self.z_pos = z_transformer


        return loss

    def get_z1_embeds(self):
        if self.z1_pos1 is None:
            raise ValueError("z1尚未生成！需先执行model(feats, adjs_norm)前向传播")
        # 转CPU→解耦梯度→转numpy（避免CUDA张量问题，且numpy格式便于保存）
        return self.z1_pos1.detach().cpu().numpy()

    def projection(self, feats):
        # 投影
        # 投影后的特征矩阵列表
        projection_feat_list = []

        for i in range(0,len(feats)):
            h_zzz = F.elu(self.dropout(self.fc_list[i](feats[i])))
            projection_feat_list.append(h_zzz)

        return projection_feat_list


    def get_embeds(self):

        # # # 元路径邻接矩阵存储
        # A = self.adjs_norm.numpy()  # 将PyTorch张量转回numpy数组
        # A = np.where(A > 0, 1, 0)  # 二值化处理（根据实际需求调整阈值）
        # x, y = np.nonzero(A)
        # value = np.ones(len(x))  # 直接生成全1数组，避免循环
        # metapath_matrix = sp.coo_matrix((value, (x, y)), shape=A.shape)
        # sp.save_npz("../data/" + self.dataset + "/final_meta_path11.npz", metapath_matrix)
        return self.z_pos.detach()
