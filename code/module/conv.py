
import torch as th
from torch import nn

from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
import torch.nn.functional as F
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair

class myGATConv(nn.Module):

    def __init__(self,
                 in_feats,          # 输入维度
                 out_feats,         # 输出维度
                 num_heads,         # 头数
                 num_etypes,        # 边类型数
                 feat_drop=0.,
                 attn_drop=0.,
                 alpha=0.2,
                 residual=True):
        super().__init__()
        # 维度注释初始化
        self.num_heads = num_heads  # 注意力头数 H
        self.out_feats = out_feats // num_heads  # 每头特征维度 d = out_feats / H

        # 节点特征投影 (公式中的W)
        self.fc = nn.Linear(in_feats, num_heads * self.out_feats)  # 输入: (N, in_feats) → 输出: (N, H*d)

        # 边类型处理 (公式中的W_e)
        self.fc_edge = nn.Embedding(num_etypes, num_heads * self.out_feats)  # 输入: (E,) → 输出: (E, H*d)

        # 注意力参数 (公式中的a^T)
        self.attn = nn.Parameter(th.FloatTensor(1, num_heads, 3 * self.out_feats))  # 形状: (1, H, 3d)

        # 残差连接
        self.res_fc = nn.Linear(in_feats,
                                num_heads * self.out_feats) if residual else None  # 输入: (N, in_feats) → 输出: (N, H*d)

        # Dropout层
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        # 激活函数
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.reset_parameters()

    def reset_parameters(self):
        """参数初始化"""
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn, gain=gain)
        if self.res_fc is not None:
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def concat_features(self, edges):
        """将源节点、目标节点和边特征拼接"""
        return {
            'concat': th.cat([
                edges.src['h_src'],  # 源节点特征 (E, H, d)
                edges.dst['h_dst'],  # 目标节点特征 (E, H, d)
                edges.data['e']  # 边特征 (E, H, d)
            ], dim=-1)  # 拼接后维度 (E, H, 3d)
        }

    def forward(self, graph, feat, etype_ids):
        with graph.local_scope():
            # 输入特征: (N, in_feats)
            feat = self.feat_drop(feat)  # 维度保持不变

            h = self.fc(feat)
            h = h.view(-1, self.num_heads, self.out_feats)
            e = self.fc_edge(etype_ids)  # 边类型嵌入 → (E, H*d)
            e = e.view(-1, self.num_heads, self.out_feats)

            # 将节点特征存入图结构
            graph.srcdata['h_src'] = h  # 源节点特征 (N, H, d)
            graph.dstdata['h_dst'] = h  # 目标节点特征 (N, H, d)
            graph.edata['e'] = e  # 边特征 (E, H, d)

            # 拼接源节点和目标节点特征 → (E, H, 2d)
            # 使用 lambda 函数手动拼接源节点和目标节点特征
            graph.apply_edges(self.concat_features)
            concat_features = graph.edata['concat']  # 直接获取拼接后的特征
            # graph.apply_edges(fn.u_concat_v('h_src', 'h_dst', 'concat'))
            attn_score = (concat_features * self.attn).sum(dim=-1)

            attn_score = self.leaky_relu(attn_score)

            attn_weights = edge_softmax(graph, attn_score)
            attn_weights = self.attn_drop(attn_weights)

            graph.edata['attn'] = attn_weights.unsqueeze(-1)

            # 消息生成与聚合
            graph.update_all(
                # 消息生成：源节点特征 × 注意力权重
                # 输入: h_src (N, H, d), attn (E, H, 1)
                # 输出: m (E, H, d)
                fn.u_mul_e('h_src', 'attn', 'm'),

                # 消息聚合：按目标节点求和
                # 输入: m (E, H, d)
                # 输出: h (N, H, d)
                fn.sum('m', 'h')
            )

            # 重塑输出维度 → (N, H*d)
            h = graph.dstdata['h'].view(-1, self.num_heads * self.out_feats)

            if self.res_fc is not None:
                # 残差投影 → (N, H*d)
                res = self.res_fc(feat)
                # 残差相加 → (N, H*d)
                h = h + res
            h = F.elu(h)  # 维度保持 [N, H*d]
            return h

