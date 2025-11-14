
import math
import torch
import torch.nn as nn


class GraphConv(nn.Module):

    def __init__(self, input_dim, output_dim, gc_drop):

        super(GraphConv, self).__init__()

        # 初始化可学习权重矩阵
        weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.weight = self.reset_parameters(weight)

        # 配置Dropout层
        if gc_drop:
            self.gc_drop = nn.Dropout(gc_drop)
        else:
            self.gc_drop = lambda x: x

        # 使用PReLU作为默认激活函数（可被forward参数覆盖）
        self.act = nn.PReLU()

    def reset_parameters(self, weight):
        """Xavier风格的均匀分布初始化"""
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)
        return weight

    def forward(self, x, adj, activation=None):

        # 特征线性变换 + Dropout
        x_hidden = self.gc_drop(torch.mm(x, self.weight))       # (n, input_dim) -> (n, output_dim)
        x = torch.spmm(adj, x_hidden)           # (n, n) * (n, output_dim) -> (n, output_dim)

        # 应用激活函数（允许外部指定）
        if activation is None:
            outputs = self.act(x)
        else:
            outputs = activation(x)
        return outputs


class GCN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):

        super(GCN, self).__init__()
        # 构建两层图卷积网络
        self.gc1 = GraphConv(input_dim, hidden_dim, dropout)        # 含激活函数
        self.gc2 = GraphConv(hidden_dim, output_dim, dropout)       # 无激活函数

    def forward(self, feat, adj):

        hidden = self.gc1(feat, adj)
        Z = self.gc2(hidden, adj, activation=lambda x: x)           # (n, hidden_dim) -> (n, output_dim)
        layernorm = nn.LayerNorm(Z.size(), eps=1e-05, elementwise_affine=False)
        outputs = layernorm(Z)
        return outputs

class Attention(nn.Module):

    def __init__(self, hidden_dim, attn_drop=0):

        super(Attention, self).__init__()
        # 特征变换层（将输入映射到注意力空间）
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)      # 适配Tanh的权值初始化

        self.tanh = nn.Tanh()                                   # 非线性激活

        # 可学习的注意力向量（用于计算各嵌入的权重）
        self.att = nn.Parameter(torch.empty(
            size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        # 权重归一化层（需注意未指定dim可能导致的问题）
        self.softmax = nn.Softmax()

        # 注意力权重Dropout
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):

        beta = []
        attn_curr = self.attn_drop(self.att)  # (1, hidden_dim)
        for embed in embeds:
            # MLP + 平均池化
            sp = self.tanh(self.fc(embed)).mean(dim=0)      # (N, hidden_dim) -> (hidden_dim,)
            # 标量列表    计算得分：(1, hidden_dim) @ (hidden_dim, 1) -> 标量
            beta.append(attn_curr.matmul(sp.t()))

        # 拼接所有得分并归一化
        beta = torch.cat(beta, dim=-1).view(-1)     # (K,) K为嵌入数量，即元路径数量
        beta = self.softmax(beta)                   # 权重归一化（建议显式dim=0）
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i]*beta[i]
        return z_mp