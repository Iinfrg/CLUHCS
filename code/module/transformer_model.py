import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

def gelu(x):

    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class TransformerModel(nn.Module):
    """
    基于Transformer的多跳特征编码器，用于学习节点在多跳邻域中的语义表示
    主要流程：
    1. 输入特征投影 → 2. Transformer编码 → 3. 层次化注意力聚合 → 4. 输出投影
    """
    def __init__(
            self,
            hops: int,
            input_dim: int,
            n_layers=6,
            num_heads=8,
            hidden_dim=64,
            ffn_dim=64,
            dropout_rate=0.0,
            attention_dropout_rate=0.1
    ):
        super().__init__()
        self.seq_len = hops + 1  # 序列长度 = 跳数 + 1（中心节点）
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = 2 * hidden_dim  # FFN中间层通常扩大一倍
        self.num_heads = num_heads  # 多头注意力头数
        self.n_layers = n_layers  # Transformer编码器层数
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.att_embeddings_nope = nn.Linear(self.input_dim, self.hidden_dim)
        encoders = [
            EncoderLayer(
                self.hidden_dim,
                self.ffn_dim,
                self.dropout_rate,
                self.attention_dropout_rate,
                self.num_heads
            ) for _ in range(self.n_layers)
        ]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)
        # 输出投影层（降维到hidden_dim/2，适应下游任务）
        # self.out_proj = nn.Linear(self.hidden_dim, int(self.hidden_dim / 2))
        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)  # 输入为[中心节点;邻居节点]拼接
        # self.scaling = nn.Parameter(torch.ones(1) * 0.5)
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data: torch.Tensor) -> torch.Tensor:
        tensor = batched_data
        for enc_layer in self.layers:
            tensor = enc_layer(tensor)  # 每层保持 (B, seq_len, hidden_dim)

        output = self.final_ln(tensor)  # (B, seq_len, hidden_dim)
        # 中心节点: 位置0 → (B, 1, hidden_dim)
        # 邻居节点: 位置1~seq_len-1 → (B, seq_len-1, hidden_dim)
        node_tensor = output[:, 0:1, :]  # (B, 1, hidden_dim)
        neighbor_tensor = output[:, 1:, :]  # (B, seq_len-1, hidden_dim)
        # 将中心节点特征复制扩展，与每个邻居对齐
        target = node_tensor.repeat(1, self.seq_len - 1, 1)  # (B, seq_len-1, hidden_dim)
        # 拼接中心节点与邻居节点特征
        combined = torch.cat([target, neighbor_tensor], dim=2)  # (B, seq_len-1, 2*hidden_dim)
        # 计算注意力得分 → (B, seq_len-1, 1)
        layer_atten = self.attn_layer(combined)
        # Softmax归一化 → 每个样本的邻居权重和为1
        layer_atten = F.softmax(layer_atten, dim=1)  # (B, seq_len-1, 1)
        weighted_neighbor = neighbor_tensor * layer_atten  # (B, seq_len-1, hidden_dim)
        aggregated_neighbor = torch.sum(weighted_neighbor, dim=1, keepdim=True)  # (B, 1, hidden_dim)
        fused_output = node_tensor + aggregated_neighbor  # (B, 1, hidden_dim)
        fused_output = fused_output.squeeze(1)  # (B, hidden_dim)
        # output = F.relu(self.out_proj(fused_output))  # (B, hidden_dim//2)
        return fused_output


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):             # (B, seq_len, hidden_dim)
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
