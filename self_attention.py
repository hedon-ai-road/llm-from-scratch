import torch
import torch.nn as nn
from torch import Tensor

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        # 1. 计算 3 个维度的矩阵嵌入向量
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 2. 计算注意力分数
        attn_scores = queries @ keys.T

        # 3. 注意力分数缩放与归一化
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1,
        )

        # 4. 计算上下文向量
        context_vec = attn_weights @ values
        return context_vec


inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],   # Your
        [0.55, 0.87, 0.66],   # journey
        [0.57, 0.85, 0.64],   # starts
        [0.22, 0.58, 0.33],   # with
        [0.77, 0.25, 0.10],   # one
        [0.05, 0.80, 0.55],   # step
    ]
)

# 复制 inputs 来模拟批量输入
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)  # torch.Size([2, 6, 3])

class CausalAttention(nn.Module):
    """一个简化的因果注意力类"""
    def __init__(self, d_in, d_out, context_length,
        dropout, qkv_bias=False) :
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1)
        )

    def forward(self, x):
        # [batch_size, sequence_length, embedding_dim] = [2, 6, 3]
        b, num_tokens, d_in = x.shape

        # 计算权重矩阵
        keys = self.W_key(x)  # [2, 6, 2]
        queries = self.W_query(x) # [2, 6, 2]
        values = self.W_value(x) # [2, 6, 2]

        # keys.transpose(1, 2)：将键矩阵的维度从 [b, num_tokens, d_out] 转换为 [b, d_out, num_tokens] => [2, 6, 2] -> [2, 2, 6]
        attn_scores = queries @ keys.transpose(1 ,2)  # [2, 6, 2] @ [2, 2, 6] = [2, 6, 6]
        
        # 在 Pytorch 中，所有以下划线结尾的操作都会直接作用于元数据，从而减少不必要的内存复制
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        # 进行 dropout
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    """一个实现多头注意力的封装类"""
    def __init__(self, d_in, d_out, context_length,
                dropout, num_heads, qkv_bias=False) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias,
            ) for _ in range (num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


"""
输入: [2, 6, 3]
    ↓
Q,K,V: [2, 6, 2]
    ↓
重塑: [2, 6, 2, 1] (2个头，每个头1维)
    ↓
转置: [2, 2, 6, 1] (2个头并行计算)
    ↓
注意力: [2, 2, 6, 6] (每个头有自己的注意力矩阵)
    ↓
上下文: [2, 2, 6, 1] (每个头的结果)
    ↓
合并: [2, 6, 2] (所有头的结果合并)
    ↓
输出投影: [2, 6, 2]
"""
class MultiHeadAttention(nn.Module):
    """一个高效的多头注意力类"""
    def __init__(self, d_in: int, d_out: int,
                context_length: int, dropout: float, num_heads: int, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out  # 输出维度
        self.num_heads = num_heads # 头数量
        self.head_dim = d_out // num_heads # 每个头的维度
        
        # 初始化可训练的权重矩阵，分别代表查询向量、键向量、值向量
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 使用一个线性层来组合头的输出
        self.out_proj = nn.Linear(d_out, d_out)

        # 掩码 + dropout
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        # [batch_size, sequence_length, embedding_dim]
        b, num_tokens, d_in = x.shape

        # 计算权重矩阵 Q/K/V
        keys: Tensor = self.W_key(x)
        queries: Tensor = self.W_query(x)
        values: Tensor = self.W_value(x)

        # 重塑为多头格式
        # 将 [batch, seq_len, d_out] 重塑为 [batch, seq_len, num_heads, head_dim]
        # [2, 6, 4] -> [2, 6, 2, 2]
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 调整维度顺序，让每个头独立计算注意力，便于批量处理所有头
        # 从形状 (b, num_tokens, num_heads, head_dim)
        # 转换到 (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # 计算注意力分数，这样每个批次(2)的每个头(2)都有了一个 6×6 的注意力分数矩阵
        attn_scores = queries @ keys.transpose(2, 3) # [2, 2, 6, 2] @ [2, 2, 2, 6] = [2, 2, 6, 6]
        mask_bool: Tensor = self.mask.bool()[:num_tokens, :num_tokens]

        # 应用因果掩码
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 归一化权重
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) # [2, 2, 6, 6]

        # 使用 dropout 掩码减少过拟合
        attn_weights = self.dropout(attn_weights) # [2, 2, 6, 6]

        # 每个头计算自己的上下文向量
        context_vec: Tensor = (attn_weights @ values).transpose(1, 2) # [2, 2, 6, 6] @ [2, 2, 6, 2] = [2, 2, 6, 2] -> [2, 6, 2, 2]
        # 重塑回原始格式 [2, 6, 2, 2] -> [2, 6, 4]
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # 通过输出投影层 [2, 6, 4]
        context_vec = self.out_proj(context_vec)
        return context_vec