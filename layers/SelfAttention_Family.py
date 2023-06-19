import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
# from reformer_pytorch import LSHSelfAttention


class FullAttention(nn.Module):
    """
    only for calculate attn = QK/sqrt(d_k) * V
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.mask_flag = mask_flag
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # 这里的输入是Q、K、V经过每个head的线性层后的输出结果。每个头都把形状为bs*len*d_model的向量映射为bs*len*d_k
        # 最后再拼接到一起，就变成了bs * len * head * d_k
        B, L, H, E = queries.shape  # bs * len_query * head * d_k
        _, S, _, D = values.shape  # bs * len_value * head * d_v

        scale = self.scale or 1 / torch.sqrt(E)  # 1/sqrt(d_k)

        # 计算注意力分数，这里实际上先将queries和keys的维度做了变换，再做了矩阵乘法
        scores = torch.einsum('blhe,bshe->bhls', queries, keys)  # Q * K , shape: bs * head * len_q * len_k

        # mask(option)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill(attn_mask.mask, -np.inf)  # bs * head * len_q * len_k

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum('bhls,bshd->blhd', scores, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)  # return V: bs * len * head * d_v


class AttentionLayer(nn.Module):
    """
    multi-head attn, end to end
    """
    def __init__(self, attention, num_heads, d_model, d_k=None, d_v=None):
        super().__init__()

        d_k = d_k or d_model // num_heads
        d_v = d_v or d_model // num_heads

        self.inner_attention = attention

        self.query_projection = nn.Linear(d_model, d_k)
        self.key_projection = nn.Linear(d_model, d_k)
        self.value_projection = nn.Linear(d_model, d_v)

        self.out_projection  = nn.Linear(d_v*num_heads, d_model)
        self.num_heads = num_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = values.shape
        H = self.num_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau=tau, delta=delta)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
