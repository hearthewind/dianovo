import math
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn import flash_attn_varlen_func as attn_unpadded_func # This is the equivalent for flash attention 2
from torch.nn.attention import sdpa_kernel, SDPBackend

from torch.utils.checkpoint import checkpoint

class DeepNorm(nn.Module):
    def __init__(self, normalized_shape, alpha, dropout_rate) -> None:
        super().__init__()
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout_rate)
        self.ln = nn.LayerNorm(normalized_shape)
    
    def forward(self, x, postx):
        return self.ln(x*self.alpha + self.dropout(postx))

class FlashQKV(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, softmax_scale=None, attention_dropout=0.0, causal=False, device='gpu'):
        super().__init__()
        assert attn_unpadded_func is not None, "FlashAttention is not installed"
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.causal = causal
        self.device = device

    def forward(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value.
                q has shape (B, S, Hq, D).
                k, v has shape (B, S, Hk, D).

                Hq = c * Hk for some c
        Returns:
        --------
            out:  (B, S, Hq, D).
        """
        # assert q.dtype in [torch.float16, torch.bfloat16]
        # assert q.is_cuda #TODO(m) no need for these check anymore

        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]

        if self.device == 'gpu':
            q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device = q.device)
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k.device)

            post_v = attn_unpadded_func(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                seqlen_q, seqlen_k,
                self.drop.p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
            )

            post_v = rearrange(post_v, '(b s) ... -> b s ...', b=batch_size)
        elif self.device == 'cpu':
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                post_v = F.scaled_dot_product_attention(q, k, v,
                                                    attn_mask=None,
                                                    dropout_p=self.drop.p if self.training else 0.0,
                                                    is_causal=self.causal,
                                                    scale=self.softmax_scale,
                                                    enable_gqa=True)
            post_v = post_v.permute(0, 2, 1, 3)

        return post_v

class MultiHeadRelation(nn.Module):
    def __init__(self, 
                 hidden_size: int,
                 num_heads: int,
                 d_relation: int,
                 alpha: float,
                 beta: float,
                 dropout_rate: float,
                 device='gpu'):
        super().__init__()
        self.d_relation = d_relation
        self.num_kv_heads = num_heads
        self.head_dim = d_relation // num_heads
        assert self.head_dim * self.num_kv_heads == d_relation
        self.hidden_size = hidden_size
        assert self.d_relation % 8 == 0
        assert self.hidden_size % self.d_relation == 0
        self.num_q_heads = self.hidden_size // self.head_dim

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, self.d_relation)
        self.linear_v = nn.Linear(hidden_size, self.d_relation)
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.dn = DeepNorm(hidden_size, alpha, dropout_rate)
        
        #根据DeepNet，对初始化值做修正.
        nn.init.xavier_normal_(self.linear_v.weight, gain=beta)
        nn.init.xavier_normal_(self.output_layer.weight, gain=beta)

        self.self_attention = FlashQKV(softmax_scale=1.0/math.sqrt(self.head_dim), attention_dropout=dropout_rate, device=device)

    def forward(self, peak_features, peak_mzs_embed, neg_peak_mzs_embed):
        peak_num = peak_features.size(1)
        q, k = self.linear_q(peak_features), self.linear_k(peak_features)
        q = q.reshape(-1, peak_num, self.num_q_heads, self.head_dim)
        k = k.reshape(-1, peak_num, self.num_kv_heads, self.head_dim)

        q_minus = q[:, :, 0::2, :]
        q_plus = q[:, :, 1::2, :]

        q_minus = self.apply_rope(q_minus, peak_mzs_embed, self.num_q_heads // 2)
        q_plus = self.apply_rope(q_plus, neg_peak_mzs_embed, self.num_q_heads // 2)
        q = torch.cat([q_minus, q_plus], dim=-2)

        k = self.apply_rope(k, peak_mzs_embed, self.num_kv_heads)

        v = self.linear_v(peak_features).view(-1, peak_num, self.num_kv_heads, self.head_dim)

        post_node = self.self_attention(q, k, v)  # (batch_size, seq_len, num_q_heads, head_dim)
        post_node = post_node.flatten(2,3)
        post_node = self.output_layer(post_node)
        peak_features = self.dn(peak_features, post_node)
        return peak_features
    @staticmethod
    def apply_rope(x, dis, num_heads):
        dis_sin, dis_cos = dis.chunk(2,dim=-1)
        dis_cos = dis_cos.unsqueeze(-2).repeat((1, 1, num_heads, 1))
        dis_sin = dis_sin.unsqueeze(-2).repeat((1, 1, num_heads, 1))
        x0, x1 = x[..., 0::2], x[..., 1::2]
        return torch.concat([x0*dis_cos-x1*dis_sin,\
                             x1*dis_cos+x0*dis_sin], dim=-1)

class FFNGLU(nn.Module):
    def __init__(self, hidden_size: int, alpha: float, beta: float, dropout_rate: float):
        super().__init__()
        self.pre_ffn_gate = nn.Sequential(nn.Linear(hidden_size, 4*hidden_size, bias=False),
                                          nn.ReLU()
                                          )
        self.pre_ffn = nn.Linear(hidden_size, 4*hidden_size, bias=False)
        self.post_ffn = nn.Linear(4*hidden_size, hidden_size, bias=False)
        self.dn = DeepNorm(hidden_size, alpha, dropout_rate)
        
        #根据DeepNet，对初始化值做修正.
        nn.init.xavier_normal_(self.pre_ffn_gate[0].weight, gain=beta)
        nn.init.xavier_normal_(self.pre_ffn.weight, gain=beta)
        nn.init.xavier_normal_(self.post_ffn.weight, gain=beta)

    def forward(self, x):
        postx = self.post_ffn(self.pre_ffn_gate(x)*self.pre_ffn(x))
        x = self.dn(x, postx)
        return x

class GNovaEncoderLayer(nn.Module):
    def __init__(self, 
                 hidden_size: int,
                 num_heads: int,
                 d_relation: int,
                 alpha: float, 
                 beta: float, 
                 dropout_rate: float,
                 device='gpu'):

        super().__init__()
        self.relation = MultiHeadRelation(hidden_size, num_heads, d_relation, alpha, beta, dropout_rate, device)
        self.ffn = FFNGLU(hidden_size, alpha, beta, dropout_rate)

    def forward(self, peak_features, peak_mzs_embed, neg_peak_mzs_embed):
        peak_features = checkpoint(self.relation, peak_features, peak_mzs_embed, neg_peak_mzs_embed, use_reentrant=False)
        peak_features = checkpoint(self.ffn, peak_features, use_reentrant=False)
        return peak_features
