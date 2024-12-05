import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from gnova.modules.gnova_encoder_layer import DeepNorm, FlashQKV, FFNGLU


class MaskedSelfRelation(nn.Module):
    def __init__(self,
                 tgt_hidden_size: int,
                 num_heads: int,
                 d_relation: int,
                 alpha: float,
                 beta: float,
                 dropout_rate: float):
        """_summary_

        Args:
            hidden_size (int): same with transformer hidden_size
            d_relation (int): relation matrix dimention
            num_head (int): same with transformer num_head
            layer_num (int): How many layers in total
        """
        super().__init__()
        self.tgt_hidden_size = tgt_hidden_size
        assert self.tgt_hidden_size//d_relation*d_relation == self.tgt_hidden_size
        self.num_kv_heads = num_heads
        self.head_dim = d_relation // num_heads
        assert self.head_dim * self.num_kv_heads == d_relation
        self.num_q_heads = self.tgt_hidden_size // self.head_dim

        self.d_relation = d_relation
        assert self.d_relation % 8 == 0

        self.linear_q = nn.Linear(tgt_hidden_size, tgt_hidden_size)
        self.linear_k = nn.Linear(tgt_hidden_size, self.d_relation)
        self.linear_v = nn.Linear(tgt_hidden_size, self.d_relation)

        self.output_layer = nn.Linear(tgt_hidden_size, tgt_hidden_size)

        self.dn = DeepNorm(tgt_hidden_size, alpha, dropout_rate)
        
        #根据DeepNet，对初始化值做修正.
        nn.init.xavier_normal_(self.linear_v.weight, gain=beta)
        nn.init.xavier_normal_(self.output_layer.weight, gain=beta)

        self.self_attention = FlashQKV(softmax_scale=1.0 / math.sqrt(self.head_dim), attention_dropout=dropout_rate, causal=True)

    def forward(self, tgt):
        """_summary_

        Args:
            tgt (Tensor): aa information from last layer

        Returns:
            tgt (Tensor): tgt information from this layer
        """
        batch_size = tgt.size(0)

        tgt_q = self.linear_q(tgt).view(batch_size, -1, self.num_q_heads, self.head_dim)
        tgt_k = self.linear_k(tgt).view(batch_size, -1, self.num_kv_heads, self.head_dim)
        tgt_v = self.linear_v(tgt).view(batch_size, -1, self.num_kv_heads, self.head_dim)

        post_node = self.self_attention(tgt_q, tgt_k, tgt_v)  # (batch_size, seq_len, num_q_heads, head_dim)
        post_node = post_node.flatten(2, 3)
        post_node = self.output_layer(post_node)
        post_tgt = self.dn(tgt, post_node)
        return post_tgt # [b, q_len, tgt_hidden_size]

class TransRelation(nn.Module):
    def __init__(self,
                 tgt_hidden_size: int,
                 mem_hidden_size: int,
                 num_heads: int,
                 d_relation: int,
                 alpha: float,
                 beta: float,
                 dropout_rate: float):
        """_summary_

        Args:
            hidden_size (int): same with transformer hidden_size
            d_relation (int): relation matrix dimention
            num_head (int): same with transformer num_head
            layer_num (int): How many layers in total
        """
        super().__init__()
        self.tgt_hidden_size = tgt_hidden_size
        assert self.tgt_hidden_size//d_relation*d_relation == self.tgt_hidden_size

        self.num_kv_heads = num_heads
        self.head_dim = d_relation // num_heads
        assert self.head_dim * self.num_kv_heads == d_relation
        self.num_q_heads = self.tgt_hidden_size // self.head_dim

        self.d_relation = d_relation
        assert self.d_relation % 8 == 0

        self.linear_q = nn.Linear(tgt_hidden_size, tgt_hidden_size)
        self.linear_k = nn.Linear(mem_hidden_size, self.d_relation)
        self.linear_v = nn.Linear(mem_hidden_size, self.d_relation)

        self.output_layer = nn.Linear(tgt_hidden_size, tgt_hidden_size)

        self.dn = DeepNorm(tgt_hidden_size, alpha, dropout_rate)
        
        #根据DeepNet，对初始化值做修正.
        nn.init.xavier_normal_(self.linear_q.weight)
        nn.init.xavier_normal_(self.linear_k.weight)
        nn.init.xavier_normal_(self.linear_v.weight, gain=beta)
        nn.init.xavier_normal_(self.output_layer.weight, gain=beta)

        self.trans_attention = FlashQKV(softmax_scale=1.0 / math.sqrt(self.head_dim), attention_dropout=dropout_rate)

    def forward(self, tgt, step_mass_embed, mem, peak_mzs_embed, neg_peak_mzs_embed):
        """_summary_

        Args:
            node (Tensor): node information from last layer
            edge (Tensor): edge information from edge encoder
            drctn (IntTensor): direction mark
            rel_mask (Tensor): relation mask for ignore some pair of nodes which don't have any connection

        Returns:
            node (Tensor): node information from last layer
        """
        batch_size = tgt.size(0)
        
        tgt_q = self.linear_q(tgt).view(batch_size, -1, self.num_q_heads, self.head_dim)
        mem_k = self.linear_k(mem).view(batch_size, -1, self.num_kv_heads, self.head_dim)
        mem_v = self.linear_v(mem).view(batch_size, -1, self.num_kv_heads, self.head_dim)

        q = self.apply_rope(tgt_q, step_mass_embed, self.num_q_heads)

        k_minus = mem_k[:, :, 0::2, :]
        k_plus = mem_k[:, :, 1::2, :]

        k_minus = self.apply_rope(k_minus, peak_mzs_embed, self.num_kv_heads // 2)
        k_plus = self.apply_rope(k_plus, neg_peak_mzs_embed, self.num_kv_heads // 2)
        k = torch.cat([k_minus, k_plus], dim=-2)

        v = mem_v

        post_node = self.trans_attention(q, k, v)  # (batch_size, q_len, num_q_heads, head_dim)
        post_node = post_node.flatten(2, 3)
        mem_v = self.output_layer(post_node)
        mem_v = self.dn(tgt, mem_v)
        return mem_v # [b, q_len, tgt_hidden_size]

    @staticmethod
    def apply_rope(x, dis, num_heads):
        dis_sin, dis_cos = dis.chunk(2,dim=-1)
        dis_cos = dis_cos.unsqueeze(-2).repeat((1, 1, num_heads, 1))
        dis_sin = dis_sin.unsqueeze(-2).repeat((1, 1, num_heads, 1))
        x0, x1 = x[..., 0::2], x[..., 1::2]
        return torch.concat([x0*dis_cos-x1*dis_sin,\
                             x1*dis_cos+x0*dis_sin], dim=-1)

class GNovaDecoderLayer(nn.Module):
    def __init__(
            self, tgt_hidden_size: int,
            mem_hidden_size: int,
            num_heads: int,
            d_relation: int,
            alpha,
            beta,
            dropout_rate: float):

        super().__init__()
        self.self_relation = MaskedSelfRelation(tgt_hidden_size, num_heads, d_relation, alpha, beta, dropout_rate)
        self.trans_relation = TransRelation(tgt_hidden_size, mem_hidden_size, num_heads, d_relation, alpha, beta, dropout_rate)
        self.ffn = FFNGLU(tgt_hidden_size, alpha, beta, dropout_rate)

    def forward(self, *, tgt, step_mass_ebed, mem, peak_mzs_embed, neg_peak_mzs_embed):
        tgt = checkpoint(self.self_relation, tgt, use_reentrant=False)
        tgt = checkpoint(self.trans_relation, tgt, step_mass_ebed, mem, peak_mzs_embed, neg_peak_mzs_embed, use_reentrant=False)
        tgt = checkpoint(self.ffn, tgt, use_reentrant=False)
        return tgt
