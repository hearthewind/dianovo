import torch
from flash_attn import flash_attn_func
from torch import nn
from torch.utils.checkpoint import checkpoint
from rnova.module.encoder_layer import FFNGLU



class MaskedSelfMultiHeadAttn(nn.Module):
    def __init__(self, cfg):
        """_summary_

        Args:
            hidden_size (int): same with transformer hidden_size
            num_head (int): same with transformer num_head
        """
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.decoder.hidden_size
        self.num_heads = cfg.decoder.num_heads
        assert self.hidden_size % self.num_heads == 0
        self.head_size = self.hidden_size // self.num_heads

        self.linear_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_v = nn.Linear(self.hidden_size, self.hidden_size)


        self.output_layer = nn.Linear(self.hidden_size, self.hidden_size)

        self.ln = nn.LayerNorm(self.hidden_size)


    def forward(self, tgt, step_mass_embed):
        """_summary_

        Args:
            tgt (Tensor): aa information from last layer

        Returns:
            tgt (Tensor): tgt information from this layer
        """
        batch_size = tgt.size(0)

        tgt_q = self.linear_q(tgt).view(batch_size, -1, self.num_heads, self.head_size)
        tgt_k = self.linear_k(tgt).view(batch_size, -1, self.num_heads, self.head_size)
        tgt_v = self.linear_v(tgt).view(batch_size, -1, self.num_heads, self.head_size)

        tgt_q, tgt_k = self.apply_rope(tgt_q, step_mass_embed), self.apply_rope(tgt_k, step_mass_embed)

        post_node = flash_attn_func(tgt_q, tgt_k, tgt_v, causal=True)  # (batch_size, seq_len, num_q_heads, head_dim)
        post_node = post_node.flatten(2, 3)
        post_node = self.output_layer(post_node)
        post_tgt = self.ln(tgt + post_node)
        return post_tgt  # [b, q_len, tgt_hidden_size]

    @staticmethod
    def apply_rope(x, dis):
        dis_sin, dis_cos = dis.chunk(2, dim=-1)
        x0, x1 = x[..., 0::2], x[..., 1::2]
        return torch.concat([x0 * dis_cos - x1 * dis_sin, \
                             x1 * dis_cos + x0 * dis_sin], dim=-1)


class TransMultiHeadAttn(nn.Module):
    def __init__(self,
                 cfg):
        """_summary_

        Args:
            cfg: config
        """
        super().__init__()
        self.cfg = cfg
        self.tgt_hidden_size = cfg.decoder.hidden_size
        self.num_heads = cfg.decoder.num_heads
        assert self.tgt_hidden_size % self.num_heads == 0
        self.head_size = self.tgt_hidden_size // self.num_heads

        self.mem_hidden_size = cfg.encoder.hidden_size
        self.linear_q = nn.Linear(self.tgt_hidden_size, self.tgt_hidden_size)
        self.linear_k = nn.Linear(self.mem_hidden_size, self.tgt_hidden_size)
        self.linear_v = nn.Linear(self.mem_hidden_size, self.tgt_hidden_size)

        self.output_layer = nn.Linear(self.tgt_hidden_size, self.tgt_hidden_size)

        self.ln = nn.LayerNorm(self.tgt_hidden_size)

    def forward(self, tgt, step_mass_embed, mem, peak_mzs_embed):
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

        tgt_q = self.linear_q(tgt).view(batch_size, -1, self.num_heads, self.head_size)
        mem_k = self.linear_k(mem).view(batch_size, -1, self.num_heads, self.head_size)
        mem_v = self.linear_v(mem).view(batch_size, -1, self.num_heads, self.head_size)

        tgt_q = self.apply_rope(tgt_q, step_mass_embed)

        mem_k = self.apply_rope(mem_k, peak_mzs_embed)

        post_node = flash_attn_func(tgt_q, mem_k, mem_v)  # (batch_size, q_len, num_q_heads, head_dim)
        post_node = post_node.flatten(2, 3)
        post_node = self.output_layer(post_node)
        post_node = self.ln(tgt + post_node)
        return post_node  # [b, q_len, tgt_hidden_size]

    @staticmethod
    def apply_rope(x, dis):
        dis_sin, dis_cos = dis.chunk(2, dim=-1)
        x0, x1 = x[..., 0::2], x[..., 1::2]
        return torch.concat([x0 * dis_cos - x1 * dis_sin, \
                             x1 * dis_cos + x0 * dis_sin], dim=-1)

class RNovaDecoderLayer(nn.Module):
    def __init__(self, cfg):

        super().__init__()
        self.self_relation = MaskedSelfMultiHeadAttn(cfg)
        self.trans_relation = TransMultiHeadAttn(cfg)

        tgt_hidden_size = cfg.decoder.hidden_size
        self.ffn = FFNGLU(tgt_hidden_size)

    def forward(self, *, tgt, step_mass_embed, mem, peak_mzs_embed):
        tgt = checkpoint(self.self_relation, tgt, step_mass_embed, use_reentrant=False)
        tgt = checkpoint(self.trans_relation, tgt, step_mass_embed, mem, peak_mzs_embed, use_reentrant=False)
        tgt = checkpoint(self.ffn, tgt, use_reentrant=False)
        return tgt
