import torch
import torch.nn as nn
from flash_attn import flash_attn_func
from torch.utils.checkpoint import checkpoint

class SelfMultiHeadAttn(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int):
        super().__init__()
        assert hidden_size%num_heads == 0
        self.num_heads = num_heads
        self.head_size = hidden_size//num_heads
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self,x,pos):
        batch_size = x.size(0)
        q, k, v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        q = q.view(batch_size, -1, self.num_heads, self.head_size)
        k = k.view(batch_size, -1, self.num_heads, self.head_size)
        v = v.view(batch_size, -1, self.num_heads, self.head_size)
        q, k = self.apply_rope(q, pos), self.apply_rope(k, pos)
        postx = flash_attn_func(q, k, v).flatten(2,3)
        postx = self.output_layer(postx)
        x = self.ln(x + postx)
        return x
    
    @staticmethod
    def apply_rope(x, dis):
        dis_sin, dis_cos = dis.chunk(2,dim=-1)
        x0, x1 = x[..., 0::2], x[..., 1::2]
        return torch.concat([x0*dis_cos-x1*dis_sin,\
                             x1*dis_cos+x0*dis_sin], dim = -1)

class FFNGLU(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

        # 根据“GLU Variants Improve Transformer”，采用GEGLU结构做FFN.
        assert hidden_size>0
        self.pre_ffn_gate = nn.Sequential(nn.Linear(hidden_size, 4*hidden_size, bias=False),
                                          nn.ReLU(inplace=True)
                                          )
        self.pre_ffn = nn.Linear(hidden_size, 4*hidden_size, bias=False)
        self.post_ffn = nn.Linear(4*hidden_size, hidden_size, bias=False)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        gate = self.pre_ffn_gate(x)
        v = self.pre_ffn(x)
        postx = self.post_ffn(gate*v)
        x = self.ln(x + postx)
        return x

class RNovaEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.mha = SelfMultiHeadAttn(hidden_size, num_heads)
        self.ffn = FFNGLU(hidden_size)

    def forward(self,x,pos):
        x = checkpoint(self.mha, x, pos, use_reentrant=False)
        x = checkpoint(self.ffn, x, use_reentrant=False)
        #x = self.mha(x,pos)
        #x = self.ffn(x)
        return x