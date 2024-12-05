import torch
from torch import nn


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim, lambda_max, lambda_min, output_dtype) -> None:
        super().__init__()
        self.output_dtype = output_dtype
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        base = lambda_max/(2*torch.pi)
        scale = lambda_min/lambda_max
        div_term = base*scale**(torch.arange(0, dim, 2, dtype=torch.float64)/dim)
        self.register_buffer('div_term', div_term)

    def forward(self, mass_position):
        pe_sin = torch.sin(mass_position.unsqueeze(dim=-1) / self.div_term)
        pe_cos = torch.cos(mass_position.unsqueeze(dim=-1) / self.div_term)
        return torch.concat([pe_sin, pe_cos],dim=-1).to(self.output_dtype)