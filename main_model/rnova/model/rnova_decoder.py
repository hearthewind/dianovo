import torch
from torch import nn

from rnova.module.decoder_layer import RNovaDecoderLayer
from rnova.module.sin_embedding import SinusoidalPositionEmbedding

class RelativeInputEmbedding(nn.Module):
    def __init__(self, cfg, output_dtype) -> None:
        super().__init__()
        self.cfg = cfg
        self.relative_moverz_embedding = SinusoidalPositionEmbedding(cfg.decoder.hidden_size // cfg.decoder.num_heads,
                                                                     cfg.embedding.moverz_lambda_max,
                                                                     cfg.embedding.moverz_lambda_min,
                                                                     output_dtype)

    def forward(self, moverz):
        relative_moverz = self.relative_moverz_embedding(moverz)
        relative = torch.stack([relative_moverz], dim=-2)
        relative = relative.repeat_interleave(self.cfg.decoder.num_heads, dim=-2)
        return relative

class RNovaDecoder(nn.Module):
    def __init__(self,cfg, output_dtype) -> None:
        super().__init__()
        self.cfg = cfg
        self.rnova_decoder_layers = nn.ModuleList([RNovaDecoderLayer(cfg) for _ in range(cfg.decoder.num_layers)])

        self.mass_embedding = RelativeInputEmbedding(cfg, output_dtype)
        self.pos_embedding = nn.Embedding(self.cfg.data.peptide_max_len, self.cfg.decoder.hidden_size)

    def forward(self, tgt, step_mass, pos, graph_node, peak_mzs):
        step_mass_embed = self.mass_embedding(step_mass)
        peak_mzs_embed = self.mass_embedding(peak_mzs)
        pos_embed = self.pos_embedding(pos)
        tgt = tgt + pos_embed

        for rnova_decoder_layer in self.rnova_decoder_layers:
            tgt = rnova_decoder_layer(tgt=tgt, step_mass_embed=step_mass_embed,
                                      mem=graph_node, peak_mzs_embed=peak_mzs_embed)
        return tgt
