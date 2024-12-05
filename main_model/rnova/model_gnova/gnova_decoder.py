import torch.nn as nn

from rnova.model_gnova.gnova_encoder import SinusoidalPositionEmbedding
from rnova.module_gnova import GNovaDecoderLayer


class GNovaDecoder(nn.Module):
    def __init__(self, cfg, output_dtype) -> None:
        super().__init__()
        self.cfg = cfg
        self.gnova_decoder_layers = nn.ModuleList([GNovaDecoderLayer(tgt_hidden_size = cfg.decoder_gnova.hidden_size,
                                                                      mem_hidden_size = cfg.encoder_gnova.hidden_size,
                                                                      num_heads = cfg.decoder_gnova.num_heads,
                                                                      d_relation = cfg.decoder_gnova.d_relation,
                                                                      alpha=(2 * cfg.decoder_gnova.num_layers) ** 0.25,
                                                                      beta=(8 * cfg.decoder_gnova.num_layers) ** -0.25,
                                                                      dropout_rate = cfg.decoder_gnova.dropout_rate) for _ in range(cfg.decoder_gnova.num_layers)])

        self.head_size = cfg.decoder_gnova.d_relation // cfg.decoder_gnova.num_heads

        self.step_mass_embedding = SinusoidalPositionEmbedding(self.head_size, output_dtype)
        self.peak_mzs_embedding = SinusoidalPositionEmbedding(self.head_size, output_dtype)
        self.pos_embedding = nn.Embedding(cfg.data.peptide_max_len, cfg.decoder_gnova.hidden_size)

    def forward(self, tgt, step_mass, pos, encoder_output, peak_mzs):
        step_mass_embed = self.step_mass_embedding(step_mass)
        pos_embed = self.pos_embedding(pos)
        tgt = tgt + pos_embed

        peak_mzs_embed = self.peak_mzs_embedding(peak_mzs)
        neg_peak_mzs_embed = self.peak_mzs_embedding(-peak_mzs)

        for gnova_decoder_layer in self.gnova_decoder_layers:
            tgt = gnova_decoder_layer(tgt=tgt, step_mass_ebed=step_mass_embed, \
                                      mem=encoder_output, peak_mzs_embed=peak_mzs_embed, neg_peak_mzs_embed=neg_peak_mzs_embed)
        return tgt
