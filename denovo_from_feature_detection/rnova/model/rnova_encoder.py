import torch
from torch import nn
from torch.nn.functional import glu

from rnova.data.dataset import astral_filtering
from rnova.module.dilatedcnn import DilatedConvolutionModule
from rnova.module.encoder_layer import RNovaEncoderLayer
from rnova.module.sin_embedding import SinusoidalPositionEmbedding


class AbosoluteInputEmbedding(nn.Module):
    def __init__(self, cfg, output_dtype) -> None:
        super().__init__()
        self.moverz_embedding = nn.Sequential(SinusoidalPositionEmbedding(cfg.encoder.hidden_size,
                                                                          cfg.embedding.moverz_lambda_max,
                                                                          cfg.embedding.moverz_lambda_min,
                                                                          output_dtype),
                                              nn.Linear(cfg.encoder.hidden_size, cfg.encoder.hidden_size))

        self.peak_class_embedding = nn.Embedding(
            cfg.data.product_max_charge * 6 + 2 + cfg.data.precursor_max_charge + 1 + 30, cfg.encoder.hidden_size)
        self.pos_embedding = nn.Embedding(cfg.data.ms2_max_peak_count, cfg.encoder.hidden_size)
        self.ms1_ms2_embedding = nn.Embedding(2, cfg.encoder.hidden_size)

    def forward(self, moverz, peak_class, pos, ms1_ms2_flag):
        moverz = self.moverz_embedding(moverz)
        peak_class = self.peak_class_embedding(peak_class)
        pos = self.pos_embedding(pos)
        ms1_ms2_flag = self.ms1_ms2_embedding(ms1_ms2_flag)
        x = moverz + peak_class + pos + ms1_ms2_flag
        return x


class RelativeInputEmbedding(nn.Module):
    def __init__(self, cfg, output_dtype) -> None:
        super().__init__()
        self.cfg = cfg
        self.relative_moverz_embedding = SinusoidalPositionEmbedding(cfg.encoder.hidden_size // cfg.encoder.num_heads,
                                                                     cfg.embedding.moverz_lambda_max,
                                                                     cfg.embedding.moverz_lambda_min,
                                                                     output_dtype)

    def forward(self, moverz):
        relative_moverz = self.relative_moverz_embedding(moverz)
        relative = torch.stack([relative_moverz], dim=-2)
        relative = relative.repeat_interleave(self.cfg.encoder.num_heads, dim=-2)
        return relative


class RNovaEncoder(nn.Module):
    def __init__(self, cfg, output_dtype, device) -> None:
        super().__init__()
        self.abosolute_embedding = AbosoluteInputEmbedding(cfg, output_dtype)
        self.relative_embedding = RelativeInputEmbedding(cfg, output_dtype)

        self.peak_feature_proj = DilatedConvolutionModule(8, cfg.encoder.hidden_size, 5)
        self.peak_xgram_proj = DilatedConvolutionModule(1, cfg.encoder.hidden_size, 5)

        self.encoder = nn.ModuleList(
            [RNovaEncoderLayer(cfg.encoder.hidden_size, cfg.encoder.num_heads, cfg.encoder.dropout_rate, cfg.device) \
             for _ in range(cfg.encoder.num_layers)])

        self.gnova_proj = nn.Linear(cfg.encoder.hidden_size, cfg.encoder.hidden_size * 2)
        self.gnova_dropout = nn.Dropout(cfg.encoder.dropout_rate)

        self.input_ln = nn.LayerNorm(cfg.encoder.hidden_size)
        self.output_ln = nn.LayerNorm(cfg.encoder.hidden_size)

        self.cfg = cfg
        self.device = device

    def forward(self, moverz, xgram, feature, peak_class_index, pos_index, ms1_ms2_flag, \
                gnova_encoder_output_list, meta_info_list):
        xgram = xgram.unsqueeze(-1)
        xgram_info = self.peak_feature_proj(feature) + self.peak_xgram_proj(xgram)

        indices = [x['indices'] for x in meta_info_list]
        sizes = [x['sizes'] for x in meta_info_list]
        precursor_charges = [x['precursor_charge'] for x in meta_info_list]

        gnova_features = self.process_all_gnova_features(gnova_encoder_output_list, indices, sizes, precursor_charges)
        gnova_features = glu(self.gnova_proj(self.gnova_dropout(gnova_features)))  # gated mechanism

        absolute_embedded = self.abosolute_embedding(moverz, peak_class_index, pos_index, ms1_ms2_flag)

        x = absolute_embedded + xgram_info + gnova_features
        x = self.input_ln(x)

        relative = self.relative_embedding(moverz)
        for encoder_layer in self.encoder:
            x = encoder_layer(x, relative)
        return self.output_ln(x)

    def process_all_gnova_features(self, gnova_encoder_output_list: list, indices: list, sizes: list,
                                   precursor_charges: list):
        features = []
        for i, gnova_encoder_output in enumerate(gnova_encoder_output_list):
            size_dict = sizes[i]
            ms1_sort_indices = indices[i]['ms1_sort']
            ms2_sort_indices = indices[i]['ms2_sort']
            ms1_multiscan_indices = indices[i]['ms1_multiscan']
            ms2_multiscan_indices = indices[i]['ms2_multiscan']
            precursor_charge = precursor_charges[i]

            feature = self.process_gnova_feature(gnova_encoder_output, size_dict, \
                                                 ms1_sort_indices, ms2_sort_indices, \
                                                 ms1_multiscan_indices, ms2_multiscan_indices, \
                                                 precursor_charge)
            features.append(feature)

        ret = torch.concat(features, dim=0)
        return ret

    def process_gnova_feature(self, gnova_encoder_output: torch.Tensor, size_dict: dict, \
                              ms1_sort_indices, ms2_sort_indices, \
                              ms1_multiscan_indices, ms2_multiscan_indices, \
                              precursor_charge: int):
        num_ms1 = size_dict['ms1']

        ms1_feature = gnova_encoder_output[:, :num_ms1, :]
        ms2_feature = gnova_encoder_output[:, num_ms1:, :]

        if astral_filtering:
            if ms1_multiscan_indices is not None:
                ms1_feature = ms1_feature[:, ms1_multiscan_indices, :]
            ms2_feature = ms2_feature[:, ms2_multiscan_indices, :]

        ms1_feature = torch.concat([ms1_feature for _ in range(1, self.cfg.data.precursor_max_charge + 1)], dim=1)
        ms1_feature = ms1_feature[:, ms1_sort_indices, :]

        ms2_feature_list = []
        for ion in self.cfg.data.n_term_ion:
            for charge in range(1, self.cfg.data.product_max_charge + 1):
                if precursor_charge == 2 and charge > 1: continue

                ms2_feature_list += [ms2_feature]

        for ion in self.cfg.data.c_term_ion:
            for charge in range(1, self.cfg.data.product_max_charge + 1):
                if precursor_charge == 2 and charge > 1: continue

                ms2_feature_list += [ms2_feature]

        ms2_feature = torch.concat(ms2_feature_list, dim=1)
        ms2_feature = ms2_feature[:, ms2_sort_indices, :]

        feature_dim = ms1_feature.shape[-1]
        ms2_feature = torch.concat([torch.zeros(1, 1, feature_dim, device=self.device), ms2_feature,
                                    torch.zeros(1, 1, feature_dim, device=self.device)], dim=1)

        feature = torch.concat([ms1_feature, ms2_feature], dim=1)
        return feature
