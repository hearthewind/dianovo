import math

import torch
import torch.nn as nn
from math import pi
from rnova.module_gnova import GNovaEncoderLayer
import torch.nn.functional as F

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim, output_dtype, lambda_max=1e4, lambda_min=1e-5) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        base = lambda_max/(2*pi)
        scale = lambda_min/lambda_max
        self.div_term = nn.Parameter(base*scale**(torch.arange(0, dim, 2)/dim), requires_grad=False)
        self.output_dtype = output_dtype

    def forward(self, mass_position):
        pe_sin = torch.sin(mass_position.unsqueeze(dim=-1) / self.div_term)
        pe_cos = torch.cos(mass_position.unsqueeze(dim=-1) / self.div_term)
        return torch.concat([pe_sin, pe_cos],dim=-1).to(self.output_dtype)


class DilatedConvolutionModule(nn.Module):
    def __init__(self, feature_dim, output_dim=256, max_time=5):
        super(DilatedConvolutionModule, self).__init__()

        hidden_size = output_dim // 8

        # Determine the necessary number of layers and dilation to reduce timesteps to 1
        self.conv_layers = []
        self.ln_layers = []
        current_timesteps = max_time
        layer_idx = 0
        while current_timesteps > 1:
            dilation = 2 ** layer_idx
            kernel_size = 3  # You can choose a different size if needed
            padding = (kernel_size - 1) * dilation // 2  # to maintain output length

            # Ensure the last convolutional layer reduces the timestep to 1
            final_layer = False
            if (current_timesteps - 1) // (dilation * (kernel_size - 1)) < 1:
                final_layer = True
                dilation = math.ceil((current_timesteps - 1) / (kernel_size - 1))
                padding = 0  # No padding for the last layer to reduce to single timestep

            conv = nn.Conv1d(feature_dim if layer_idx == 0 else hidden_size, hidden_size,
                             kernel_size=kernel_size, dilation=dilation, padding=padding)
            ln = nn.LayerNorm([hidden_size, max_time if not final_layer else 1])
            self.conv_layers.append(conv)
            self.ln_layers.append(ln)
            self.add_module('conv{}'.format(layer_idx), conv)
            self.add_module('ln{}'.format(layer_idx), ln)

            # Calculate the size of the output after this layer
            current_timesteps = math.floor((current_timesteps + 2 * padding - dilation * (kernel_size - 1) - 1) / 1) + 1
            layer_idx += 1

        self.output_layer = nn.Conv1d(hidden_size, output_dim, kernel_size=1)
        self.output_ln = nn.LayerNorm(output_dim)

    def forward(self, x):
        # x shape: (batch_size, num_points, timesteps, feature_dim)
        # We need to process over the timesteps, so we transpose the timesteps and feature_dim
        batch_size, num_points, timesteps, feature_dim = x.size()
        x = x.transpose(2, 3)  # New shape: (batch_size, num_points, feature_dim, timesteps)

        # Now, flatten the batch_size and num_points dimensions to treat each series of points as independent
        x = x.contiguous().view(batch_size * num_points, feature_dim, timesteps)

        # Pass through the dilated convolution layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            x = self.ln_layers[i](x)
            x = F.relu(x)

        # Pass through the output layer to get the final feature dimension we want
        x = self.output_layer(x)

        # Now we need to reshape to get back to the (batch_size, num_points, output_dim)
        x = x.view(batch_size, num_points, -1)

        x = self.output_ln(x)

        return x

class GNovaEncoder(nn.Module):
    def __init__(self, cfg, output_dtype):
        """_summary_

        Args:
            cfg (_type_): _description_
        """
        super().__init__()
        # self.peak_feature_proj = nn.Linear(5 * 8, cfg.encoder_gnova.hidden_size)
        # self.peak_xgram_proj = nn.Linear(5, cfg.encoder_gnova.hidden_size)

        self.peak_feature_proj = DilatedConvolutionModule(8, cfg.encoder_gnova.hidden_size, 5)
        self.peak_xgram_proj = DilatedConvolutionModule(1, cfg.encoder_gnova.hidden_size, 5)

        self.peak_feature_ln = nn.LayerNorm(cfg.encoder_gnova.hidden_size)

        self.head_size = cfg.encoder_gnova.d_relation // cfg.encoder_gnova.num_heads
        self.peak_mzs_embedding = SinusoidalPositionEmbedding(self.head_size, output_dtype)

        self.peak_flag_embedding = nn.Embedding(cfg.data.precursor_max_charge + 3, cfg.encoder_gnova.hidden_size, padding_idx=0)
        
        self.genova_encoder_layers = nn.ModuleList([ \
            GNovaEncoderLayer(hidden_size = cfg.encoder_gnova.hidden_size, num_heads=cfg.encoder_gnova.num_heads, d_relation = cfg.encoder_gnova.d_relation,
                              alpha = (2*cfg.encoder_gnova.num_layers)**0.25, beta = (8*cfg.encoder_gnova.num_layers)**-0.25,
                              dropout_rate = cfg.encoder_gnova.dropout_rate, device_type=cfg.device) for _ in range(cfg.encoder_gnova.num_layers)])
        
    def forward(self, moverz, xgram, feature, peak_flag_index):
        # peak_num = peak_features.size(1)
        # peak_features = peak_features.view(-1, peak_num, 5 * 8) # this is for simple mlp

        xgram = xgram.unsqueeze(-1)

        peak_features = self.peak_feature_proj(feature) + self.peak_xgram_proj(xgram) + self.peak_flag_embedding(peak_flag_index.long())
        peak_features = self.peak_feature_ln(peak_features)

        peak_mzs_embed = self.peak_mzs_embedding(moverz)
        neg_peak_mzs_embed = self.peak_mzs_embedding(-moverz)
        all_peak_features = []
        for genova_encoder_layer in self.genova_encoder_layers:
            peak_features = genova_encoder_layer(peak_features, peak_mzs_embed, neg_peak_mzs_embed)
            all_peak_features.append(peak_features)
        return all_peak_features
