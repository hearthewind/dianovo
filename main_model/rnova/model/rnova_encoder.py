import math

import torch
from flash_attn import flash_attn_func
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

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
        
        self.peak_class_embedding = nn.Embedding(cfg.data.product_max_charge*6+2+cfg.data.precursor_max_charge+1 + 30, cfg.encoder.hidden_size)
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
        self.relative_moverz_embedding = SinusoidalPositionEmbedding(cfg.encoder.hidden_size//cfg.encoder.num_heads, 
                                                                     cfg.embedding.moverz_lambda_max, 
                                                                     cfg.embedding.moverz_lambda_min,
                                                                     output_dtype)
    
    def forward(self, moverz):
        relative_moverz = self.relative_moverz_embedding(moverz)
        relative = torch.stack([relative_moverz],dim=-2)
        relative = relative.repeat_interleave(self.cfg.encoder.num_heads,dim=-2)
        return relative

class GNovaAttention(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 dropout=0.1,
                 device_type='gpu'):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.device_type = device_type

    def forward(self, x, gnova_embedding):
        batch_size = x.size(0)
        q, k, v = self.linear_q(x), self.linear_k(gnova_embedding), self.linear_v(gnova_embedding)
        q = q.view(batch_size, -1, self.num_heads, self.head_size)
        k = k.view(batch_size, -1, self.num_heads, self.head_size)
        v = v.view(batch_size, -1, self.num_heads, self.head_size)

        if self.device_type == 'gpu':
            postx = flash_attn_func(q, k, v).flatten(2, 3)
        elif self.device_type == 'cpu':
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            postx = F.scaled_dot_product_attention(q, k, v,
                                                   attn_mask=None,
                                                   is_causal=False,
                                                   scale=1.0 / self.head_size ** 0.5)
            postx = postx.permute(0, 2, 1, 3).flatten(2, 3)
        else:
            raise ValueError(f'Unsupported device type: {self.device_type}')

        postx = self.dropout(self.output_layer(postx))
        return postx

class SeqGate(nn.Module):
    """
    Sequence-scalar gate g ∈ (0,1) with:
        • avg + max pooling (richer stats)
        • Gaussian jitter on logits  (noise_std)
    """
    def __init__(self, hidden_size: int, noise_std: float = 0.05, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
        self.proj      = nn.Linear(hidden_size * 4, 1)
        self.noise_std = noise_std
        self.sigmoid   = nn.Sigmoid()

    def _entropy_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits : [B, L, C] – final softmax-preactivation of gnova
        returns: [B, 1]    – confidence in (0,1], higher = more certain
        """
        B, L, C = logits.shape
        p = F.softmax(logits, dim=-1)  # [B,L,C]
        entropy = -(p * torch.log(p + self.eps)).sum(dim=-1)  # [B,L]
        mean_ent = entropy.mean(dim=1, keepdim=True)  # [B,1]
        # normalise: 0 = max entropy, 1 = zero entropy
        confidence = 1.0 - mean_ent / math.log(float(C) + self.eps)
        confidence = confidence.clamp(0.0, 1.0)  # safety
        return confidence  # [B,1]

    def forward(
            self,
            x: torch.Tensor,  # [B, L, D]   clean encoder output
            gnova_emb: torch.Tensor,  # [B, L, D]   projected gnova
            gnova_logits: torch.Tensor,  # [B, L, C]   gnova classification logits
    ):
        # ---- pooled stats (idea B) ----
        avg_x = x.mean(dim=1)  # [B,D]
        max_x = x.max(dim=1).values  # [B,D]
        avg_g = gnova_emb.mean(dim=1)  # [B,D]
        max_g = gnova_emb.max(dim=1).values  # [B,D]
        seq_vec = torch.cat([avg_x, max_x, avg_g, max_g], dim=-1)

        # ---- raw gate from pooled stats ----
        logits_gate = self.proj(seq_vec)  # [B,1]
        if self.training and self.noise_std > 0:
            logits_gate = logits_gate + torch.randn_like(logits_gate) * self.noise_std
        gate_raw = self.sigmoid(logits_gate)  # [B,1]

        # ---- entropy confidence (idea A) ----
        conf = self._entropy_confidence(gnova_logits)  # [B,1]

        # ---- final gate ----
        # print('confidence value:', conf.detach().to(torch.float32).cpu().numpy().reshape(-1)) #TODO(m) check confidence value
        gate = (gate_raw * conf).view(-1, 1, 1)  # broadcast [B,1,1]
        return gate  # ready for path-drop


class GnovaPathDrop(nn.Module):
    """
    • During the first `warmup_steps` : gnova OFF (gate masked to 0).
    • After warm-up : stochastic depth with prob `p_drop`.
      The keep_mask *multiplies the gate*, so when the path is dropped
      gnova has literally no influence and the gate gets a learning signal.
    """
    def __init__(self, warmup_steps: int = 0):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))

    def forward(self, gate: torch.Tensor, gnova_emb: torch.Tensor):
        """
        gate      : [B,1,1]   scalar per sequence  (0…1)
        gnova_emb : [B,L,D]   projected gnova
        returns   : gated (and maybe dropped) gnova term  [B,L,D]
        """
        B = gnova_emb.size(0)

        if self.training and self.step < self.warmup_steps:
            keep_mask = torch.zeros(B, 1, 1, device=gnova_emb.device)
        else:  # eval or after warmup
            keep_mask = torch.full((B, 1, 1), 1.0,
                                   device=gnova_emb.device)

        if self.training:
            self.step.data += 1

        gate = gate * keep_mask              # mask also zeros gate
        return gate * gnova_emb              # ready for residual add

class RNovaEncoder(nn.Module):
    def __init__(self, cfg, output_dtype) -> None:
        super().__init__()
        self.abosolute_embedding = AbosoluteInputEmbedding(cfg, output_dtype)
        self.relative_embedding = RelativeInputEmbedding(cfg, output_dtype)

        self.peak_feature_proj = DilatedConvolutionModule(8, cfg.encoder.hidden_size, 5)
        self.peak_xgram_proj = DilatedConvolutionModule(1, cfg.encoder.hidden_size, 5)

        self.encoder = nn.ModuleList([RNovaEncoderLayer(cfg.encoder.hidden_size, cfg.encoder.num_heads, cfg.encoder.dropout_rate, cfg.device) \
                                      for _ in range(cfg.encoder.num_layers)])

        self.gnova_proj = nn.Sequential(
            nn.Dropout(cfg.encoder.dropout_rate * 2),
            nn.Linear(cfg.encoder.hidden_size, cfg.encoder.hidden_size // 8),
            nn.ReLU(),
            nn.Dropout(cfg.encoder.dropout_rate),
            nn.Linear(cfg.encoder.hidden_size // 8, cfg.encoder.hidden_size),
            nn.Dropout(cfg.encoder.dropout_rate),
        )

        self.gnova_attention = GNovaAttention(cfg.encoder.hidden_size, cfg.encoder.num_heads, cfg.encoder.dropout_rate, cfg.device)

        self.seq_gate = SeqGate(cfg.encoder.hidden_size)
        self.gnova_path_drop = GnovaPathDrop(warmup_steps=cfg.train.warmup_steps * 2)

        self.input_ln = nn.LayerNorm(cfg.encoder.hidden_size)
        self.output_ln = nn.LayerNorm(cfg.encoder.hidden_size)

        self.gnova2_ln = nn.LayerNorm(cfg.encoder.hidden_size, elementwise_affine=False)
        self.gnova4_ln = nn.LayerNorm(cfg.encoder.hidden_size, elementwise_affine=False)

        self.cfg = cfg
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))

    def forward(self, moverz, xgram, feature, peak_class_index, pos_index, ms1_ms2_flag, \
                gnova_encoder_output_list, meta_info_list):
        if self.training:
            self.step.data += 1

        xgram = xgram.unsqueeze(-1)
        xgram_info = self.peak_feature_proj(feature) + self.peak_xgram_proj(xgram)

        indices = [x['indices'] for x in meta_info_list]
        sizes = [x['sizes'] for x in meta_info_list]
        precursor_charges = [x['precursor_charge'] for x in meta_info_list]

        absolute_embedded = self.abosolute_embedding(moverz, peak_class_index, pos_index, ms1_ms2_flag)

        x = absolute_embedded + xgram_info
        x = self.input_ln(x)

        relative = self.relative_embedding(moverz)
        for encoder_layer in self.encoder:
            x = encoder_layer(x, relative)

        gnova_features2, gnova_features4, gnova_logits = self.process_all_gnova_features(gnova_encoder_output_list, indices, sizes, precursor_charges)

        gnova_features2 = self.gnova_proj(gnova_features2)
        gnova_features2 = self.gnova2_ln(gnova_features2)
        gnova_features2 = self.gnova_attention(x, gnova_features2)

        gnova_features4 = self.gnova4_ln(gnova_features4)

        gate = self.seq_gate(x, gnova_features4, gnova_logits)
        gated_output = self.gnova_path_drop(gate, gnova_features2)

        x = x + gated_output

        return self.output_ln(x)

    def process_all_gnova_features(self, gnova_encoder_output_list: list, indices: list, sizes: list, precursor_charges: list):
        features2 = []
        features4 = []
        logits = []
        for i, (gnova_encoder_outputs, gnova_logits) in enumerate(gnova_encoder_output_list):
            size_dict = sizes[i]
            ms1_sort_indices = indices[i]['ms1_sort']
            ms2_sort_indices = indices[i]['ms2_sort']
            ms1_multiscan_indices = indices[i]['ms1_multiscan']
            ms2_multiscan_indices = indices[i]['ms2_multiscan']
            precursor_charge = precursor_charges[i]

            feature2 = self.process_gnova_feature(gnova_encoder_outputs[1], size_dict, \
                                                 ms1_sort_indices, ms2_sort_indices, \
                                                 ms1_multiscan_indices, ms2_multiscan_indices, \
                                                 precursor_charge)
            feature4 = self.process_gnova_feature(gnova_encoder_outputs[3], size_dict, \
                                                    ms1_sort_indices, ms2_sort_indices, \
                                                    ms1_multiscan_indices, ms2_multiscan_indices, \
                                                    precursor_charge)
            logit = self.process_gnova_feature(gnova_logits, size_dict, \
                                                    ms1_sort_indices, ms2_sort_indices, \
                                                    ms1_multiscan_indices, ms2_multiscan_indices, \
                                                    precursor_charge)
            features2.append(feature2)
            features4.append(feature4)
            logits.append(logit)

        ret2 = torch.concat(features2, dim=0)
        ret4 = torch.concat(features4, dim=0)
        ret_logit = torch.concat(logits, dim=0)
        return ret2, ret4, ret_logit

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
        ms2_feature = torch.concat([torch.zeros(1, 1, feature_dim, device=ms2_feature.device), ms2_feature, torch.zeros(1, 1, feature_dim, device=ms2_feature.device)], dim=1)

        feature = torch.concat([ms1_feature, ms2_feature], dim=1)
        return feature
