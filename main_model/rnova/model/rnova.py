import torch
from torch import nn

from rnova.model.rnova_decoder import RNovaDecoder
from rnova.model.rnova_encoder import RNovaEncoder
from rnova.model_gnova import GNova
from utils.cofragment_peptide_processing import label_types
from utils.data.BasicClass import Residual_seq


class RNova(nn.Module):
    def __init__(self, cfg, output_dtype, device):
        super().__init__()
        self.cfg = cfg
        self.encoder = RNovaEncoder(cfg, output_dtype, device)

        if self.cfg.task == 'node_classification':
            self.output_linear = nn.Linear(cfg.encoder.hidden_size, 1)
        elif self.cfg.task == 'optimal_path':
            self.decoder = RNovaDecoder(cfg, output_dtype)
            self.query_node_linear = nn.Sequential(nn.LayerNorm(cfg.decoder.hidden_size),
                                                   nn.Linear(cfg.decoder.hidden_size, cfg.decoder.hidden_size),
                                                   )

            self.graph_node_linear = nn.Sequential(nn.LayerNorm(cfg.encoder.hidden_size),
                                                   nn.Linear(cfg.encoder.hidden_size, cfg.decoder.hidden_size),
                                                   )
        elif self.cfg.task == 'sequence_generation':
            dict_len = len(Residual_seq.output_aadict()) + 3
            self.tgt_embedding = nn.Embedding(dict_len, self.cfg.decoder.hidden_size, padding_idx=0)
            self.decoder = RNovaDecoder(cfg, output_dtype)
            self.decoder_output_linear = nn.Sequential(nn.LayerNorm(cfg.decoder.hidden_size),
                                                       nn.Linear(cfg.decoder.hidden_size, dict_len)
                                                       )
        else:
            raise NotImplementedError

    def forward(self, gnova_model, enc_rnova, enc_gnova, meta_info_list, decoder_input=None):
        gnova_encoder_output_list = []
        with torch.no_grad():
            for encoder_input in enc_gnova:
                try:
                    gnova_encoder_output = gnova_model.encoder(**encoder_input)
                except AttributeError:
                    gnova_encoder_output = gnova_model.module.encoder(**encoder_input)
                gnova_encoder_output_list.append(gnova_encoder_output)
        return self.forward_rnova(enc_rnova, gnova_encoder_output_list, meta_info_list, decoder_input)

    def forward_rnova(self, encoder_input, gnova_encoder_output_list, meta_info_list, decoder_input=None):
        encoder_output = self.encoder(**encoder_input, gnova_encoder_output_list=gnova_encoder_output_list, meta_info_list=meta_info_list)

        if self.cfg.task == 'node_classification':
            label = self.output_linear(encoder_output)
            return label
        elif self.cfg.task == 'optimal_path':
            assert decoder_input is not None
            graph_probs = decoder_input['graph_probs']
            step_mass = decoder_input['step_mass']
            pos = decoder_input['pos']

            query_node = graph_probs@encoder_output

            peak_mzs = encoder_input['moverz']

            query_node = self.decoder(query_node, step_mass, pos, encoder_output, peak_mzs)
            query_node = self.query_node_linear(query_node)
            graph_node = self.graph_node_linear(encoder_output).transpose(1, 2)

            graph_probability = query_node @ graph_node

            return graph_probability
        elif self.cfg.task == 'sequence_generation':
            assert decoder_input is not None
            seq = decoder_input['seq']
            pos = decoder_input['pos']
            step_mass = decoder_input['step_mass']
            tgt = self.tgt_embedding(seq)

            peak_mzs = encoder_input['moverz']
            decoder_output = self.decoder(tgt, step_mass, pos, encoder_output, peak_mzs)
            decoder_output = self.decoder_output_linear(decoder_output)
            return decoder_output
        else:
            raise NotImplementedError
