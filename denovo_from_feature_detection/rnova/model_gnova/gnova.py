import torch.nn as nn
import torch

from utils.cofragment_peptide_processing import label_types
from utils.data.BasicClass import Residual_seq
from .gnova_decoder import GNovaDecoder
from .gnova_encoder import GNovaEncoder


class GNova(nn.Module):
    def __init__(self,cfg, output_dtype) -> None:
        super().__init__()
        self.cfg = cfg
        self.dict_len = len(Residual_seq.output_aadict()) + 3
        self.ion_types = len(label_types)

        self.encoder = GNovaEncoder(cfg, output_dtype)

        self.iontype_finear = nn.Sequential(
            nn.Linear(cfg.encoder.hidden_size, cfg.encoder.hidden_size),
            nn.LayerNorm(cfg.encoder.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.encoder.hidden_size, len(label_types))
        )
    
    def forward(self, encoder_input, decoder_input=None):
        encoder_output = self.encoder(**encoder_input)

        if self.cfg.task == 'pretrain':
            iontype_label = self.iontype_finear(encoder_output)

            # ionsource_embedding = self.ionsource_linear(encoder_output)
            ionsource_embedding = None

            return iontype_label, ionsource_embedding
        elif self.cfg.task == 'peak_classification':
            ret = self.output_linear(encoder_output)
            return ret
        elif self.cfg.task == 'optimal_path':
            assert decoder_input is not None
            peak_probs = decoder_input['peak_probs']
            step_mass = decoder_input['step_mass']
            pos = decoder_input['pos']

            query_node = peak_probs@encoder_output

            peak_mzs = encoder_input['moverz']

            query_node = self.decoder(query_node, step_mass, pos, encoder_output, peak_mzs)

            query_node = self.query_node_linear(query_node) # batch_size, seq_len, hidden_size
            graph_node = self.graph_node_linear(encoder_output).transpose(1, 2) # batch_size, hidden_size, peak_num

            graph_probability = query_node @ graph_node # batch_size, seq_len, peak_num

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
