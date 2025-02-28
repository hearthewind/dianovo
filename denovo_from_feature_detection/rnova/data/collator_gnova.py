import numpy as np
import torch
from torch.nn.functional import pad


class GenovaCollator(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        spec_processed = [record[0] for record in batch]
        encoder_input = self.encoder_collate(spec_processed)

        idxes = [record[1] for record in batch]

        return encoder_input, idxes


    def encoder_collate(self, specs):
        list_ret = []
        for spec in specs:
            encoder_input = {
                'moverz': spec['moverz'].unsqueeze(0),
                'xgram': spec['xgram'].unsqueeze(0),
                'feature': spec['feature'].unsqueeze(0),
                'peak_flag_index': spec['peak_flag_index'].unsqueeze(0),
            }

            list_ret.append(encoder_input)

        return list_ret
