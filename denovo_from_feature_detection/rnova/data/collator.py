import numpy as np
import torch
from torch.nn.functional import pad

from rnova.data.collator_gnova import GenovaCollator


class RNovaCollator(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.gnova_collator = GenovaCollator(cfg)

    def __call__(self, batch):
        rnova = [record['rnova'] for record in batch]
        gnova = [record['gnova'] for record in batch]
        sizes_list = [record[-1] for record in gnova]
        meta_info_list = [record[-1] for record in rnova]

        for meta_info, sizes in zip(meta_info_list, sizes_list):
            meta_info['sizes'] = sizes

        ret = {'rnova': self.processe_rnova(rnova), 'gnova': self.gnova_collator(gnova), 'meta_info': meta_info_list}
        return ret

    def processe_rnova(self, batch):
        spec_processed = [record[0] for record in batch]
        encoder_input = self.encoder_collate(spec_processed)

        if self.cfg.task == 'optimal_path':
            idx = [record[1] for record in batch]
            return encoder_input, idx
        elif self.cfg.task == 'sequence_generation':
            idx = [record[1] for record in batch]
            return encoder_input, idx
        else:
            raise NotImplementedError

    def encoder_collate(self, spec_processed):
        moverz = torch.stack([record['moverz'] for record in spec_processed])
        xgram = torch.stack([record['xgram'] for record in spec_processed])
        feature = torch.stack([record['feature'] for record in spec_processed])
        peak_class_index = torch.stack([record['peak_class_index'] for record in spec_processed])
        pos_index = torch.stack([record['pos_index'] for record in spec_processed])
        ms1_ms2_flag = torch.stack([record['ms1_ms2_flag'] for record in spec_processed])

        encoder_input = {'moverz': moverz,
                         'xgram': xgram,
                         'feature': feature,
                         'peak_class_index': peak_class_index,
                         'pos_index': pos_index,
                         'ms1_ms2_flag': ms1_ms2_flag}

        return encoder_input

