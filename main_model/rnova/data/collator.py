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

        if self.cfg.task == 'node_classification':
            label = torch.stack([record[1] for record in batch])
            label_mask = torch.stack([record[2] for record in batch])
            return encoder_input, label, label_mask
        elif self.cfg.task == 'optimal_path':
            tgts = [record[1] for record in batch]
            decoder_input = self.decoder_collate(tgts)
            label, label_mask = self.label_collate(batch)
            idx = [record[4] for record in batch]
            return encoder_input, decoder_input, label, label_mask, idx
        elif self.cfg.task == 'sequence_generation':
            tgts = [record[1] for record in batch]
            decoder_input = self.decoder_collate(tgts)
            label, label_mask = self.label_collate(batch)
            idx = [record[3] for record in batch]
            return encoder_input, decoder_input, label, label_mask, idx
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

    def label_collate(self, batch):
        if self.cfg.task == 'optimal_path':
            labels = [record[2] for record in batch]
            label_masks = [record[3] for record in batch]

            shape_list = np.array([label.shape for label in labels])
            seqdblock_max = shape_list[:, 0].max()
            graph_max = shape_list[:, 1].max()

            result = []
            result_mask = []
            for i, label in enumerate(labels):
                assert graph_max - label.shape[1] == 0
                label_mask = label_masks[i]

                label = pad(label, [0, 0, 0, seqdblock_max - label.shape[0]])
                label_mask = pad(label_mask, [0, 0, 0, seqdblock_max - label_mask.shape[0]])

                result.append(label)
                result_mask.append(label_mask)
            result = torch.stack(result).float()
            result_mask = torch.stack(result_mask)
            return result, result_mask
        elif self.cfg.task == 'sequence_generation':
            labels = [record[2] for record in batch]
            shape_list = np.array([label.shape for label in labels])
            seqdblock_max = int(shape_list.max())
            batch_size = len(labels)

            result = []
            result_pading_mask = torch.ones(batch_size, seqdblock_max, dtype=bool)
            for i, label in enumerate(labels):
                result_pading_mask[i, label.shape[0]:] = 0
                label = pad(label, [0, seqdblock_max - label.shape[0]])
                result.append(label)
            return torch.stack(result), result_pading_mask
        else:
            raise NotImplementedError

    def decoder_collate(self, tgts):
        if self.cfg.task == 'optimal_path':
            tgts_list = [tgt['tgt'] for tgt in tgts]
            step_mass_list = [tgt['step_mass'] for tgt in tgts]

            shape_list = np.array([tgt.shape for tgt in tgts_list])
            seqdblock_max = shape_list[:, 0].max()
            graph_max = shape_list[:, 1].max()

            graph_probabilities = []
            step_masses = []
            for i in range(len(tgts_list)):
                assert graph_max == shape_list[i, 1]
                graph_probabilities.append(pad(tgts_list[i], [0, 0, 0, seqdblock_max - shape_list[i, 0]]))
                step_masses.append(pad(step_mass_list[i], [0, seqdblock_max - shape_list[i, 0]]))

            graph_probabilities = torch.stack(graph_probabilities).float()  # batch_size, seq_len, peak_num
            step_masses = torch.stack(step_masses)

            return {'graph_probs': graph_probabilities, 'step_mass': step_masses, 'pos': torch.arange(seqdblock_max)}
        elif self.cfg.task == 'sequence_generation':

            tgts_list = [tgt['tgt'] for tgt in tgts]
            step_mass_list = [tgt['step_mass'] for tgt in tgts]

            shape_list = np.array([tgt.shape[0] for tgt in tgts_list])
            seqdblock_max = shape_list.max()

            seqs = []
            step_masses = []
            for i in range(len(tgts_list)):
                seqs.append(pad(tgts_list[i], [0, seqdblock_max - shape_list[i]]))
                step_masses.append(pad(step_mass_list[i], [0, seqdblock_max - shape_list[i]]))

            seqs = torch.stack(seqs)
            step_masses = torch.stack(step_masses)
            return {'seq': seqs, 'step_mass': step_masses, 'pos': torch.arange(seqdblock_max)}
        else:
            raise NotImplementedError
