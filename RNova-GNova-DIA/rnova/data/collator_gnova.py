import numpy as np
import torch
from torch.nn.functional import pad


class GenovaCollator(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        spec_processed = [record[0] for record in batch]
        encoder_input = self.encoder_collate(spec_processed)

        idxes = [record[2] for record in batch]

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

    def cofragment_collate(self, cofragment_labels):
        ion_type = [x['ion_type'] for x in cofragment_labels]
        ion_source = [x['ion_source'] for x in cofragment_labels]

        ion_type = torch.stack(ion_type)
        ion_source = torch.stack(ion_source)

        ret = {'ion_type': ion_type, 'ion_source': ion_source}
        return ret

    def decoder_collate(self, tgts):
        if self.cfg.task == 'optimal_path':
            tgts_list = [tgt['tgt'] for tgt in tgts]
            step_mass_list = [tgt['step_mass'] for tgt in tgts]

            shape_list = np.array([tgt.shape for tgt in tgts_list])
            seqdblock_max = shape_list[:, 0].max()
            peak_max = shape_list[:, 1].max()

            peak_probabilities = []
            step_masses = []
            for i in range(len(tgts_list)):
                assert peak_max == shape_list[i, 1]
                peak_probabilities.append(pad(tgts_list[i], [0, 0, 0, seqdblock_max - shape_list[i, 0]]))
                step_masses.append(pad(step_mass_list[i], [0, seqdblock_max - shape_list[i, 0]]))

            peak_probabilities = torch.stack(peak_probabilities) # batch_size, seq_len, peak_num
            step_masses = torch.stack(step_masses)
            return {'peak_probs': peak_probabilities, 'step_mass': step_masses, 'pos': torch.arange(seqdblock_max)}
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
        else: raise NotImplementedError

    def label_collate(self, seq_labels):
        if self.cfg.task == 'optimal_path':
            shape_list = np.array([label.shape for label in seq_labels])
            seqdblock_max = shape_list[:, 0].max()
            peak_max = shape_list[:, 1].max()
            batch_size = len(seq_labels)

            result = []
            result_pading_mask = torch.ones(batch_size,seqdblock_max,dtype=bool)
            for i, label in enumerate(seq_labels):
                result_pading_mask[i, label.shape[0]:] = 0

                assert peak_max == label.shape[1]
                label = pad(label,[0, 0, 0, seqdblock_max - label.shape[0]])

                result.append(label)
            result = torch.stack(result)
            return result, result_pading_mask
        elif self.cfg.task == 'sequence_generation':
            shape_list = np.array([label.shape for label in seq_labels])
            seqdblock_max = int(shape_list.max())
            batch_size = len(seq_labels)

            result = []
            result_pading_mask = torch.ones(batch_size,seqdblock_max,dtype=bool)
            for i, label in enumerate(seq_labels):
                result_pading_mask[i, label.shape[0]:] = 0
                label = pad(label,[0,seqdblock_max-label.shape[0]])
                result.append(label)
            return torch.stack(result), result_pading_mask
        else: raise NotImplementedError
