import numpy as np
import torch
from torch.nn.functional import pad


class GenovaCollator(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        if self.cfg.task == 'peak_classification':
            specs = [record[0] for record in batch]
            labels = [record[1] for record in batch]

            ms1_peak_nums = [spec['ms1_peak_num'] for spec in specs]
            encoder_input = self.encoder_collate(specs)
            label, label_mask = self.label_collate(labels, ms1_peak_nums)

            return encoder_input, label, label_mask
        elif self.cfg.task == 'peak_classification_multiclass':
            specs = [record[0] for record in batch]
            labels = [record[1] for record in batch]

            ms1_peak_nums = [spec['ms1_peak_num'] for spec in specs]
            encoder_input = self.encoder_collate(specs)
            label, label_mask = self.label_collate(labels, ms1_peak_nums)

            return encoder_input, label, label_mask
        elif self.cfg.task == "simple_decoder" or self.cfg.task == "simple_decoder_v2":
            specs = [record[0] for record in batch]
            tgts = [record[1] for record in batch]
            peak_labels = [record[2] for record in batch]
            seq_labels = [record[3] for record in batch]
            idxes = [record[4] for record in batch]

            ms1_peak_nums = [spec['ms1_peak_num'] for spec in specs]
            encoder_input = self.encoder_collate(specs)
            decoder_input = self.decoder_collate(tgts)
            peak_label, peak_label_mask, seq_label, seq_label_mask = self.dec_label_collate(seq_labels, ms1_peak_nums, peak_labels)

            return encoder_input, decoder_input, peak_label, peak_label_mask, seq_label, seq_label_mask, idxes
        elif self.cfg.task == "optimal_path":
            specs = [record[0] for record in batch]
            tgts = [record[1] for record in batch]
            seq_peak_labels = [record[2] for record in batch]
            idxes = [record[3] for record in batch]

            encoder_input = self.encoder_collate(specs)
            decoder_input = self.decoder_collate(tgts)
            label, label_mask = self.dec_label_collate(seq_peak_labels)

            return encoder_input, decoder_input, label, label_mask, idxes
        elif self.cfg.task == "sequence_generation":
            specs = [record[0] for record in batch]
            tgts = [record[1] for record in batch]
            labels = [record[2] for record in batch]
            idxes = [record[3] for record in batch]

            encoder_input = self.encoder_collate(specs)
            decoder_input = self.decoder_collate(tgts)
            label, label_mask = self.dec_label_collate(labels)

            return encoder_input, decoder_input, label, label_mask, idxes
        else:
            raise NotImplementedError

    def encoder_collate(self, specs):
        peak_features = torch.stack([spec['peak_features'] for i, spec in enumerate(specs)])
        peak_xgrams = torch.stack([spec['peak_xgrams'] for i, spec in enumerate(specs)])
        peak_mzs = torch.stack([spec['peak_mzs'] for i, spec in enumerate(specs)])
        peak_charges = torch.stack([spec['peak_charges'] for i, spec in enumerate(specs)])

        encoder_input = {
            'peak_features': peak_features,
            'peak_xgrams': peak_xgrams,
            'peak_mzs': peak_mzs,
            'peak_charges': peak_charges
        }

        return encoder_input

    def decoder_collate(self, tgts):
        if self.cfg.task == 'simple_decoder' or self.cfg.task == 'simple_decoder_v2':
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
        elif self.cfg.task == 'optimal_path':
            tgts_list = [tgt['tgt'] for tgt in tgts]
            step_mass_list = [tgt['step_mass'] for tgt in tgts]

            shape_list = np.array([tgt.shape for tgt in tgts_list])
            seqdblock_max = shape_list[:, 0].max()
            peak_max = shape_list[:, 1].max()

            peak_probabilities = []
            step_masses = []
            for i in range(len(tgts_list)):
                assert peak_max - shape_list[i, 1] == 0
                peak_probabilities.append(pad(tgts_list[i], [0, 0, 0, 0, 0, seqdblock_max - shape_list[i, 0]]))
                step_masses.append(pad(step_mass_list[i], [0, seqdblock_max - shape_list[i, 0]]))

            peak_probabilities = torch.stack(peak_probabilities) # batch_size, seq_len, peak_num, ion_types
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

    def label_collate(self, labels, ms1_peak_nums):
        if self.cfg.task == 'peak_classification':
            effective_labels = []
            for i, label in enumerate(labels):
                ms1_peak_num = ms1_peak_nums[i]
                effective_label = label[:-ms1_peak_num]
                effective_labels.append(effective_label)

            padded_label = torch.stack(labels)
            label_mask = torch.stack([pad(torch.ones_like(label, dtype=bool), (0, ms1_peak_nums[i])) for i, label in enumerate(effective_labels)])

            return padded_label, label_mask
        elif self.cfg.task == 'peak_classification_multiclass':
            effective_labels = []
            for i, label in enumerate(labels):
                ms1_peak_num = ms1_peak_nums[i]
                effective_label = label[:-ms1_peak_num]
                effective_labels.append(effective_label)

            padded_label = torch.stack(labels)
            label_mask = torch.stack([pad(torch.ones_like(label, dtype=bool), (0, ms1_peak_nums[i])) for i, label in enumerate(effective_labels)])
            return padded_label, label_mask
        else: raise NotImplementedError

    def dec_label_collate(self, seq_labels, ms1_peak_nums=None, peak_labels=None):
        if self.cfg.task == 'simple_decoder' or self.cfg.task == 'simple_decoder_v2':
            assert peak_labels is not None
            assert ms1_peak_nums is not None

            effective_peak_labels = []
            for i, peak_label in enumerate(peak_labels):
                ms1_peak_num = ms1_peak_nums[i]
                effective_peak_label = peak_label[:-ms1_peak_num]
                effective_peak_labels.append(effective_peak_label)

            padded_peak_label = torch.stack(peak_labels)
            peak_label_mask = torch.stack([pad(torch.ones_like(peak_label, dtype=bool), (0, ms1_peak_nums[i])) for i, peak_label in enumerate(effective_peak_labels)])

            shape_list = np.array([label.shape for label in seq_labels])
            seqdblock_max = shape_list.max()
            padded_seq_label = []
            seq_label_pading_mask = torch.ones(len(seq_labels), seqdblock_max, dtype=bool)
            for i, label in enumerate(seq_labels):
                seq_label_pading_mask[i, label.shape[0]:] = 0
                label = pad(label,[0,seqdblock_max-label.shape[0]])
                padded_seq_label.append(label)
            padded_seq_label = torch.stack(padded_seq_label)

            return padded_peak_label, peak_label_mask, padded_seq_label, seq_label_pading_mask
        elif self.cfg.task == 'optimal_path':
            shape_list = np.array([label.shape for label in seq_labels])
            seqdblock_max = shape_list[:, 0].max()
            peak_max = shape_list[:, 1].max()
            batch_size = len(seq_labels)

            result = []
            result_pading_mask = torch.ones(batch_size,seqdblock_max,dtype=bool)
            for i, label in enumerate(seq_labels):
                result_pading_mask[i, label.shape[0]:] = 0

                assert peak_max-label.shape[1] == 0
                label = pad(label,[0,0,0,seqdblock_max-label.shape[0]])

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
