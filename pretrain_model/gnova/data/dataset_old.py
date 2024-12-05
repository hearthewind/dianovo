import os
import gzip

import numpy as np
import torch
import pickle
from torch.utils.data import Dataset

from gnova.data.multiaa import multiaa_list
from gnova.utils.data.BasicClass import Residual_seq


def one_hot_encode(label: torch.Tensor, num_classes: int):
    label = label.long()
    one_hot = torch.eye(num_classes)[label]
    return one_hot

class GenovaDataset(Dataset):
    def __init__(self, cfg, *, spec_header, dataset_dir_path):
        super().__init__()
        self.cfg = cfg
        self.spec_header = spec_header
        self.dataset_dir_path = dataset_dir_path

        self.aa_id = {aa: i for i, aa in enumerate(Residual_seq.output_aalist(), start=3)}
        self.aa_id['<pad>'] = 0
        self.aa_id['<bos>'] = 1
        self.aa_id['<eos>'] = 2

        self.multiaa_id = {aa: i for i, aa in enumerate(multiaa_list, start=3)}
        self.multiaa_id['<pad>'] = 0
        self.multiaa_id['<bos>'] = 1
        self.multiaa_id['<eos>'] = 2

    def __getitem__(self, idx):
        if isinstance(idx, str): spec_head = dict(self.spec_header.loc[idx])
        else: spec_head = dict(self.spec_header.iloc[idx])
        with open(os.path.join(self.dataset_dir_path, spec_head['MSGP File Name']), 'rb') as f:
            f.seek(spec_head['MSGP Datablock Pointer'])
            try:
                spec = pickle.loads(gzip.decompress(f.read(spec_head['MSGP Datablock Length'])))
            except Exception as e:
                print("Cannot read data")
                print("file_name, ", spec_head['MSGP File Name'])
                raise(e)
        peak_mzs = torch.Tensor(spec['peak_mzs'])
        peak_features = torch.Tensor(spec['peak_features'])
        peak_xgrams = torch.Tensor(spec['peak_xgrams'])
        peak_charges = torch.Tensor(spec['peak_charges'])
        precursor_charge = int(spec_head['Charge'])
        ms1_peak_num = min(4, precursor_charge)

        spec_processed = {'peak_num': spec_head['Peak Number'],
                          'peak_features':peak_features, # (peak_num, scan_size, 8)
                          'peak_xgrams':peak_xgrams, # (peak_num, scan_size)
                          'peak_charges': peak_charges, # (peak_num)
                          'peak_mzs':peak_mzs, # (peak_num)
                          'ms1_peak_num':ms1_peak_num}


        if self.cfg.task == 'peak_classification':
            peak_labels = (spec['peak_labels'] > 0).float() # (peak_num)
            return spec_processed, peak_labels
        elif self.cfg.task == 'peak_classification_multiclass':
            peak_labels = spec['peak_labels']

            return spec_processed, peak_labels.reshape(-1).long()
        elif self.cfg.task == 'simple_decoder':
            seq = spec_head['Annotated Sequence'].replace('L', 'I').replace(' ', '')
            seq_id = self.seq2id(seq)
            step_mass = torch.Tensor(np.insert(Residual_seq(seq).step_mass[:-1], 0, 0.0))

            tgt = {}
            tgt['tgt'] = torch.Tensor(seq_id[:-1])
            tgt['step_mass'] = torch.Tensor(step_mass)

            peak_labels = spec['peak_labels'].reshape(-1).long()
            seq_labels = torch.Tensor(seq_id[1:])
            return spec_processed, tgt, peak_labels, seq_labels, idx
        elif self.cfg.task == 'simple_decoder_v2':
            seq = spec_head['Fragmented Sequence'].split(' ')
            seq_id = self.seq2multiid(seq)

            step_mass = [0.0]
            for i, fragment in enumerate(seq):
                step_mass.append(step_mass[i] + Residual_seq(fragment).mass)

            step_mass = torch.Tensor(step_mass[:-1])

            tgt = {}
            tgt['tgt'] = torch.Tensor(seq_id[:-1])
            tgt['step_mass'] = torch.Tensor(step_mass)

            peak_labels = spec['peak_labels'].reshape(-1).long()
            seq_labels = torch.Tensor(seq_id[1:])
            return spec_processed, tgt, peak_labels, seq_labels, idx
        elif self.cfg.task == 'optimal_path':
            seq = spec_head['Annotated Sequence'].replace('L', 'I').replace(' ', '')
            candidate_step_mass = torch.Tensor(np.insert(Residual_seq(seq).step_mass, 0, 0.0))

            seq_peak_labels = spec['seq_peak_labels']
            seq_len, peak_num = seq_peak_labels.shape
            seq_peak_labels = torch.cat([torch.zeros((1, peak_num), dtype=torch.long), seq_peak_labels[:, :]], 0)

            tmp_peak_labels = []
            tmp_step_mass = []
            for l in range(seq_len + 1):
                current_peak_label = seq_peak_labels[l, :]
                if current_peak_label.any():
                    tmp_peak_labels.append(current_peak_label.unsqueeze(0))
                    tmp_step_mass.append(candidate_step_mass[l])
            seq_peak_labels = torch.cat(tmp_peak_labels, 0)
            step_mass = torch.Tensor(tmp_step_mass)

            tgt = {}
            tgt['tgt'] = one_hot_encode(torch.Tensor(seq_peak_labels[:-1]), 14)
            tgt['step_mass'] = torch.Tensor(step_mass[:-1])

            return spec_processed, tgt, seq_peak_labels[1:], idx
        elif self.cfg.task == 'sequence_generation':
            seq = spec_head['Annotated Sequence'].replace('L', 'I').replace(' ', '')
            seq_id = self.seq2id(seq)
            step_mass = torch.Tensor(np.insert(Residual_seq(seq).step_mass[:-1], 0, 0.0))

            tgt = {}
            tgt['tgt'] = torch.Tensor(seq_id[:-1])
            tgt['step_mass'] = torch.Tensor(step_mass)

            seq_labels = torch.Tensor(seq_id[1:])
            return spec_processed, tgt, seq_labels, idx
        else:
            raise NotImplementedError
    
    def __len__(self):
        return len(self.spec_header)

    def seq2id(self, seq):
        return torch.LongTensor([self.aa_id['<bos>']]+[self.aa_id[aa] for aa in seq])

    def seq2multiid(self, seq):
        return torch.LongTensor([self.multiaa_id['<bos>']]+[self.multiaa_id[fragment] for fragment in seq])

