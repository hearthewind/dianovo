import gzip
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from rnova.data.dataset_gnova import GenovaDataset
from utils.data.BasicClass import Residual_seq, Ion, Composition

astral_filtering = True

class RNovaDataset(Dataset):
    def __init__(self, cfg, spec_header, dataset_dir_path):
        super().__init__()
        self.spec_header = spec_header
        self.dataset_dir_path = dataset_dir_path
        self.cfg = cfg
        self.model_ion_tokenize = {f'{charge}{ion}': j for i, charge in enumerate(range(1, self.cfg.data.product_max_charge+1),start=2) for j, ion in enumerate(self.cfg.data.n_term_ion+self.cfg.data.c_term_ion,start=i)}

        self.aa_id = {aa: i for i, aa in enumerate(Residual_seq.output_aalist(), start=3)}
        self.aa_id['<pad>'] = 0
        self.aa_id['<bos>'] = 1
        self.aa_id['<eos>'] = 2

        self.gnova_dataset = GenovaDataset(self.cfg)

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

        rnova = self.process_rnova(idx, spec_head, spec)
        gnova = self.gnova_dataset.process_gnova(idx, spec_head, spec)

        return {'rnova': rnova, 'gnova': gnova}

    def __len__(self):
        return len(self.spec_header)

    def process_rnova(self, idx, spec_head, spec):
        ms1_mzs = spec['ms1_mzs']
        ms1_xgrams = spec['ms1_xgrams']
        ms1_features = spec['ms1_features']
        ms2_mzs = spec['ms2_mzs']
        ms2_xgrams = spec['ms2_xgrams']
        ms2_features = spec['ms2_features']

        seq = spec_head['Annotated Sequence']
        precursor_charge = int(spec_head['Charge'])

        precursor_mass = Residual_seq(seq).mass
        ms2_spectrum_mz, ms2_spectrum_xgrams, ms2_spectrum_features, ms2_multiscan_indices = self.ms2_read_combine(
            ms2_mzs, ms2_xgrams, ms2_features)
        ms1_spectrum_mz, ms1_spectrum_xgrams, ms1_spectrum_features, ms1_multiscan_indices = self.ms1_read_combine(
            ms1_mzs, ms1_xgrams, ms1_features)

        ms1_spectrum_mz, \
            ms1_spectrum_xgrams, \
            ms1_spectrum_features, \
            ms1_charge_index, \
            ms1_pos_index, \
            ms1_sort_indices = self.ms1_preprocessing(ms1_spectrum_mz, ms1_spectrum_xgrams, ms1_spectrum_features, \
                                                         precursor_mass, precursor_charge)
        ms2_spectrum_mz, \
            ms2_spectrum_xgrams, \
            ms2_spectrum_features, \
            ms2_spectrum_ion_class_index, \
            ms2_spectrum_pos_index, \
            ms2_sort_indices = self.ms2_preprocessing(ms2_spectrum_mz, ms2_spectrum_xgrams, ms2_spectrum_features, \
                                                         precursor_mass, precursor_charge)

        mz, xgram, feature, peak_class_index, pos_index, ms1_ms2_flag = self.ms1_ms2_combination(ms1_spectrum_mz,
                                                                        ms1_spectrum_xgrams,
                                                                        ms1_spectrum_features,
                                                                        ms1_charge_index,
                                                                        ms1_pos_index,
                                                                        ms2_spectrum_mz,
                                                                        ms2_spectrum_xgrams,
                                                                        ms2_spectrum_features,
                                                                        ms2_spectrum_ion_class_index,
                                                                        ms2_spectrum_pos_index)

        spec_processed = {'moverz': mz,
                          'xgram': xgram,
                          'feature': feature,
                          'peak_class_index': peak_class_index,
                          'pos_index': pos_index,
                          'ms1_ms2_flag': ms1_ms2_flag}

        meta_info = {
            'precursor_charge': precursor_charge,
            'indices': {
                'ms1_sort': ms1_sort_indices,
                'ms2_sort': ms2_sort_indices,
                'ms1_multiscan': ms1_multiscan_indices,
                'ms2_multiscan': ms2_multiscan_indices,
            }
        }

        if self.cfg.task == 'node_classification':
            label, label_mask = self.tonkenize(seq, ms2_spectrum_mz)
            label = F.pad(label, (len(ms1_spectrum_mz), 0))
            label_mask = F.pad(label_mask, (len(ms1_spectrum_mz), 0))

            return spec_processed, label, label_mask, meta_info
        elif self.cfg.task == 'optimal_path':
            candidate_step_mass = torch.Tensor(np.insert(Residual_seq(seq).step_mass, 0, 0.0))
            graph_label = self.tokenize_sequential(seq, ms2_spectrum_mz)

            nonzero_entries = graph_label.any(-1)
            step_mass = candidate_step_mass[nonzero_entries]
            graph_label = graph_label[nonzero_entries]

            graph_label = F.pad(graph_label,
                                (len(ms1_spectrum_mz), 0))  # [seq_len, ms1_spectrum_mz.size + ms2_spectrum_mz.size]
            label_mask = torch.ones_like(graph_label, dtype=torch.bool)
            label_mask[:, :len(ms1_spectrum_mz)] = 0

            tgt = {}
            tgt['tgt'] = graph_label[:-1]
            tgt['step_mass'] = step_mass[:-1]
            return spec_processed, tgt, graph_label[1:], label_mask[1:], idx, meta_info
        elif self.cfg.task == 'sequence_generation':
            seq = spec_head['Annotated Sequence'].replace('L', 'I').replace(' ', '')
            seq_id = self.seq2id(seq)
            step_mass = torch.Tensor(np.insert(Residual_seq(seq).step_mass[:-1], 0, 0.0))

            tgt = {}
            tgt['tgt'] = torch.Tensor(seq_id[:-1])
            tgt['step_mass'] = torch.Tensor(step_mass)

            seq_labels = torch.Tensor(seq_id[1:])
            return spec_processed, tgt, seq_labels, idx, meta_info
        else:
            raise NotImplementedError

    def ms1_read_combine(self, ms1_spectrum_mz, ms1_spectrum_xgrams, ms1_spectrum_features):
        ms1_spectrum_mz, indices = ms1_spectrum_mz.sort()
        ms1_spectrum_xgrams = ms1_spectrum_xgrams[indices]
        ms1_spectrum_features = ms1_spectrum_features[indices]

        if astral_filtering:
            try:
                ms1_multiscan_indices = torch.sum(ms1_spectrum_xgrams.bool(), dim=1) >= 2
                ms1_spectrum_mz = ms1_spectrum_mz[ms1_multiscan_indices]
                ms1_spectrum_xgrams = ms1_spectrum_xgrams[ms1_multiscan_indices]
                ms1_spectrum_features = ms1_spectrum_features[ms1_multiscan_indices]
            except IndexError as _:
                ms1_multiscan_indices = None
        else:
            ms1_multiscan_indices = None

        return ms1_spectrum_mz, ms1_spectrum_xgrams, ms1_spectrum_features, ms1_multiscan_indices


    def ms2_read_combine(self, ms2_spectrum_mz, ms2_spectrum_xgrams, ms2_spectrum_features):
        ms2_spectrum_mz, indices = ms2_spectrum_mz.sort()
        ms2_spectrum_xgrams = ms2_spectrum_xgrams[indices]
        ms2_spectrum_features = ms2_spectrum_features[indices]

        if astral_filtering:
            ms2_multiscan_indices = torch.sum(ms2_spectrum_xgrams.bool(), dim=1) >= 2
            ms2_spectrum_mz = ms2_spectrum_mz[ms2_multiscan_indices]
            ms2_spectrum_xgrams = ms2_spectrum_xgrams[ms2_multiscan_indices]
            ms2_spectrum_features = ms2_spectrum_features[ms2_multiscan_indices]
        else:
            ms2_multiscan_indices = None

        return ms2_spectrum_mz, ms2_spectrum_xgrams, ms2_spectrum_features, ms2_multiscan_indices
    
    def ms1_preprocessing(self, ms1_spectrum_mz, ms1_spectrum_xgrams, ms1_spectrum_features, \
                          precursor_mass, precursor_charge):
        ms1_spectrum_mz = ms1_spectrum_mz-Composition('proton').mass
        ms1_pos_index = torch.arange(len(ms1_spectrum_mz),dtype=torch.long)

        ms1_charge_index = torch.tensor([i for i in range(1,self.cfg.data.precursor_max_charge+1) for _ in range(len(ms1_spectrum_mz))],dtype=torch.long)
        ms1_spectrum_mz = torch.concat([ms1_spectrum_mz*i-Composition('H2O').mass for i in range(1,self.cfg.data.precursor_max_charge+1)])

        ms1_spectrum_xgrams = torch.concat([ms1_spectrum_xgrams for _ in range(1,self.cfg.data.precursor_max_charge+1)])
        ms1_spectrum_features = torch.concat([ms1_spectrum_features for _ in range(1,self.cfg.data.precursor_max_charge+1)])
        ms1_pos_index = torch.concat([ms1_pos_index for _ in range(self.cfg.data.precursor_max_charge)])

        ms1_spectrum_mz, indices = ms1_spectrum_mz.sort()
        ms1_spectrum_xgrams = ms1_spectrum_xgrams[indices]
        ms1_spectrum_features = ms1_spectrum_features[indices]
        ms1_charge_index = ms1_charge_index[indices]
        ms1_pos_index = ms1_pos_index[indices]

        target_mz_mask = torch.zeros_like(ms1_spectrum_mz, dtype=torch.bool)
        target_mz_index_left = torch.searchsorted(ms1_spectrum_mz, precursor_mass - self.cfg.data.ms1_threshold)
        target_mz_index_tight = torch.searchsorted(ms1_spectrum_mz, precursor_mass + self.cfg.data.ms1_threshold)
        target_mz_mask[target_mz_index_left:target_mz_index_tight] = True
        target_charge_mask = ms1_charge_index == precursor_charge

        target_ms1_mask = torch.logical_and(target_mz_mask, target_charge_mask)
        ms1_charge_index[target_ms1_mask] += 30 # make sure they have distinct embeddings

        return ms1_spectrum_mz, ms1_spectrum_xgrams, ms1_spectrum_features, ms1_charge_index, ms1_pos_index, indices
    
    def ms2_preprocessing(self, ms2_spectrum_mz, ms2_spectrum_xgrams, ms2_spectrum_features, precursor_mass, precursor_charge):
        ms2_spectrum_mz_list = []
        ms2_spectrum_xgrams_list = []
        ms2_spectrum_features_list = []
        ms2_spectrum_ion_class_list = []
        ms2_spectrum_pos_list = []
        for ion in self.cfg.data.n_term_ion:
            for charge in range(1,self.cfg.data.product_max_charge+1):
                if precursor_charge==2 and charge>1: continue

                temp_mz = Ion.peak2sequencemz(ms2_spectrum_mz, f'{charge}{ion}')

                # mask = torch.logical_and(0<temp_mz, temp_mz<precursor_mass)
                mask = temp_mz > -float('inf')
                ms2_spectrum_mz_list += [temp_mz[mask]]
                ms2_spectrum_xgrams_list += [ms2_spectrum_xgrams[mask]]
                ms2_spectrum_features_list += [ms2_spectrum_features[mask]]
                ms2_spectrum_ion_class_list += [torch.tensor([self.model_ion_tokenize[f'{charge}{ion}']]*len(ms2_spectrum_mz),dtype=torch.long)[mask]]
                ms2_spectrum_pos_list += [torch.arange(1,len(ms2_spectrum_mz)+1,dtype=torch.long)[mask]]
        
        for ion in self.cfg.data.c_term_ion:
            for charge in range(1,self.cfg.data.product_max_charge+1):
                if precursor_charge==2 and charge>1: continue

                #For DIA with precursor
                temp_mz = precursor_mass - Ion.peak2sequencemz(ms2_spectrum_mz, f'{charge}{ion}')
                
                #For DIA without precursor
                #temp_mz = Ion.peak2sequencemz(ms2_spectrum_mz, f'{charge}{ion}')

                # mask = torch.logical_and(0<temp_mz, temp_mz<precursor_mass)
                mask = temp_mz > -float('inf')
                ms2_spectrum_mz_list += [temp_mz[mask]]
                ms2_spectrum_xgrams_list += [ms2_spectrum_xgrams[mask]]
                ms2_spectrum_features_list += [ms2_spectrum_features[mask]]
                ms2_spectrum_ion_class_list += [torch.tensor([self.model_ion_tokenize[f'{charge}{ion}']]*len(ms2_spectrum_mz))[mask]]
                ms2_spectrum_pos_list += [torch.arange(1,len(ms2_spectrum_mz)+1)[mask]]

        ms2_spectrum_mz = torch.concat(ms2_spectrum_mz_list)
        ms2_spectrum_xgrams = torch.concat(ms2_spectrum_xgrams_list)
        ms2_spectrum_features = torch.concat(ms2_spectrum_features_list)
        ms2_spectrum_ion_class_index = torch.concat(ms2_spectrum_ion_class_list)
        ms2_spectrum_pos_index = torch.concat(ms2_spectrum_pos_list)

        ms2_spectrum_mz, indices = ms2_spectrum_mz.sort()
        ms2_spectrum_xgrams = ms2_spectrum_xgrams[indices]
        ms2_spectrum_features = ms2_spectrum_features[indices]
        ms2_spectrum_ion_class_index = ms2_spectrum_ion_class_index[indices]
        ms2_spectrum_pos_index = ms2_spectrum_pos_index[indices]

        scan_size = ms2_spectrum_xgrams.shape[1]
        feature_dim = ms2_spectrum_features.shape[2]

        ms2_spectrum_mz = torch.concat([torch.tensor([0]), ms2_spectrum_mz, torch.tensor([precursor_mass])])
        ms2_spectrum_xgrams = torch.concat([torch.zeros(1, scan_size), ms2_spectrum_xgrams, torch.zeros((1, scan_size))])
        ms2_spectrum_features = torch.concat([torch.zeros(1, scan_size, feature_dim), ms2_spectrum_features, torch.zeros(1, scan_size, feature_dim)])
        ms2_spectrum_ion_class_index = torch.concat([torch.tensor([0]), ms2_spectrum_ion_class_index, torch.tensor([1])])
        ms2_spectrum_pos_index = torch.concat([torch.tensor([0]), ms2_spectrum_pos_index, torch.tensor([ms2_spectrum_pos_index.max()+1])])


        return ms2_spectrum_mz, ms2_spectrum_xgrams, ms2_spectrum_features, ms2_spectrum_ion_class_index, ms2_spectrum_pos_index, indices

    def ms1_ms2_combination(self, ms1_spectrum_mz,
                            ms1_spectrum_xgrams,
                            ms1_spectrum_features,
                            ms1_charge_index,
                            ms1_pos_index,
                            ms2_spectrum_mz,
                            ms2_spectrum_xgrams,
                            ms2_spectrum_features,
                            ms2_spectrum_ion_class_index,
                            ms2_spectrum_pos_index):
        mz = torch.concat([ms1_spectrum_mz,ms2_spectrum_mz])
        xgram = torch.concat([ms1_spectrum_xgrams,ms2_spectrum_xgrams])
        feature = torch.concat([ms1_spectrum_features,ms2_spectrum_features])
        peak_class_index = torch.concat([ms1_charge_index,ms2_spectrum_ion_class_index])
        pos_index = torch.concat([ms1_pos_index,ms2_spectrum_pos_index])
        ms1_ms2_flag = torch.tensor([0]*len(ms1_spectrum_mz)+[1]*len(ms2_spectrum_mz))

        return mz, xgram, feature, peak_class_index, pos_index, ms1_ms2_flag
        
    def tonkenize(self, seq, ms2_spectrum_mz):
        label = torch.zeros_like(ms2_spectrum_mz, dtype=torch.float)
        label_mass_lower_bound = ms2_spectrum_mz.numpy().searchsorted(Residual_seq(seq).step_mass[:-1]-self.cfg.data.ms2_threshold)
        label_mass_upper_bound = ms2_spectrum_mz.numpy().searchsorted(Residual_seq(seq).step_mass[:-1]+self.cfg.data.ms2_threshold)
        for lower, upper in zip(label_mass_lower_bound, label_mass_upper_bound): label[lower:upper] = 1

        label_mask = torch.ones_like(label,dtype=torch.bool)
        label_mask[0] = 0
        label_mask[-1] = 0
        
        return label, label_mask

    def tokenize_sequential(self, seq, ms2_spectrum_mz):
        theo_node_mass = Residual_seq(seq).step_mass[:-1]
        start_index = ms2_spectrum_mz.numpy().searchsorted(theo_node_mass - self.cfg.data.ms2_threshold)
        end_index = ms2_spectrum_mz.numpy().searchsorted(theo_node_mass + self.cfg.data.ms2_threshold)
        graph_label = np.zeros((ms2_spectrum_mz.numpy().size, theo_node_mass.size))

        for i, (lower_bound, higher_bound) in enumerate(zip(start_index, end_index)):
            graph_label[:,i][lower_bound:higher_bound] = 1

        graph_label = np.concatenate([np.zeros((len(ms2_spectrum_mz), 1)), graph_label, np.zeros((len(ms2_spectrum_mz), 1))], axis=1)

        graph_label[0, 0] = 1
        graph_label[-1, -1] = 1

        return torch.IntTensor(graph_label.T)  # [theo_node_mass.size, ms2_spectrum_mz.size]

    def seq2id(self, seq):
        return torch.LongTensor([self.aa_id['<bos>']]+[self.aa_id[aa] for aa in seq])
