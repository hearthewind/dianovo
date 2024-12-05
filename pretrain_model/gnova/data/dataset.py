import os
import gzip

import numpy as np
import torch
import pickle
from torch.utils.data import Dataset

from utils.cofragment_peptide_processing import ion_types, label_dict
from utils.data.BasicClass import Residual_seq, Ion
from utils.data.theo_peak_information import get_theoretical_peaks


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

        self.ion_types = ion_types
        self.label_dict = label_dict

        self.mass_error_da = self.cfg.data.ms2_threshold

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

        ms1_mzs = spec['ms1_mzs']
        ms1_xgrams = spec['ms1_xgrams']
        ms1_features = spec['ms1_features']
        ms2_mzs = spec['ms2_mzs']
        ms2_xgrams = spec['ms2_xgrams']
        ms2_features = spec['ms2_features']

        target_ms1_labels = spec['target_ms1_labels']
        cofragment_ms1_labels = spec['cofragment_ms1_labels']
        target_ms2_labels = spec['target_ms2_labels']
        cofragment_ms2_labels = spec['cofragment_ms2_labels']

        seq = spec_head['Annotated Sequence']
        precursor_charge = int(spec_head['Charge'])
        precursor_mz = spec_head['m/z']
        precursor_mass = Ion.precursorion2mass(precursor_mz, precursor_charge)

        ms2_spectrum_mz, ms2_spectrum_xgrams, ms2_spectrum_features, target_ms2_labels, cofragment_ms2_labels = self._ms2_read_combine(ms2_mzs, ms2_xgrams, ms2_features, target_ms2_labels, cofragment_ms2_labels)
        ms1_spectrum_mz, ms1_spectrum_xgrams, ms1_spectrum_features, target_ms1_labels, cofragment_ms1_labels = self._ms1_read_combine(ms1_mzs, ms1_xgrams, ms1_features, target_ms1_labels, cofragment_ms1_labels)

        ms1_iontype_label = self._cofragment_iontype_proessing(target_ms1_labels, cofragment_ms1_labels)
        ms1_ionsource_label = self._cofragment_ionsource_processing(target_ms1_labels, cofragment_ms1_labels)
        ms2_iontype_label = self._cofragment_iontype_proessing(target_ms2_labels, cofragment_ms2_labels)
        ms2_ionsource_label = self._cofragment_ionsource_processing(target_ms2_labels, cofragment_ms2_labels)

        ms1_charge_index = (self.cfg.data.precursor_max_charge + 1) * torch.ones_like(ms1_spectrum_mz)
        target_mz_mask = torch.zeros_like(ms1_spectrum_mz, dtype=torch.bool)
        target_mz_index_left = torch.searchsorted(ms1_spectrum_mz, precursor_mz - self.cfg.data.ms1_threshold)
        target_mz_index_tight = torch.searchsorted(ms1_spectrum_mz, precursor_mz + self.cfg.data.ms1_threshold)
        target_mz_mask[target_mz_index_left:target_mz_index_tight] = True
        ms1_charge_index[target_mz_mask] = min(precursor_charge, self.cfg.data.precursor_max_charge)

        ms2_charge_index = (self.cfg.data.precursor_max_charge + 2) * torch.ones_like(ms2_spectrum_mz)

        mz, xgram, feature, peak_flag_index, \
            ion_type_label, ion_source_label = self._ms1_ms2_combination(ms1_spectrum_mz,
                                                                         ms1_spectrum_xgrams,
                                                                         ms1_spectrum_features,
                                                                         ms1_charge_index,
                                                                         ms2_spectrum_mz,
                                                                         ms2_spectrum_xgrams,
                                                                         ms2_spectrum_features,
                                                                         ms2_charge_index,
                                                                         ms1_iontype_label,
                                                                         ms1_ionsource_label,
                                                                         ms2_iontype_label,
                                                                         ms2_ionsource_label)

        spec_processed = {'moverz': mz,
                          'xgram': xgram,
                          'feature': feature,
                          'peak_flag_index': peak_flag_index}

        cofragment_labels = {'ion_type': ion_type_label,
                             'ion_source': ion_source_label}

        if self.cfg.task == 'pretrain':
            return spec_processed, cofragment_labels, idx
        elif self.cfg.task == 'peak_classification':
            label = torch.cat([target_ms1_labels, target_ms2_labels])
            # label = self._generate_peak_label(seq, precursor_charge, ms2_spectrum_mz, ms1_spectrum_mz) #TODO(m) maybe performance gain?

            return spec_processed, cofragment_labels, label
        elif self.cfg.task == 'optimal_path':
            seq = seq.replace('L', 'I').replace(' ', '')
            candidate_step_mass = torch.Tensor(np.insert(Residual_seq(seq).step_mass, 0, 0.0))

            seq_peak_labels = self._generate_sequential_peak_labels(seq, precursor_charge, ms2_spectrum_mz, ms1_spectrum_mz)
            seq_len, peak_num = seq_peak_labels.shape

            tmp_peak_labels = []
            tmp_step_mass = []
            for l in range(seq_len + 1):
                current_peak_label = seq_peak_labels[l, :]
                if current_peak_label.any():
                    tmp_peak_labels.append(current_peak_label.unsqueeze(0))
                    tmp_step_mass.append(candidate_step_mass[l])
            seq_peak_labels = torch.cat(tmp_peak_labels, 0)
            step_mass = torch.Tensor(tmp_step_mass)

            seq_peak_labels = seq_peak_labels > 0

            tgt = {}
            tgt['tgt'] = torch.Tensor(seq_peak_labels[:-1])
            tgt['step_mass'] = torch.Tensor(step_mass[:-1])

            return spec_processed, tgt, seq_peak_labels[1:], idx
        elif self.cfg.task == 'sequence_generation':
            seq = seq.replace('L', 'I').replace(' ', '')
            seq_id = self._seq2id(seq)
            step_mass = torch.Tensor(np.insert(Residual_seq(seq).step_mass[:-1], 0, 0.0))

            tgt = {}
            tgt['tgt'] = torch.Tensor(seq_id[:-1])
            tgt['step_mass'] = torch.Tensor(step_mass)

            seq_labels = torch.Tensor(seq_id[1:])

            return spec_processed, tgt, seq_labels, idx
        else:
            raise NotImplementedError

    def _generate_peak_label(self, mod_seq: str, charge: int, ms2_mzs: torch.Tensor, ms1_mzs: torch.Tensor):
        # precursor_mass should be theoretical
        precursor_mass_theo = Residual_seq(mod_seq).mass
        precursor_mz = Ion.precursormass2ion(precursor_mass_theo, charge)

        mass_threshold = self.mass_error_da # + self.mass_error_ppm*precursor_mass*1e-6

        theo_fragments = []
        for location in range(0, len(mod_seq) - 1):
            theoretical_peaks = get_theoretical_peaks(mod_seq, precursor_mass_theo, location)
            for ion in self.ion_types:
                mz = theoretical_peaks[ion]
                if ion == '1a' and mz > 300.0: continue
                elif ion == '2y' and mz < 400.0: continue

                theo_fragments.append((mz, ion))

        ms2_labels = []
        for mz in ms2_mzs:
            fragment = False
            for theo_mz, ion in theo_fragments:
                if mz - mass_threshold <= theo_mz <= mz + mass_threshold:
                    ms2_labels.append(ion)
                    fragment = True
                    break
            if not fragment:
                ms2_labels.append('noise')

        ms1_labels = []
        for mz in ms1_mzs:
            precursor = False
            if mz - mass_threshold <= precursor_mz <= mz + mass_threshold:
                ms1_labels.append('ms1')
                precursor = True
            if not precursor:
                ms1_labels.append('noise')

        ms2_labels = [self.label_dict[label] for label in ms2_labels]
        ms1_labels = [self.label_dict[label] for label in ms1_labels]
        ret = torch.tensor(ms1_labels + ms2_labels)
        return ret

    def _generate_sequential_peak_labels(self, mod_seq: str, charge: int, ms2_mzs: torch.Tensor, ms1_mzs: torch.Tensor):
        precursor_mass_theo = Residual_seq(mod_seq).mass
        precursor_mz_theo = Ion.precursormass2ion(precursor_mass_theo, charge)

        mass_threshold = self.mass_error_da

        num_ms2_peaks = len(ms2_mzs)
        num_ms1_peaks = len(ms1_mzs)
        ret = np.zeros((num_ms1_peaks + num_ms2_peaks, len(mod_seq)))

        for location in range(0, len(mod_seq) - 1):
            theo_fragments = []
            theoretical_peaks = get_theoretical_peaks(mod_seq, precursor_mass_theo, location)
            for ion in self.ion_types:
                theo_mz = theoretical_peaks[ion]
                if ion == '1a' and theo_mz > 300.0:
                    continue
                elif ion == '2y' and theo_mz < 400.0:
                    continue

                theo_fragments.append((theo_mz, ion))

            ms2_labels = []
            for mz in ms2_mzs:
                fragment = False
                for theo_mz, ion in theo_fragments:
                    if mz - mass_threshold <= theo_mz <= mz + mass_threshold:
                        ms2_labels.append(ion)
                        fragment = True
                        break
                if not fragment:
                    ms2_labels.append('noise')
            ms2_labels = [self.label_dict[label] for label in ms2_labels]
            ms1_labels = [self.label_dict['noise']] * num_ms1_peaks
            tot_labels = np.array(ms1_labels + ms2_labels)
            ret[:, location] = tot_labels

        ms2_labels = []
        for mz in ms2_mzs:
            if mz - mass_threshold <= precursor_mz_theo <= mz + mass_threshold:
                ms2_labels.append('ms1')
            else:
                ms2_labels.append('noise')
        ms2_labels = [self.label_dict[label] for label in ms2_labels]

        ms1_labels = []
        for mz in ms1_mzs:
            if mz - mass_threshold <= precursor_mz_theo <= mz + mass_threshold:
                ms1_labels.append('ms1')
            else:
                ms1_labels.append('noise')
        ms1_labels = [self.label_dict[label] for label in ms1_labels]

        tot_labels = np.array(ms1_labels + ms2_labels)
        ret[:, len(mod_seq) - 1] = tot_labels

        # for the last graph_label, the one corresponds to seq_len, the labels for the additional ms1 nodes should also be preursor
        # this is used for determing when to stop optimal path decode

        ret = ret.transpose() # (seq_len, peak_num)
        return torch.tensor(ret)

    def _ms2_read_combine(self, ms2_spectrum_mz, ms2_spectrum_xgrams, ms2_spectrum_features, \
                          target_ms2_labels, cofragment_ms2_labels):
        ms2_spectrum_mz, indices = ms2_spectrum_mz.sort()
        ms2_spectrum_xgrams = ms2_spectrum_xgrams[indices]
        ms2_spectrum_features = ms2_spectrum_features[indices]

        target_ms2_labels = target_ms2_labels[indices]
        for i, x in enumerate(cofragment_ms2_labels):
            cofragment_ms2_labels[i] = x[indices]

        return ms2_spectrum_mz, ms2_spectrum_xgrams, ms2_spectrum_features, \
               target_ms2_labels, cofragment_ms2_labels

    def _ms1_read_combine(self, ms1_spectrum_mz, ms1_spectrum_xgrams, ms1_spectrum_features, \
                          target_ms1_labels, cofragment_ms1_labels):
        ms1_spectrum_mz, indices = ms1_spectrum_mz.sort()
        ms1_spectrum_xgrams = ms1_spectrum_xgrams[indices]
        ms1_spectrum_features = ms1_spectrum_features[indices]

        target_ms1_labels = target_ms1_labels[indices]
        for i, x in enumerate(cofragment_ms1_labels):
            cofragment_ms1_labels[i] = x[indices]

        return ms1_spectrum_mz, ms1_spectrum_xgrams, ms1_spectrum_features, \
               target_ms1_labels, cofragment_ms1_labels

    def _cofragment_iontype_proessing(self, target_labels, cofragment_labels):
        final_ion_label = torch.zeros_like(target_labels)

        for i in range(len(cofragment_labels)):
            cofragment = cofragment_labels[i]

            for j in range(len(cofragment)):
                if cofragment[j] > 0:
                    final_ion_label[j] = cofragment[j]

        for j in range(len(target_labels)):
            if target_labels[j] > 0:
                final_ion_label[j] = target_labels[j]

        return final_ion_label

    def _cofragment_ionsource_processing(self, target_labels, cofragment_labels):
        final_source_label = torch.zeros_like(target_labels)

        for i, cofragment in enumerate(cofragment_labels, start=2):
            ion_indices = cofragment > 0
            final_source_label[ion_indices] = i

        ion_indices = target_labels > 0
        final_source_label[ion_indices] = 1
        return final_source_label

    def _ms1_ms2_combination(self, ms1_spectrum_mz,
                             ms1_spectrum_xgrams,
                             ms1_spectrum_features,
                             ms1_charge_index,
                             ms2_spectrum_mz,
                             ms2_spectrum_xgrams,
                             ms2_spectrum_features,
                             ms2_charge_index,
                             ms1_iontype_label,
                             ms1_ionsource_label,
                             ms2_iontype_label,
                             ms2_ionsource_label):
        mz = torch.concat([ms1_spectrum_mz,ms2_spectrum_mz])
        xgram = torch.concat([ms1_spectrum_xgrams,ms2_spectrum_xgrams])
        feature = torch.concat([ms1_spectrum_features,ms2_spectrum_features])
        peak_flag_index = torch.concat([ms1_charge_index,ms2_charge_index])
        iontype_label = torch.cat([ms1_iontype_label,ms2_iontype_label])
        ionsource_label = torch.cat([ms1_ionsource_label, ms2_ionsource_label])

        return mz, xgram, feature, peak_flag_index, \
               iontype_label, ionsource_label
    
    def __len__(self):
        return len(self.spec_header)

    def _seq2id(self, seq):
        return torch.LongTensor([self.aa_id['<bos>']]+[self.aa_id[aa] for aa in seq])

