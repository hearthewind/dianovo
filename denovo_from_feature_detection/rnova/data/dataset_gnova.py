import torch

from utils.cofragment_peptide_processing import ion_types, label_dict
from utils.data.BasicClass import Residual_seq, Ion


class GenovaDataset():
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.aa_id = {aa: i for i, aa in enumerate(Residual_seq.output_aalist(), start=3)}
        self.aa_id['<pad>'] = 0
        self.aa_id['<bos>'] = 1
        self.aa_id['<eos>'] = 2

        self.ion_types = ion_types
        self.label_dict = label_dict

        self.mass_error_da = self.cfg.data.ms2_threshold

    def process_gnova(self, idx, spec_head, spec):

        ms1_mzs = spec['ms1_mzs']
        ms1_xgrams = spec['ms1_xgrams']
        ms1_features = spec['ms1_features']
        ms2_mzs = spec['ms2_mzs']
        ms2_xgrams = spec['ms2_xgrams']
        ms2_features = spec['ms2_features']

        precursor_charge = int(spec_head['Charge'])
        precursor_mz = float(spec_head['m/z'])
        precursor_mass = Ion.precursorion2mass(precursor_mz, precursor_charge)

        ms2_spectrum_mz, ms2_spectrum_xgrams, ms2_spectrum_features = self._ms2_read_combine(ms2_mzs, ms2_xgrams, ms2_features)
        ms1_spectrum_mz, ms1_spectrum_xgrams, ms1_spectrum_features = self._ms1_read_combine(ms1_mzs, ms1_xgrams, ms1_features)

        ms1_charge_index = (self.cfg.data.precursor_max_charge + 1) * torch.ones_like(ms1_spectrum_mz)
        target_mz_mask = torch.zeros_like(ms1_spectrum_mz, dtype=torch.bool)
        target_mz_index_left = torch.searchsorted(ms1_spectrum_mz, precursor_mz - self.cfg.data.ms1_threshold)
        target_mz_index_tight = torch.searchsorted(ms1_spectrum_mz, precursor_mz + self.cfg.data.ms1_threshold)
        target_mz_mask[target_mz_index_left:target_mz_index_tight] = True
        ms1_charge_index[target_mz_mask] = min(precursor_charge, self.cfg.data.precursor_max_charge)

        ms2_charge_index = (self.cfg.data.precursor_max_charge + 2) * torch.ones_like(ms2_spectrum_mz)

        mz, xgram, feature, peak_flag_index = self._ms1_ms2_combination(ms1_spectrum_mz,
                                                                         ms1_spectrum_xgrams,
                                                                         ms1_spectrum_features,
                                                                         ms1_charge_index,
                                                                         ms2_spectrum_mz,
                                                                         ms2_spectrum_xgrams,
                                                                         ms2_spectrum_features,
                                                                         ms2_charge_index)

        spec_processed = {'moverz': mz,
                          'xgram': xgram,
                          'feature': feature,
                          'peak_flag_index': peak_flag_index}


        sizes = {'ms1': len(ms1_mzs), 'ms2': len(ms2_mzs)}

        return spec_processed, idx, sizes

    def _ms2_read_combine(self, ms2_spectrum_mz, ms2_spectrum_xgrams, ms2_spectrum_features):
        ms2_spectrum_mz, indices = ms2_spectrum_mz.sort()
        ms2_spectrum_xgrams = ms2_spectrum_xgrams[indices]
        ms2_spectrum_features = ms2_spectrum_features[indices]

        return ms2_spectrum_mz, ms2_spectrum_xgrams, ms2_spectrum_features

    def _ms1_read_combine(self, ms1_spectrum_mz, ms1_spectrum_xgrams, ms1_spectrum_features):
        ms1_spectrum_mz, indices = ms1_spectrum_mz.sort()
        ms1_spectrum_xgrams = ms1_spectrum_xgrams[indices]
        ms1_spectrum_features = ms1_spectrum_features[indices]

        return ms1_spectrum_mz, ms1_spectrum_xgrams, ms1_spectrum_features


    def _ms1_ms2_combination(self, ms1_spectrum_mz,
                             ms1_spectrum_xgrams,
                             ms1_spectrum_features,
                             ms1_charge_index,
                             ms2_spectrum_mz,
                             ms2_spectrum_xgrams,
                             ms2_spectrum_features,
                             ms2_charge_index):
        mz = torch.concat([ms1_spectrum_mz,ms2_spectrum_mz])
        xgram = torch.concat([ms1_spectrum_xgrams,ms2_spectrum_xgrams])
        feature = torch.concat([ms1_spectrum_features,ms2_spectrum_features])
        peak_flag_index = torch.concat([ms1_charge_index,ms2_charge_index])

        return mz, xgram, feature, peak_flag_index

