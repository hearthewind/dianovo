import torch

import sys

from utils.cofragment_peptide_processing import process_one_peptide_and_cofragment
from utils.data.processing_ms1 import create_interpolated_ms1
from utils.data.processing_ms2_cy import bin_one_peptide

sys.path.append('../')


class GraphGenerator:
    def __init__(self,
                 scan_size=5):
        self.scan_size = scan_size
        
    def __call__(self, peptide, ms1_df, ms2_df, diann_result):
        ms1_mzs, ms1_xgrams, ms1_features = create_interpolated_ms1(peptide, ms1_df, ms2_df, scan_size=self.scan_size)
        ms2_mzs, ms2_xgrams, ms2_features = bin_one_peptide(peptide, ms2_df, scan_size=self.scan_size)

        cofragment_ms2s, cofragment_ms1s, target_ms2s, target_ms1s = process_one_peptide_and_cofragment(peptide, diann_result, ms2_df, ms2_mzs, ms1_mzs)
        for i, x in enumerate(cofragment_ms2s):
            cofragment_ms2s[i] = torch.tensor(x)
        for i, x in enumerate(cofragment_ms1s):
            cofragment_ms1s[i] = torch.tensor(x)
        target_ms2s = torch.tensor(target_ms2s)
        target_ms1s = torch.tensor(target_ms1s)

        return torch.Tensor(ms1_mzs), torch.Tensor(ms1_xgrams), torch.Tensor(ms1_features), \
               torch.Tensor(ms2_mzs), torch.Tensor(ms2_xgrams), torch.Tensor(ms2_features), \
               cofragment_ms1s, cofragment_ms2s, target_ms1s, target_ms2s


graph_gen = GraphGenerator()


