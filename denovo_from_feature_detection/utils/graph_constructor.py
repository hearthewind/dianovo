import torch

import sys

from utils.data.processing_ms1 import create_interpolated_ms1
from utils.data.processing_ms2 import bin_one_peptide

sys.path.append('../')


class GraphGenerator:
    def __init__(self,
                 scan_size=5):
        self.scan_size = scan_size
        
    def __call__(self, precursor, ms1_df, ms2_df):
        ms1_mzs, ms1_xgrams, ms1_features = create_interpolated_ms1(precursor, ms1_df, ms2_df, scan_size=self.scan_size)
        ms2_mzs, ms2_xgrams, ms2_features = bin_one_peptide(precursor, ms2_df, scan_size=self.scan_size)

        return torch.Tensor(ms1_mzs), torch.Tensor(ms1_xgrams), torch.Tensor(ms1_features), \
               torch.Tensor(ms2_mzs), torch.Tensor(ms2_xgrams), torch.Tensor(ms2_features)


graph_gen = GraphGenerator()


