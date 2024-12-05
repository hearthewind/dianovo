import numpy as np

import sys
sys.path.append('../../../')
from utils.data.configs import mz_max

class PeakFeatureGeneration:
    def __init__(self, local_sliding_window=50, data_acquisition_upper_limit=mz_max):
        self.local_sliding_window = local_sliding_window
        self.data_acquisition_upper_limit = data_acquisition_upper_limit

    def __call__(self, product_ions_moverz, product_ions_intensity):
        positive_mask = product_ions_moverz > 0
        product_ions_moverz = product_ions_moverz[positive_mask]
        product_ions_intensity = product_ions_intensity[positive_mask]

        normalize_moverz = self.normalize_moverzCal(product_ions_moverz)
        relative_intensity = self.relative_intensityCal(product_ions_intensity)
        total_rank = self.total_rankCal(product_ions_intensity)
        total_halfrank = self.total_halfrankCal(product_ions_intensity)
        local_mask = self.local_intensity_mask(product_ions_moverz)
        local_significant = self.local_significantCal(local_mask, product_ions_intensity)
        local_rank = self.local_rankCal(local_mask, product_ions_intensity)
        local_halfrank = self.local_halfrankCal(local_mask, product_ions_intensity)
        local_reletive_intensity = self.local_reletive_intensityCal(local_mask, product_ions_intensity)

        product_ions_feature = np.stack([normalize_moverz,
                                         relative_intensity,
                                         local_significant,
                                         total_rank,
                                         total_halfrank,
                                         local_rank,
                                         local_halfrank,
                                         local_reletive_intensity]).transpose()

        return product_ions_feature  # shape: (num_peaks, num_features), num_features = 8

    def normalize_moverzCal(self, moverz):
        return np.exp(-moverz / self.data_acquisition_upper_limit)

    def relative_intensityCal(self, intensity):
        return intensity / intensity.max()

    def local_intensity_mask(self, mz):
        right_boundary = np.reshape(mz + self.local_sliding_window, (-1, 1))
        left_boundary = np.reshape(mz - self.local_sliding_window, (-1, 1))
        mask = np.logical_and(right_boundary > mz, left_boundary < mz)
        return mask

    def local_significantCal(self, mask,
                             intensity):  # This feature need to be fixed use signal to ratio to replace intensity.
        # 这个feature为了要映射到[1,+infinity)并且不让tan在正无穷和负无穷之间来回横跳，特意在最小intentisy的基础上减了0.5
        # 让原始值到不了1
        local_significant = []
        for i in range(len(intensity)):
            local_intensity_list = intensity[mask[i]]
            local_significant.append(np.tanh((intensity[i] / local_intensity_list.min() - 1) / 2))
        return np.array(local_significant)

    def local_rankCal(self, mask, intensity):
        local_rank = []
        for i in range(len(intensity)):
            local_intensity_list = intensity[mask[i]]
            local_rank.append(np.sum(intensity[i] > local_intensity_list) / len(local_intensity_list))
        return np.array(local_rank)

    def local_halfrankCal(self, mask, intensity):
        local_halfrank = []
        for i in range(len(intensity)):
            local_intensity_list = intensity[mask[i]]
            local_halfrank.append(np.sum(intensity[i] / 2 > local_intensity_list) / len(local_intensity_list))
        return np.array(local_halfrank)

    def local_reletive_intensityCal(self, mask, intensity):
        local_reletive_intensity = []
        for i in range(len(intensity)):
            local_intensity_list = intensity[mask[i]]
            local_reletive_intensity.append(intensity[i] / local_intensity_list.max())
        return np.array(local_reletive_intensity)

    def total_rankCal(self, intensity):
        temp_intensity = intensity.reshape((-1, 1))
        return np.sum(temp_intensity > intensity, axis=1) / len(intensity)

    def total_halfrankCal(self, intensity):
        half_intensity = intensity / 2
        half_intensity = half_intensity.reshape((-1, 1))
        return np.sum(half_intensity > intensity, axis=1) / len(intensity)