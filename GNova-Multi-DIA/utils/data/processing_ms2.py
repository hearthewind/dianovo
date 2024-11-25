import numpy as np
import pandas as pd

import sys
sys.path.append('../../')
from utils.data.configs import mz_max, num_bins
from utils.data.peptide import get_ms2_from_precursor
from utils.data.processing_utils import bin_one_spectrum, find_best_scan_index, feature_gen


def bin_one_peptide(peptide: dict, ms2_df: pd.DataFrame, scan_size=5, neighboring=True):
    related_ms2 = get_ms2_from_precursor(peptide, ms2_df)
    scan_count = len(related_ms2)
    rt_peak = peptide['rt']

    best_scan_index = find_best_scan_index(related_ms2, rt_peak)

    scan_center = best_scan_index
    scan_left_count = scan_center
    scan_right_count = scan_count - scan_left_count - 1
    scan_size_half = scan_size // 2
    scan_left_count = min(scan_left_count, scan_size_half)
    scan_right_count = min(scan_right_count, scan_size_half)

    binned_spectra = []
    peak_feature_dicts = []

    if scan_left_count < scan_size_half:
        for _ in range(scan_size_half - scan_left_count):
            binned_spectra.append(np.zeros(shape=num_bins, dtype=np.float32))
            peak_feature_dicts.append({})

    for index in range(scan_center - scan_left_count, scan_center + scan_right_count + 1):
        ms2 = related_ms2.iloc[index]
        binned_spectrum = bin_one_spectrum(ms2)
        binned_spectra.append(np.array(binned_spectrum))
        peak_feature_dicts.append(gen_feature_for_one_binned_spectrum(binned_spectrum))

    if scan_right_count < scan_size_half:
        for _ in range(scan_size_half - scan_right_count):
            binned_spectra.append(np.zeros(shape=num_bins, dtype=np.float32))
            peak_feature_dicts.append({})

    binned_spectra = np.array(binned_spectra)

    def get_feature(mz, feature_dict):
        return feature_dict.get(mz, np.zeros(8))

    ret = []
    for i in range(num_bins):
        mz = (i + 0.5) * mz_max / num_bins
        intensities = binned_spectra[:, i]

        if any(intensities > 0):

            if neighboring:
                try:
                    intensities = binned_spectra[:, i] + binned_spectra[:, i - 1] + binned_spectra[:, i + 1]
                except IndexError:
                    try:
                        intensities = binned_spectra[:, i] + binned_spectra[:, i - 1]
                    except IndexError:
                        intensities = binned_spectra[:, i] + binned_spectra[:, i + 1]
            else:
                intensities = binned_spectra[:, i]

            feature_vectors = []
            for j in range(scan_size):
                feature_dict = peak_feature_dicts[j]
                feature_vec = get_feature(mz, feature_dict)
                feature_vectors.append(feature_vec)

            feature_vectors = np.array(feature_vectors)

            ret.append((mz, intensities, feature_vectors))

    ms2_mzs = np.array([x[0] for x in ret])
    ms2_xgrams = np.array([x[1] for x in ret])
    ms2_features = np.array([x[2] for x in ret])

    return ms2_mzs, ms2_xgrams, ms2_features

def gen_feature_for_one_binned_spectrum(binned_spectrum: list):
    mzs = []
    intensities = []

    for i in range(num_bins):
        mz = (i + 0.5) * mz_max / num_bins
        intensity = binned_spectrum[i]
        if intensity > 0:
            mzs.append(mz)
            intensities.append(intensity)

    features = feature_gen(np.array(mzs), np.array(intensities))

    ret = {}
    for i, mz in enumerate(mzs):
        ret[mz] = features[i]

    return ret
