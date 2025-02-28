import copy

import numpy as np
import pandas as pd

import sys

from utils.data.processing_utils import find_best_scan_index, bin_one_spectrum, feature_gen

sys.path.append('../../')
from utils.data.configs import mz_max, num_bins
from utils.data.peptide import get_ms1_from_precursor, get_ms2_from_precursor, filter_ms1_precursor_window


def create_interpolated_ms1(peptide: dict, ms1_df: pd.DataFrame, ms2_df: pd.DataFrame, scan_size=5, neighboring=True):
    related_ms1 = get_ms1_from_precursor(peptide, ms1_df)
    related_ms2 = get_ms2_from_precursor(peptide, ms2_df)
    related_ms1 = filter_ms1_precursor_window(related_ms1, related_ms2)

    all_ms1_rts = [ms1.rt for _, ms1 in related_ms1.iterrows()]
    all_ms2_rts = [ms2.rt for _, ms2 in related_ms2.iterrows()]

    binned_spectra = []

    for i, ms1 in related_ms1.iterrows():
        binned_ms1 = np.array(bin_one_spectrum(ms1))
        binned_spectra.append(binned_ms1)

    binned_spectra = np.array(binned_spectra)

    binned_ms1 = []
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

            binned_ms1.append((mz, intensities))

    interp_ms1 = []
    for mz, intensities in binned_ms1:
        train_ms1_intensities = intensities
        interp_ms1_intensities = interp_one_mz(all_ms1_rts, all_ms2_rts, train_ms1_intensities)
        interp_ms1.append((mz, interp_ms1_intensities))

    feature_dicts = gen_feature_for_binned_ms1(interp_ms1, related_ms2)

    scan_count = len(related_ms2)
    rt_peak = peptide['rt']

    best_scan_index = find_best_scan_index(related_ms2, rt_peak)

    scan_center = best_scan_index
    scan_left_count = scan_center
    scan_right_count = scan_count - scan_left_count - 1
    scan_size_half = scan_size // 2
    scan_left_count = min(scan_left_count, scan_size_half)
    scan_right_count = min(scan_right_count, scan_size_half)

    selected_xgrams = [[] for _ in range(len(interp_ms1))]
    selected_feature_dicts = []

    if scan_left_count < scan_size_half:
        for _ in range(scan_size_half - scan_left_count):
            for i, _ in enumerate(binned_ms1):
                selected_xgrams[i].append(0.0)
            selected_feature_dicts.append({})

    for index in range(scan_center - scan_left_count, scan_center + scan_right_count + 1):
        for i, (_, xgram) in enumerate(interp_ms1):
            selected_xgrams[i].append(xgram[index])

        selected_feature_dicts.append(feature_dicts[index])

    if scan_right_count < scan_size_half:
        for _ in range(scan_size_half - scan_right_count):
            for i, _ in enumerate(binned_ms1):
                selected_xgrams[i].append(0.0)
            selected_feature_dicts.append({})

    def get_feature(mz, feature_dict):
        return feature_dict.get(mz, np.zeros(8))

    ret = []
    for i, (mz, _) in enumerate(interp_ms1):
        xgram = np.array(selected_xgrams[i])

        feature_vectors = []
        for j in range(scan_size):
            feature_dict = selected_feature_dicts[j]
            feature_vec = get_feature(mz, feature_dict)
            feature_vectors.append(feature_vec)

        feature_vectors = np.array(feature_vectors)

        ret.append((mz, xgram, feature_vectors))

    ms1_mzs = np.array([x[0] for x in ret])
    ms1_xgrams = np.array([x[1] for x in ret])
    ms1_features = np.array([x[2] for x in ret])

    return ms1_mzs, ms1_xgrams, ms1_features

def interp_one_mz(all_ms1_rts, all_ms2_rts, train_ms1_intensities):
    train_ms1_rts = copy.copy(all_ms1_rts)
    train_ms1_intensities = copy.copy(train_ms1_intensities)

    ms1_rt_min = min(all_ms1_rts)
    ms1_rt_max = max(all_ms1_rts)
    ms2_rt_min = min(all_ms2_rts)
    ms2_rt_max = max(all_ms2_rts)

    if ms2_rt_min < ms1_rt_min:
        train_ms1_rts.insert(0, ms2_rt_min)
        train_ms1_intensities = np.insert(train_ms1_intensities, 0, 0.0)

    if ms2_rt_max > ms1_rt_max:
        train_ms1_rts.append(ms2_rt_max)
        train_ms1_intensities = np.insert(train_ms1_intensities, -1, 0.0)

    xp = train_ms1_rts
    fp = train_ms1_intensities
    x = all_ms2_rts

    y = np.interp(x, xp, fp)
    return np.array(y)

def gen_feature_for_binned_ms1(interp_ms1: list, related_ms2: pd.DataFrame):
    mzs = [x[0] for x in interp_ms1]
    xgrams = [x[1] for x in interp_ms1]

    scan_count = len(related_ms2)

    feature_dicts = []
    for i in range(scan_count):
        intensities = np.array([x[i] for x in xgrams])

        positive_indices = intensities > 0
        intensities = intensities[positive_indices]
        positive_mzs = np.array(mzs)[positive_indices]

        feature_dict = {}
        if len(intensities) > 0:
            features = feature_gen(positive_mzs, intensities)

            for i, mz in enumerate(positive_mzs):
                feature_dict[mz] = features[i]
        else:
            # this is normal, feature_dict will be empty
            pass

        feature_dicts.append(feature_dict)
    return feature_dicts
