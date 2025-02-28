import numpy as np
import pandas as pd

import sys
sys.path.append('../../')
from utils.data.configs import mz_max, ToleranceType, bin_size, num_bins
from utils.data.peak_feature_generator import PeakFeatureGeneration


def within_tolerance(theoretical_peak: float,
                     observed_peak: float,
                     tolerance: float=10,
                     t_type: ToleranceType=ToleranceType.PPM):
    if t_type == ToleranceType.PPM:
        maxdiff = tolerance * theoretical_peak / 1e6
    else:
        maxdiff = tolerance

    if abs(theoretical_peak - observed_peak) < maxdiff:
        return True
    else:
        return False


def bin_one_spectrum(spectrum: pd.Series):
    binned_spectrum = [0.0] * num_bins

    mzs = spectrum.mz
    intensities = spectrum.intensity

    for i, mz in enumerate(mzs):
        intensity = intensities[i]

        bin_id = int(mz / bin_size)
        try:
            binned_spectrum[bin_id] += intensity
        except IndexError:
            pass

    return binned_spectrum


feature_gen = PeakFeatureGeneration()


def find_best_scan_index(related_ms2: pd.DataFrame, rt_peak):
    best_scan_index = -1
    best_rt_dist = float('inf')
    for scan_index, (_, scan) in enumerate(related_ms2.iterrows()):
        distance = abs(scan.rt - rt_peak)
        if distance < best_rt_dist:
            best_rt_dist = distance
            best_scan_index = scan_index

    assert best_scan_index >= 0
    return int(best_scan_index)




