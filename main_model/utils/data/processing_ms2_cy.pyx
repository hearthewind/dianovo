import numpy as np
cimport numpy as cnp  # For typed arrays / memoryviews
from libc.math cimport fabs
from cython import boundscheck, wraparound
import cython

# Pandas is pure Python (cannot 'cimport')
import pandas as pd

# Project imports
from utils.data.configs import num_bins, mz_max
from utils.data.peptide import get_ms2_from_precursor
from utils.data.processing_ms2 import gen_feature_for_one_binned_spectrum
from utils.data.processing_utils import find_best_scan_index, bin_one_spectrum


@cython.profile(True)
@boundscheck(False)
@wraparound(False)
def bin_one_peptide(dict peptide, object ms2_df, int scan_size=5, bint neighboring=True):
    """
    A Cython-optimized version of bin_one_peptide.

    - Converts Python lists to float32 NumPy arrays.
    - Uses typed arrays (cimport numpy as cnp).
    - Eliminates in-loop cdef statements.
    """

    ############################################################################
    # 1) Typed variable declarations at the top
    ############################################################################
    cdef double rt_peak = peptide['rt']
    cdef int scan_count, best_scan_index, scan_center
    cdef int scan_left_count, scan_right_count, scan_size_half
    cdef int total_scans, pad_left, pad_right
    cdef int leftover, needed_left, needed_right
    cdef int row_idx, i, j, row, out_idx
    cdef double bin_mz
    cdef int n_valid
    cdef bint do_neighboring = neighboring

    # Temporary Python objects
    cdef object related_ms2 = None
    cdef object ms2 = None
    cdef object raw_b_spectrum = None
    cdef object raw_feat_obj = None

    # Typed arrays for intermediate data
    cdef cnp.ndarray[cnp.float32_t, ndim=1] b_spectrum = None
    cdef cnp.ndarray[cnp.float32_t, ndim=1] feat_arr = None

    # Dict for each scan's features
    cdef dict fdict = {}

    # We'll store a feature dict for each scan row
    cdef list peak_feature_dicts = []

    # 2D arrays for bin intensities
    cdef cnp.ndarray[cnp.float32_t, ndim=2] binned_spectra_arr
    cdef cnp.ndarray[cnp.float32_t, ndim=2] intensities_arr

    # Memoryviews for numeric loops
    cdef cnp.float32_t[:, :] binned_view
    cdef cnp.float32_t[:, :] intens_view

    # Track which bins are valid
    cdef cnp.ndarray[cnp.int32_t, ndim=1] valid_bin_arr
    cdef cnp.int32_t[:] valid_bin  # memoryview of valid_bin_arr

    # Final outputs
    cdef cnp.ndarray[cnp.float64_t, ndim=1] ms2_mzs
    cdef cnp.ndarray[cnp.float32_t, ndim=2] ms2_xgrams
    cdef cnp.ndarray[cnp.float32_t, ndim=3] ms2_features

    ############################################################################
    # 2) Retrieve MS2 scans and find best scan index
    ############################################################################
    related_ms2 = get_ms2_from_precursor(peptide, ms2_df)
    scan_count = len(related_ms2)
    best_scan_index = find_best_scan_index(related_ms2, rt_peak)

    # Determine left/right
    scan_center = best_scan_index
    scan_left_count = scan_center
    scan_right_count = scan_count - scan_center - 1
    scan_size_half = scan_size // 2

    if scan_left_count > scan_size_half:
        scan_left_count = scan_size_half
    if scan_right_count > scan_size_half:
        scan_right_count = scan_size_half

    total_scans = scan_left_count + 1 + scan_right_count
    pad_left = 0
    pad_right = 0

    # Possibly pad to match exactly scan_size
    if total_scans < scan_size:
        leftover = scan_size - total_scans

        if scan_left_count < scan_size_half:
            needed_left = scan_size_half - scan_left_count
            pad_left = min(leftover, needed_left)
            leftover -= pad_left
        else:
            needed_left = 0

        if leftover > 0 and scan_right_count < scan_size_half:
            needed_right = scan_size_half - scan_right_count
            pad_right = min(leftover, needed_right)
            leftover -= pad_right
        else:
            needed_right = 0
    else:
        leftover = 0
        needed_left = 0
        needed_right = 0

    total_scans += pad_left + pad_right

    ############################################################################
    # 3) Allocate typed arrays for binned spectra
    ############################################################################
    binned_spectra_arr = np.zeros((total_scans, num_bins), dtype=np.float32)
    intensities_arr = np.zeros((total_scans, num_bins), dtype=np.float32)

    # Convert to memoryviews for fast numeric loops
    binned_view = binned_spectra_arr
    intens_view = intensities_arr

    row_idx = 0

    ############################################################################
    # 4) Pad left
    ############################################################################
    for _ in range(pad_left):
        peak_feature_dicts.append({})
        row_idx += 1

    ############################################################################
    # 5) Fill real scans
    ############################################################################
    cdef int start_idx = scan_center - scan_left_count
    cdef int end_idx = scan_center + scan_right_count

    for i in range(start_idx, end_idx + 1):
        ms2 = related_ms2.iloc[i]                        # Python DF access
        raw_b_spectrum = bin_one_spectrum(ms2)           # returns list or array
        # Convert to float32 array
        b_spectrum = np.ascontiguousarray(raw_b_spectrum, dtype=np.float32)

        # Assign row
        binned_spectra_arr[row_idx, :] = b_spectrum

        # Feature dict
        raw_feat_obj = gen_feature_for_one_binned_spectrum(raw_b_spectrum)
        peak_feature_dicts.append(raw_feat_obj)
        row_idx += 1

    ############################################################################
    # 6) Pad right
    ############################################################################
    for _ in range(pad_right):
        peak_feature_dicts.append({})
        row_idx += 1

    ############################################################################
    # 7) Neighbor summation (if neighboring=True)
    ############################################################################
    for i in range(num_bins):
        # copy base intensities
        for row in range(total_scans):
            intens_view[row, i] = binned_view[row, i]

        if do_neighboring:
            if i > 0:
                for row in range(total_scans):
                    intens_view[row, i] += binned_view[row, i - 1]
            if i < (num_bins - 1):
                for row in range(total_scans):
                    intens_view[row, i] += binned_view[row, i + 1]

    ############################################################################
    # 8) Identify valid bins ("any > 0")
    ############################################################################
    valid_bin_arr = np.zeros(num_bins, dtype=np.int32)
    valid_bin = valid_bin_arr  # memoryview

    for i in range(num_bins):
        valid_bin[i] = 0
        for row in range(total_scans):
            if intens_view[row, i] > 0:
                valid_bin[i] = 1
                break

    # Count how many bins are valid
    n_valid = 0
    for i in range(num_bins):
        if valid_bin[i] == 1:
            n_valid += 1

    ############################################################################
    # 9) Allocate final outputs
    ############################################################################
    ms2_mzs = np.empty(n_valid, dtype=np.float64)
    ms2_xgrams = np.zeros((n_valid, total_scans), dtype=np.float32)
    ms2_features = np.zeros((n_valid, total_scans, 8), dtype=np.float32)

    out_idx = 0

    ############################################################################
    # 10) Fill final outputs (dictionary lookups in Python)
    ############################################################################
    for i in range(num_bins):
        if valid_bin[i] == 0:
            continue

        bin_mz = (i + 0.5) * mz_max / num_bins
        ms2_mzs[out_idx] = bin_mz

        # Copy intensities
        for row in range(total_scans):
            ms2_xgrams[out_idx, row] = intens_view[row, i]

        # Build feature vectors
        for row in range(total_scans):
            fdict = peak_feature_dicts[row]
            # Possibly float64 or list, so cast to float32
            raw_feat_obj = fdict.get(bin_mz, np.zeros(8, dtype=np.float32))
            feat_arr = np.ascontiguousarray(raw_feat_obj, dtype=np.float32)

            # Copy 8 floats
            for j in range(8):
                ms2_features[out_idx, row, j] = feat_arr[j]

        out_idx += 1

    return ms2_mzs, ms2_xgrams, ms2_features
