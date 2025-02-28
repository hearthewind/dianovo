# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, profile=True, language_level=3
"""
Cythonized version of the peak feature generator.

Assumes that the input m/z array is 1D and sorted in increasing order.
"""

import numpy as np
cimport numpy as np
from libc.math cimport exp, tanh

cdef class PeakFeatureGeneration:
    cdef int local_sliding_window
    cdef double data_acquisition_upper_limit

    def __init__(self, int local_sliding_window=50, double data_acquisition_upper_limit=1000.0):
        """
        Parameters:
          local_sliding_window: the half-window size in m/z units for local feature computation.
          data_acquisition_upper_limit: the upper limit used to normalize m/z.
        """
        self.local_sliding_window = local_sliding_window
        self.data_acquisition_upper_limit = data_acquisition_upper_limit

    def __call__(self, np.ndarray[double, ndim=1] product_ions_moverz,
                       np.ndarray[double, ndim=1] product_ions_intensity):
        """
        Compute the 8 features for each peak.
        Returns an array of shape (n_peaks, 8) with features:
          0. normalized m/z = exp(-m/z / data_acquisition_upper_limit)
          1. global relative intensity = intensity / max(intensity)
          2. local significant = tanh(((intensity/local_min) - 1)/2)
          3. total rank = fraction of peaks with intensity lower than the current one.
          4. total half-rank = fraction of peaks with intensity lower than half the current one.
          5. local rank = fraction (within the local window) of peaks with intensity lower than current.
          6. local half-rank = fraction (within the local window) of peaks with intensity lower than half current.
          7. local relative intensity = intensity / max(intensity within the local window).
        """
        cdef int n, i, j, window_count, count_local_rank, count_local_halfrank, left_index, right_index
        cdef double curr_mz, curr_intensity, local_min, local_max, val
        cdef double global_max_intensity

        # Filter out non-positive m/z values. (This is done in Python.)
        np_arr = product_ions_moverz > 0
        cdef np.ndarray[double, ndim=1] mz = product_ions_moverz[np_arr]
        cdef np.ndarray[double, ndim=1] intensity = product_ions_intensity[np_arr]
        n = mz.shape[0]
        if n == 0:
            return np.empty((0, 8), dtype=np.double)

        # Global features: maximum intensity and sorted intensities for total rank features.
        global_max_intensity = intensity.max()
        cdef np.ndarray[double, ndim=1] sorted_intensity = np.sort(intensity)

        # Preallocate the output feature array (n_peaks x 8)
        cdef np.ndarray[double, ndim=2] features = np.empty((n, 8), dtype=np.double)

        # Precompute total_rank and total_halfrank for each peak.
        cdef np.ndarray[double, ndim=1] total_rank = np.empty(n, dtype=np.double)
        cdef np.ndarray[double, ndim=1] total_halfrank = np.empty(n, dtype=np.double)
        for i in range(n):
            total_rank[i] = np.searchsorted(sorted_intensity, intensity[i], side='left') / float(n)
            total_halfrank[i] = np.searchsorted(sorted_intensity, intensity[i] / 2.0, side='left') / float(n)

        # Compute features for each peak.
        for i in range(n):
            curr_mz = mz[i]
            curr_intensity = intensity[i]
            # Feature 0: normalized m/z.
            features[i, 0] = exp(-curr_mz / self.data_acquisition_upper_limit)
            # Feature 1: global relative intensity.
            features[i, 1] = curr_intensity / global_max_intensity

            # Determine the local window indices using np.searchsorted (assumes mz is sorted).
            left_index = np.searchsorted(mz, curr_mz - self.local_sliding_window, side='left')
            right_index = np.searchsorted(mz, curr_mz + self.local_sliding_window, side='right')
            window_count = right_index - left_index

            if window_count <= 0:
                window_count = 1
                local_min = curr_intensity
                local_max = curr_intensity
                count_local_rank = 0
                count_local_halfrank = 0
            else:
                # Initialize local minimum and maximum using the first value in the window.
                local_min = intensity[left_index]
                local_max = intensity[left_index]
                count_local_rank = 0
                count_local_halfrank = 0
                for j in range(left_index, right_index):
                    val = intensity[j]
                    if val < local_min:
                        local_min = val
                    if val > local_max:
                        local_max = val
                    if curr_intensity > val:
                        count_local_rank += 1
                    if (curr_intensity / 2.0) > val:
                        count_local_halfrank += 1

            # Feature 2: local significant = tanh(((curr_intensity/local_min)-1)/2)
            features[i, 2] = tanh(((curr_intensity / local_min) - 1.0) / 2.0)
            # Feature 3: total rank.
            features[i, 3] = total_rank[i]
            # Feature 4: total half-rank.
            features[i, 4] = total_halfrank[i]
            # Feature 5: local rank.
            features[i, 5] = count_local_rank / float(window_count)
            # Feature 6: local half-rank.
            features[i, 6] = count_local_halfrank / float(window_count)
            # Feature 7: local relative intensity.
            features[i, 7] = curr_intensity / local_max

        return features
