import numpy as np
cimport numpy as np
cimport cython

cdef int get_bin_by_mz(double mz, double bin_width):
    return int(mz / bin_width)

def get_mz_by_bin(int bin_id, double bin_width):
    return (bin_id + 0.5) * bin_width

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_score_count_cython(np.ndarray[double, ndim=1] evidence,
                     double max_score,
                     double min_score,
                     np.ndarray[double, ndim=1] aa_freq,
                     np.ndarray[double, ndim=1] aa_masses,
                     double bin_width):
    cdef int mass_bins = evidence.shape[0]
    cdef int score_bins = int(max_score - min_score + 1)

    cdef np.ndarray[double, ndim=2] dp = np.zeros((score_bins, mass_bins), dtype=np.float64)
    cdef int zero_score = int(-min_score)
    dp[zero_score, 0] = 1.0

    cdef int min_score_int = int(np.floor(min_score))
    cdef int max_score_int = int(np.floor(max_score))

    cdef int mass, score, aa_idx
    cdef double aa_mass, evidence_mass
    cdef int score_idx, mass_idx

    for mass in range(1, mass_bins):
        evidence_mass = evidence[mass]
        for score in range(min_score_int, max_score_int):
            score_idx = score - min_score_int - int(evidence_mass)
            if 0 <= score_idx < score_bins:
                for aa_idx in range(aa_masses.shape[0]):
                    aa_mass = aa_masses[aa_idx]
                    mass_idx = get_bin_by_mz(get_mz_by_bin(mass, bin_width) - aa_mass, bin_width)
                    if 0 <= mass_idx < mass_bins:
                        if dp[score_idx, mass_idx] > 0.0:
                            dp[score - min_score_int, mass] += dp[score_idx, mass_idx] * aa_freq[aa_idx]

    cdef double score_offset = -min_score

    return score_offset, dp
