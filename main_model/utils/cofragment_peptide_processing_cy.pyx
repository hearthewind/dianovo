import numpy as np
cimport numpy as cnp
from Cython import boundscheck, wraparound
from libc.math cimport fabs
from cython.parallel import prange
cimport cython

from utils.data.BasicClass import Residual_seq
from utils.data.theo_peak_information import get_theoretical_peaks

ion_types = ['1a','1b','2a','2b','1a-NH3','1a-H2O','1b-NH3','1b-H2O'] + ['1y','1y-NH3','1y-H2O','2y'] # total 12 ions

label_types = ['noise'] + ion_types + ['ms1']
label_dict = dict(zip(label_types, np.arange(len(label_types))))

@cython.profile(True)
@boundscheck(False)
@wraparound(False)
def label_ms2_one_peptide(dict peptide, cnp.ndarray[double, ndim=1] ms2_mzs):
    cdef:
        str mod_seq = peptide['mod_sequence']
        int precursor_charge = peptide['charge']
        double precursor_mass_theo = Residual_seq(mod_seq).mass
        list theo_fragments = []
        cnp.ndarray[double, ndim=1] theo_mz_values
        cnp.ndarray[int, ndim=1] ion_indices
        double mz
        str ion
        int location, i, num_fragments, fragment_idx, num_mzs
        double mass_threshold = 0.26  # Threshold for matching
        list chosen_ion_types
        cnp.ndarray[int, ndim=1] ret
        bint fragment
        int noise_index = label_dict['noise']  # Precompute index for 'noise'

    # Store length before nogil block (avoiding calling len() inside prange)
    num_mzs = ms2_mzs.shape[0]

    # Select ion types based on charge
    if precursor_charge <= 2:
        chosen_ion_types = [x for x in ion_types if x.startswith('1')]
    else:
        chosen_ion_types = ion_types

    # Compute theoretical fragment ions
    for location in range(len(mod_seq) - 1):
        theoretical_peaks = get_theoretical_peaks(mod_seq, precursor_mass_theo, location)
        for ion in chosen_ion_types:
            mz = theoretical_peaks[ion]
            if (ion == '1a' and mz > 300.0) or (ion == '2y' and mz < 400.0):
                continue
            theo_fragments.append((mz, label_dict[ion]))

    # Prepare C arrays for efficient parallel processing
    num_fragments = len(theo_fragments)
    theo_mz_values = np.zeros(num_fragments, dtype=np.float64)
    ion_indices = np.zeros(num_fragments, dtype=np.int32)

    for i, (mz, ion_idx) in enumerate(theo_fragments):
        theo_mz_values[i] = mz
        ion_indices[i] = ion_idx

    # Allocate memory for the result array
    ret = np.empty(num_mzs, dtype=np.int32)

    # Assign labels efficiently using parallelization
    for i in prange(num_mzs, nogil=True):  # Using precomputed num_mzs
        fragment = False
        for fragment_idx in range(num_fragments):
            if fabs(ms2_mzs[i] - theo_mz_values[fragment_idx]) <= mass_threshold:
                ret[i] = ion_indices[fragment_idx]
                fragment = True
                break
        if not fragment:
            ret[i] = noise_index  # Use precomputed index for 'noise'

    return ret

# from scipy.spatial import cKDTree
#
# @cython.profile(True)
# @boundscheck(False)
# @wraparound(False)
# def label_ms2_one_peptide(dict peptide, cnp.ndarray[double, ndim=1] ms2_mzs):
#     cdef:
#         str mod_seq = peptide['mod_sequence']
#         int precursor_charge = peptide['charge']
#         double precursor_mass_theo = Residual_seq(mod_seq).mass
#         cnp.ndarray[double, ndim=1] theo_mz_values
#         cnp.ndarray[int, ndim=1] ion_indices
#         double mz
#         str ion
#         int location, i, num_fragments, num_mzs
#         double mass_threshold = 0.02  # Threshold for matching
#         int noise_index = label_dict['noise']  # Precompute index for 'noise'
#         list chosen_ion_types
#         cnp.ndarray[int, ndim=1] ret
#
#     # Determine length of ms2_mzs
#     num_mzs = ms2_mzs.shape[0]
#
#     # Select ion types
#     if precursor_charge <= 2:
#         chosen_ion_types = [x for x in ion_types if x.startswith('1')]
#     else:
#         chosen_ion_types = ion_types
#
#     # Preallocate arrays for fragments
#     cdef int max_possible_fragments = (len(mod_seq) - 1) * len(chosen_ion_types)
#     theo_mz_values = np.zeros(max_possible_fragments, dtype=np.float64)
#     ion_indices = np.zeros(max_possible_fragments, dtype=np.int32)
#     num_fragments = 0
#
#     # Compute theoretical fragments
#     for location in range(len(mod_seq) - 1):
#         theoretical_peaks = get_theoretical_peaks(mod_seq, precursor_mass_theo, location)
#         for ion in chosen_ion_types:
#             mz = theoretical_peaks[ion]
#             if (ion == '1a' and mz > 300.0) or (ion == '2y' and mz < 400.0):
#                 continue
#             theo_mz_values[num_fragments] = mz
#             ion_indices[num_fragments] = label_dict[ion]
#             num_fragments += 1
#
#     theo_mz_values = theo_mz_values[:num_fragments]
#     ion_indices = ion_indices[:num_fragments]
#
#     # Allocate result array
#     ret = np.empty(num_mzs, dtype=np.int32)
#
#     # Use KD-tree for fast matching
#     kdtree = cKDTree(theo_mz_values[:, None])
#     distances, indices = kdtree.query(ms2_mzs[:, None], distance_upper_bound=mass_threshold)
#
#     # Assign labels
#     ret.fill(noise_index)
#     valid = distances <= mass_threshold
#     ret[valid] = ion_indices[indices[valid]]
#
#     return ret
