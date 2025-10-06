import numpy as np
import pandas as pd

from utils.data.peptide import get_ms2_from_precursor
from utils.cofragment_peptide_processing_cy import label_ms2_one_peptide, label_dict
from utils.cofragment_constants import mass_threshold


def is_overlap(start1, end1, start2, end2):
    # Check if one interval starts after the other ends
    if start1 > end2 or start2 > end1:
        return False
    else:
        return True


def find_all_cofragment(peptide: dict, diann_result: list, ms2_df: pd.DataFrame):
    related_ms2 = get_ms2_from_precursor(peptide, ms2_df)
    first_ms2 = related_ms2.iloc[0]
    precursor_mz_lower = first_ms2['precursor_mz_lower']
    precursor_mz_upper = first_ms2['precursor_mz_upper']

    rt_start = peptide['rt_start']
    rt_stop = peptide['rt_stop']
    precursor_mz = peptide['precursor_mz']
    run = peptide['run']

    ret = []
    for add_peptide in diann_result:
        add_rt_start = add_peptide['rt_start']
        add_rt_stop = add_peptide['rt_stop']
        add_precursor_mz = add_peptide['precursor_mz']
        add_run = add_peptide['run']

        if run == add_run:
            if is_overlap(add_rt_start, add_rt_stop, rt_start, rt_stop):
                if precursor_mz_lower < add_precursor_mz < precursor_mz_upper:
                    if add_peptide != peptide:
                        ret.append(add_peptide)

    return ret

# def label_ms2_one_peptide(peptide: dict, ms2_mzs: np.array):
#     mod_seq = peptide['mod_sequence']
#     precursor_charge = peptide['charge']
#
#     if precursor_charge <= 2:
#         chosen_ion_types = [x for x in ion_types if x.startswith('1')]
#     else:
#         chosen_ion_types = ion_types
#
#     # precursor_mass should be theoretical
#     precursor_mass_theo = Residual_seq(mod_seq).mass
#
#     theo_fragments = []
#     for location in range(0, len(mod_seq) - 1):
#         theoretical_peaks = get_theoretical_peaks(mod_seq, precursor_mass_theo, location)
#         for ion in chosen_ion_types:
#             mz = theoretical_peaks[ion]
#             if ion == '1a' and mz > 300.0:
#                 continue
#             elif ion == '2y' and mz < 400.0:
#                 continue
#
#             theo_fragments.append((mz, ion))
#
#     labels = []
#     for mz in ms2_mzs:
#         fragment = False
#         for theo_mz, ion in theo_fragments:
#             if mz - mass_threshold <= theo_mz <= mz + mass_threshold:
#                 labels.append(ion)
#                 fragment = True
#                 break
#         if not fragment:
#             labels.append('noise')
#
#     ret = [label_dict[label] for label in labels]
#     ret = np.array(ret)
#     return ret

def label_ms1_one_peptide(peptide: dict, ms1_mzs: np.array):
    precursor_mz = peptide['precursor_mz']
    precursor_charge = peptide['charge']

    start_idx = np.searchsorted(ms1_mzs, precursor_mz - mass_threshold)
    end_idx = np.searchsorted(ms1_mzs, precursor_mz + mass_threshold)

    ret = np.zeros_like(ms1_mzs)
    ret[start_idx:end_idx] = label_dict['ms1']

    return ret

def process_one_peptide_and_cofragment(peptide: dict, diann_result: list, ms2_df: pd.DataFrame, ms2_mzs: np.array, ms1_mzs: np.array):
    all_cofragments = find_all_cofragment(peptide, diann_result, ms2_df)

    cofragment_ms2_labels = []
    cofragment_ms1_labels = []
    for cofragment in all_cofragments:
        ms2_label = label_ms2_one_peptide(cofragment, ms2_mzs)
        ms1_label = label_ms1_one_peptide(cofragment, ms1_mzs)
        if (ms2_label > 0).any():
            cofragment_ms2_labels.append(ms2_label)
            cofragment_ms1_labels.append(ms1_label)

    target_ms2_label = label_ms2_one_peptide(peptide, ms2_mzs)
    target_ms1_label = label_ms1_one_peptide(peptide, ms1_mzs)
    return cofragment_ms2_labels, cofragment_ms1_labels, target_ms2_label, target_ms1_label
