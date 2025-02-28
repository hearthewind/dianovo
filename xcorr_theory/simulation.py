import random

import numpy as np

from configs import main_iontypes, sub_iontypes, mz_min, mz_max, amino_acids, bin_width, \
    offset_window, sub_weight
from data.BasicClass import Residual_seq
from data.theo_peak_information import get_theoretical_peaks
from binning import get_bin_by_mz, get_mz_by_bin
from xcorr import process_spectra


def generate_random_peptide(l: int):
    peptide_seq = random.choices(amino_acids['ids'], k=l, weights=amino_acids['freq'])
    precursor_charge = random.randint(1, 5) #TODO(m) for debugging only
    return peptide_seq, precursor_charge

def get_signal_peaks_per_peptide(peptide_seq, precursor_charge):
    if precursor_charge <= 2:
        main_types = [x for x in main_iontypes if x.startswith('1')]
        sub_types = [x for x in sub_iontypes if x.startswith('1')]
    else:
        main_types = main_iontypes
        sub_types = sub_iontypes

    ret = []
    peptide_len = len(peptide_seq)
    for location in range(0, peptide_len - 1):
        mod_seq = ''.join(peptide_seq).replace('L', 'I')
        precursor_mass = Residual_seq(mod_seq).mass
        theo_peaks = get_theoretical_peaks(mod_seq, precursor_mass, location)
        for key, mz in theo_peaks.items():
            if key in main_types or key in sub_types:
                if mz_min < mz < mz_max - mz_min:
                    ret.append((mz, 1))

    return list(set(ret))

def generate_random_spectrum(peptide_seq, precursor_charge, signal, noise, noise_int):
    signal_peaks = get_signal_peaks_per_peptide(peptide_seq, precursor_charge)
    # signal_portion = random.sample(signal_peaks, k=signal) #TODO
    signal_portion = random.choices(signal_peaks, k=signal)
    noise_portion = [(random.uniform(mz_min, mz_max - mz_min), noise_int) for _ in range(noise)]

    spectrum = signal_portion + noise_portion

    return spectrum

def bin_spectrum(spectrum):
    num_bins = int(mz_max // bin_width)
    binned_spectrum = np.zeros(num_bins)
    for mz, intensity in spectrum:
        bin_id = get_bin_by_mz(mz)
        binned_spectrum[bin_id] += intensity
    return binned_spectrum

def xcorr_penalty(binned_spectrum):
    ret = np.zeros_like(binned_spectrum)

    for i in range(offset_window, len(binned_spectrum) - offset_window):
        offset_sum = 0.0
        for j in range(-offset_window, offset_window + 1):
            offset_sum += binned_spectrum[i + j]
        ret[i] = binned_spectrum[i] - 1.0 / (2.0 * offset_window) * offset_sum

    return ret

def generate_O(peptide_seq, precursor_charge, signal, noise, noise_int):
    exp_spectrum = generate_random_spectrum(peptide_seq, precursor_charge, signal, noise, noise_int)
    binned_spectrum = bin_spectrum(exp_spectrum)
    O = xcorr_penalty(binned_spectrum)
    return O

def generate_theoretical_spectrum(peptide_seq, precursor_charge):
    num_bins = int(mz_max // bin_width)
    binned_theo = np.zeros(num_bins)

    if precursor_charge <= 2:
        main_types = [x for x in main_iontypes if x.startswith('1')]
        sub_types = [x for x in sub_iontypes if x.startswith('1')]
    else:
        main_types = main_iontypes
        sub_types = sub_iontypes

    theo_intensity = 1.0
    peptide_len = len(peptide_seq)
    for location in range(0, peptide_len - 1):
        mod_seq = ''.join(peptide_seq).replace('L', 'I')
        precursor_mass = Residual_seq(mod_seq).mass
        theo_peaks = get_theoretical_peaks(mod_seq, precursor_mass, location)
        for key, mz in theo_peaks.items():
            # if mz_min < mz < mz_max - mz_min: #TODO
            bin_id = get_bin_by_mz(mz)
            if key in main_types:
                binned_theo[bin_id] += theo_intensity
            elif key in sub_types:
                binned_theo[bin_id] += theo_intensity * sub_weight
    return binned_theo

def compute_xcorr_ori(peptide_seq, precursor_charge, O):
    theo_spectrum = generate_theoretical_spectrum(peptide_seq, precursor_charge)
    return np.sum(theo_spectrum * O)

def run_one_peptide(l: int, signal: int, noise: int, noise_int: float):
    try:
        peptide_seq, precursor_charge = generate_random_peptide(l)
        O = generate_O(peptide_seq, precursor_charge, signal, noise, noise_int)
        xcorr_score, peptides_with_higher_score, total_number_of_peptides = process_spectra(O, peptide_seq, precursor_charge)
    except Exception as e:
        return None

    return {'peptide_seq': peptide_seq,
            'precursor_charge': precursor_charge,
            'xcorr_score': xcorr_score,
            'peptides_with_higher_score': peptides_with_higher_score,
            'total_number_of_peptides': total_number_of_peptides}