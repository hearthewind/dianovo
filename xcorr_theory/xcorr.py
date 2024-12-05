import numpy as np

from configs import main_iontypes, sub_iontypes, sub_weight, bin_width, amino_acids
from data.BasicClass import Ion, Residual_seq
from binning import get_mz_by_bin, get_bin_by_mz
from cython_xcorr import calc_score_count_cython


# convert a spectrum to n term
def convert2nterm(precursor_mass, precursor_charge, O):
    if precursor_charge <= 2:
        main_types = [x for x in main_iontypes if x.startswith('1')]
        sub_types = [x for x in sub_iontypes if x.startswith('1')]
    else:
        main_types = main_iontypes
        sub_types = sub_iontypes

    nterm_spectrum = []
    for bin_id, intensity in enumerate(O):
        mz = get_mz_by_bin(bin_id)
        for ion_type in main_types:
            seq_mass = Ion.peak2sequencemz(mz, ion_type)
            if 'y' in ion_type:
                nterm_mass = precursor_mass - seq_mass
            else:
                nterm_mass = seq_mass

            nterm_spectrum.append((nterm_mass, intensity))
        for ion_type in sub_types:
            seq_mass = Ion.peak2sequencemz(mz, ion_type)
            if 'y' in ion_type:
                nterm_mass = precursor_mass - seq_mass
            else:
                nterm_mass = seq_mass
            nterm_spectrum.append((nterm_mass, intensity * sub_weight))

    evidence = np.zeros(int(np.ceil(precursor_mass / bin_width) + 1))
    for nterm_mass, intensity in nterm_spectrum:
        bin_idx = get_bin_by_mz(nterm_mass)
        if 0 <= nterm_mass <= precursor_mass:
            evidence[bin_idx] += intensity

    return np.round(20 * evidence)


# Function to compute XCorr score for a given peptide sequence
def compute_xcorr(peptide_sequence, evidence):

    # Create the theoretical spectrum (b-ion series) for the peptide sequence
    theoretical_spectrum = np.zeros(len(evidence))
    for i in range(len(peptide_sequence) - 1):
        partial_seq = ''.join(peptide_sequence[:(i+1)]).replace('L', 'I')
        nterm_mass = Residual_seq(partial_seq).mass
        bin_idx = get_bin_by_mz(nterm_mass)
        if bin_idx < len(theoretical_spectrum):
            theoretical_spectrum[bin_idx] += 1  # Increment intensity for the theoretical spectrum

    # Calculate the XCorr score as the dot product between the evidence and theoretical spectrum
    xcorr_score = np.sum(evidence * theoretical_spectrum)

    return xcorr_score


# Main processing function
def process_spectra(O, peptide_sequence, precursor_charge):
    min_delta_mass = min(amino_acids["masses"])

    # Generate the evidence vector from the experimental spectrum
    precursor_mass = Residual_seq(''.join(peptide_sequence).replace('L', 'I')).mass
    evidence = convert2nterm(precursor_mass, precursor_charge, O)

    # Calculate the max and min possible scores based on evidence
    max_n_residue = int(np.floor(precursor_mass / min_delta_mass))
    sorted_evidence = sorted(evidence)[::-1]

    max_score = np.sum(sorted_evidence[:max_n_residue])
    min_score = np.sum(sorted_evidence[-max_n_residue:])

    # Calculate the dynamic programming matrix to get score counts
    # score_offset, score_counts = calc_score_count(evidence, max_score, min_score)
    score_offset, score_counts = calc_score_count_cython(evidence, max_score, min_score, \
                                                         np.array(amino_acids['freq']), np.array(amino_acids['masses']), \
                                                         bin_width)

    # return score_offset, score_counts

    # Calculate XCorr score for the given peptide sequence
    xcorr_score = compute_xcorr(peptide_sequence, evidence)

    # Calculate the index for the XCorr score in the score_counts array
    score_count_idx = int(xcorr_score + score_offset)
    precorsor_mass_idx = get_bin_by_mz(precursor_mass)

    # (1) Number of peptides with a score higher than the target peptide
    peptides_with_higher_score = np.sum(score_counts[score_count_idx+1:, precorsor_mass_idx])

    # (2) Total number of peptides with the same precursor mass
    total_number_of_peptides = np.sum(score_counts[:, precorsor_mass_idx])

    return xcorr_score, peptides_with_higher_score, total_number_of_peptides
