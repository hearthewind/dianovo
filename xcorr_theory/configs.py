bin_width = 0.05
offset_window = 75

# Amino Acid Information

# amino_acids = {
#     'ids': ['A', 'R', 'N', 'D', 'c', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'm', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'], # Amino acid IDs
#     "masses": [71.03711, 156.10111, 114.04293, 115.02694, 160.03065, 129.04259, 128.05858, 57.02146, 137.05891, 113.08406, 113.08406,
#                128.09496, 131.04049, 147.0354, 147.06841, 97.05276, 87.03203, 101.04768, 186.07931, 163.06333, 99.06841],  # masses for amino acids
#     "freq": [0.074, 0.042, 0.044, 0.059, 0.033, 0.058, 0.037, 0.074, 0.029, 0.038, 0.076, 0.072, 0.009, 0.009, 0.040, 0.050,
#              0.081, 0.062, 0.013, 0.033, 0.067],  # Frequencies
# }

amino_acids = {
    'ids': ['A', 'R', 'N', 'D', 'c', 'E', 'Q', 'G', 'H', 'I', 'K', 'M', 'm', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'], # Amino acid IDs
    "masses": [71.03711, 156.10111, 114.04293, 115.02694, 160.03065, 129.04259, 128.05858, 57.02146, 137.05891, 113.08406,
               128.09496, 131.04049, 147.0354, 147.06841, 97.05276, 87.03203, 101.04768, 186.07931, 163.06333, 99.06841],  # masses for amino acids
    "freq": [0.074, 0.042, 0.044, 0.059, 0.033, 0.058, 0.037, 0.074, 0.029, 0.114, 0.072, 0.009, 0.009, 0.040, 0.050,
             0.081, 0.062, 0.013, 0.033, 0.067],  # Frequencies
    # "freq": [1.0] * 20
}

main_iontypes = set(['1a','1b','2a','2b'] + ['1y','2y'])
sub_iontypes = set(['1a-NH3','1a-H2O','1b-NH3','1b-H2O'] + ['1y-NH3','1y-H2O'])
sub_weight = 1.0

mz_min = 100.0
mz_max = 2000.0
noise_intensity = 0.5
