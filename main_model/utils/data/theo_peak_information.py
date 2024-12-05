
mass_AA = {'A': 71.03711, # 0
           'R': 156.10111, # 1
           'N': 114.04293, # 2
           'D': 115.02694, # 3
           # 'C': 103.00919, # 4
           'c': 160.03065, # C(+57.02)
           # 'c': 161.01919, # C(+58.01) # orbi
           'E': 129.04259, # 5
           'Q': 128.05858, # 6
           'G': 57.02146, # 7
           'H': 137.05891, # 8
           'I': 113.08406, # 9
           'L': 113.08406, # 10
           'K': 128.09496, # 11
           'M': 131.04049, # 12
           'm': 147.0354, # M(+15.99)
           'F': 147.06841, # 13
           'P': 97.05276, # 14
           'S': 87.03203, # 15
           'T': 101.04768, # 16
           'W': 186.07931, # 17
           'Y': 163.06333, # 18
           'V': 99.06841, # 19
          }

mass_proton = 1.0073
mass_H = 1.0078
mass_N = 14.0031
mass_O = 15.9949
mass_C = 12


def get_theoretical_peaks(mod_seq: str, precursor_mass: float, location: int):
    assert 0 <= location < len(mod_seq) - 1

    prefix_mass = sum([mass_AA[aa] for aa in mod_seq[:location + 1]])
    suffix_mass = precursor_mass - prefix_mass + mass_H + mass_O
    # suffix_mass = sum([mass_AA[aa] for aa in mod_seq[location + 1:]]) + mass_O + mass_H

    neutral_mass_a = prefix_mass - mass_C - mass_H - mass_O + mass_H
    neutral_mass_a_nh3 = neutral_mass_a - mass_N - 3 * mass_H
    neutral_mass_a_h2o = neutral_mass_a - 2 * mass_H - mass_O
    neutral_mass_b = prefix_mass - mass_H + mass_H
    neutral_mass_b_nh3 = neutral_mass_b - mass_N - 3 * mass_H
    neutral_mass_b_h2o = neutral_mass_b - 2 * mass_H - mass_O
    neutral_mass_c = prefix_mass + mass_N + 2 * mass_H  + mass_H
    neutral_mass_x = suffix_mass + mass_C + mass_O - mass_H
    neutral_mass_y = suffix_mass + mass_H
    neutral_mass_y_nh3 = neutral_mass_y - mass_N - 3 * mass_H
    neutral_mass_y_h2o = neutral_mass_y - 2 * mass_H - mass_O
    neutral_mass_z = suffix_mass - mass_N - 2 * mass_H

    def get_mz_from_neutral_mass(neutral_mass: float, charge: int):
        mass = neutral_mass + charge * mass_proton
        return mass / charge

    theoretical_mzs = {'1a': get_mz_from_neutral_mass(neutral_mass_a, 1),
                       '2a': get_mz_from_neutral_mass(neutral_mass_a, 2),
                       '1a-NH3': get_mz_from_neutral_mass(neutral_mass_a_nh3, 1),
                       '2a-NH3': get_mz_from_neutral_mass(neutral_mass_a_nh3, 2),
                       '1a-H2O': get_mz_from_neutral_mass(neutral_mass_a_h2o, 1),
                       '2a-H2O': get_mz_from_neutral_mass(neutral_mass_a_h2o, 2),
                       '1b': get_mz_from_neutral_mass(neutral_mass_b, 1),
                       '2b': get_mz_from_neutral_mass(neutral_mass_b, 2),
                       '1b-NH3': get_mz_from_neutral_mass(neutral_mass_b_nh3, 1),
                       '2b-NH3': get_mz_from_neutral_mass(neutral_mass_b_nh3, 2),
                       '1b-H2O': get_mz_from_neutral_mass(neutral_mass_b_h2o, 1),
                       '2b-H2O': get_mz_from_neutral_mass(neutral_mass_b_h2o, 2),
                       '1c': get_mz_from_neutral_mass(neutral_mass_c, 1),
                       '2c': get_mz_from_neutral_mass(neutral_mass_c, 2),
                       '1x': get_mz_from_neutral_mass(neutral_mass_x, 1),
                       '2x': get_mz_from_neutral_mass(neutral_mass_x, 2),
                       '1y': get_mz_from_neutral_mass(neutral_mass_y, 1),
                       '2y': get_mz_from_neutral_mass(neutral_mass_y, 2),
                       '1y-NH3': get_mz_from_neutral_mass(neutral_mass_y_nh3, 1),
                       '2y-NH3': get_mz_from_neutral_mass(neutral_mass_y_nh3, 2),
                       '1y-H2O': get_mz_from_neutral_mass(neutral_mass_y_h2o, 1),
                       '2y-H2O': get_mz_from_neutral_mass(neutral_mass_y_h2o, 2),
                       '1z': get_mz_from_neutral_mass(neutral_mass_z, 1),
                       '2z': get_mz_from_neutral_mass(neutral_mass_z, 2)}

    return theoretical_mzs
