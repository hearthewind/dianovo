from enum import Enum

class ToleranceType(Enum):
    PPM = 0
    Da = 1


mz_max = 3500.0

ms1_tolerance = 20
ms1_tol_type = ToleranceType.PPM

ms2_tolerance = 20
ms2_tol_type = ToleranceType.PPM

bin_size = 10 * 1000 / 1e6  # 10 ppm at 1000mz   # 0.01 Da

num_bins = int(mz_max / bin_size)

print('bin_size ', bin_size)
print('mz_max ', mz_max)
print('num_bins', num_bins)
