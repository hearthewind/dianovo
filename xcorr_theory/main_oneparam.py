import pickle
import sys

import configs
from simulation_oneparam import run_one_param

if __name__ == '__main__':
    peplen, noise_int, signal, noise = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    configs.noise_intensity = noise_int
    p_value = run_one_param(peplen, signal, noise)

    print(f'peplen: {peplen}, signal: {signal}, noise: {noise}, p_value: {p_value}')