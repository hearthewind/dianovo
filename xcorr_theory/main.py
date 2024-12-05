import pickle
import sys

from simulation_oneparam import run_one_param

if __name__ == '__main__':
    signal, noise_start = int(sys.argv[1]), int(sys.argv[2])

    result = {}

    for noise in range(noise_start, 10100, 500):
    # for noise in range(noise_start, 600, 500):
        p_value = run_one_param(12, signal, noise)
        result[(signal, noise)] = p_value

    pickle.dump(result, open(f'p_values_{signal}_{noise_start}.pkl', 'wb'))
