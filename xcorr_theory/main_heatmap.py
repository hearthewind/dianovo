import pickle
import sys

from simulation_oneparam import run_one_param

if __name__ == '__main__':
    result = {}

    for signal in range(100, 1010, 100):
        for noise in range(100, 10100, 500):
            p_value = run_one_param(12, signal, noise, 0.5)
            result[(signal, noise)] = p_value

    pickle.dump(result, open(f'p_values.pkl', 'wb'))
