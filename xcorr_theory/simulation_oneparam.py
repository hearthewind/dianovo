from joblib import Parallel, delayed
from tqdm import tqdm

from simulation import run_one_peptide
from configs import sub_weight, noise_intensity, bin_width
import numpy as np

rounds = 500
n_jobs = 32

def run_one_param(l: int, signal: int, noise: int):
    print('---- For one parameter ----')
    print('l', l, 'signal', signal, 'noise', noise)
    print('sub_weight', sub_weight, 'noise_intensity', noise_intensity, 'bin_width', bin_width)

    parallel = Parallel(n_jobs=n_jobs, return_as="generator_unordered")
    simulation_generator = parallel(delayed(run_one_peptide)(l=l, signal=signal, noise=noise) for _ in range(rounds))

    sim_results = []
    for sim_dict in tqdm(simulation_generator, total=rounds):
        sim_results.append(sim_dict)

    def rank_percent_analysis(sim_results):
        ranks = [result['peptides_with_higher_score'] for result in sim_results if result is not None]
        totals = [result['total_number_of_peptides'] for result in sim_results if result is not None]
        percents = [r / totals[i] for i, r in enumerate(ranks) if totals[i] > 0.0]

        return np.median(percents)

    p_value = rank_percent_analysis(sim_results)
    return p_value