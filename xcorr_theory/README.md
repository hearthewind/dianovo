# XCorr Theory about De Novo Performance

# How to build
Please run the following command to build the Cython code

``python setup_cython.py build_ext --inplace``

# How to run simulation on one dataset parameter
Please run the following command to run simulation on one dataset parameter

``python main_oneparam.py [peptide_len] [noise_intensity] [num_signals] [num_noise]``

For example

``python main_oneparam.py 12 0.5 50 500``

# How to generate simulated heatmap of different signal and noise values
Please run

``python main_heatmap.py``