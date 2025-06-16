# DIANovo

This is the codebase for the paper "Disentangling the Complex Multiplexed DIA Spectra in De Novo Peptide Sequencing".

# Notice

Our repository adopts new checkpoints different from the previous version, please make sure to download the new checkpoints via git-lfs, the new checkpoints should automatically download to appropriate locations during ``git clone``.

On Ubuntu, you can install git-lfs by
```
sudo apt install git-lfs
git lfs install
```

# Subfolders and more README

Under the ``xcorr_theory`` folder, you can find the code for the theoretical analysis of peptide recall under different experimental conditions. Under folder ``pretrain_model`` and ``main_model`` you can find the pretrain and downstream models respectively.
Under the ``denovo_from_feature_detection`` folder, we provide the code to run de novo sequencing directly from feature detection data and mzml spectrum files.

For training, data generation, as well as denovo from feature detection result, please refer to the README file in the corresponding subfolder.

# Data directory

Here you can find sample data to try, as well as trained weights and knapsack matrix.
https://www.dropbox.com/scl/fo/ij9vjz3v9lig0vpqwmvh5/AEGAxp-NpXxKCqdDnSrKkRs?rlkey=6rdmb74ycgq2zxbrgl33fkq6a&st=jnxbe3bx&dl=0

# Software and hardware requirements

Please install the Python environment using conda by 

``conda env create -f environment_new.yml``

which should take less than 10 minutes, the program is tested on Ubuntu 24.04 with Python 3.9.

Note that this project adopts FlashAttention 2, please refer to https://github.com/Dao-AILab/flash-attention for GPU requirement.

