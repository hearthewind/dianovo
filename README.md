# DIANovo
Under construction

# Data Directory

Here you can find sample data to try, as well as trained weights.
https://www.dropbox.com/scl/fo/ij9vjz3v9lig0vpqwmvh5/AEGAxp-NpXxKCqdDnSrKkRs?rlkey=6rdmb74ycgq2zxbrgl33fkq6a&st=jnxbe3bx&dl=0

# Requirements

Please install environment by 

``pip install -r gnova_requirements.txt``

which should take less than 10 minutes

Note that this project adopts FlashAttention 2, please refer to https://github.com/Dao-AILab/flash-attention for GPU requirement

# How to run on data

Please refer to 
``
main_model/test_op_386on453.sh
``
and
``
main_model/test_sg_386on453.sh
``
to perform de novo sequencing on the test dataset, the model will generate predicted result as well as report amino acid and peptide recall.

Depending on your data size, the model could take several hours to process.

