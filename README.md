# DIANovo

This is the codebase for the paper "Disentangling the Complex Multiplexed DIA Spectra in De Novo Peptide Sequencing".

Under the ``xcorr_theory`` folder, you can find the code for the theoretical analysis of peptide recall under different experimental conditions. Under folder ``pretrain_model`` and ``main_model`` you can find the pretrain and downstream models respectively.


For training details, please refer to the README in the corresponding subfolder.

# Data Directory

Here you can find sample data to try, as well as trained weights.
https://www.dropbox.com/scl/fo/ij9vjz3v9lig0vpqwmvh5/AEGAxp-NpXxKCqdDnSrKkRs?rlkey=6rdmb74ycgq2zxbrgl33fkq6a&st=jnxbe3bx&dl=0

# Software and Hardware Requirements

Please install the Python environment by 

``pip install -r gnova_requirements.txt``

which should take less than 10 minutes, the program is tested on Ubuntu 24.04 with Python 3.9.

Note that this project adopts FlashAttention 2, please refer to https://github.com/Dao-AILab/flash-attention for GPU requirement.

# How to run on test data

Please refer to 
``
main_model/test_op_386on453.sh
``
and
``
main_model/test_sg_386on453.sh
``
to perform de novo sequencing on the test dataset,

The format for running optimal path decoding is 
```
python -u main.py mode=inference task=optimal_path device=gpu infer=optimal_path_inference \
serialized_model_path=save/ckpt \
gnova_file_name=GNova-Astral_[pretrain model name] \
wandb.project=RNova-Astral wandb.name=[optimal path model name] \
infer.optimal_path_file=[output file path] \
infer.dataset_dir=[test msgp file dir] \
infer.test_spec_header_filename=[test spec header filename]
```

The format for running sequence generation decoding is
```
python -u main.py mode=inference task=sequence_generation device=gpu infer=sequence_generation_inference \
serialized_model_path=save/ckpt \
gnova_file_name=GNova-Astral_[pretrain model name] \
wandb.project=RNova-Astral wandb.name=[sequence generation model name] \
infer.optimal_path_file=[optimal path result filepath] \
infer.output_file=[output file path] \
infer.dataset_dir=[test msgp file dir] \
infer.test_spec_header_filename=[test spec header filename]
```

The model will generate predicted result, as well as report amino acid and peptide recall afterwards after the de novo is done.

The sequence generation decoding will generate final de novo result in a .csv files, with the following columns
```
'graph_idx, 'pred_seq', 'pred_prob', 'pred_path', 'label_seq'
```

It takes around 1.5 seconds for one peptide to go through either optimal path decoding or sequence generation decoding, but you can parallelize peptides over multiple GPUs.