# Main Model

# How to build
Please run the following command to build the Cython code

``python setup_cython.py build_ext --inplace``

# How to generate training and testing data

Under the utils folder, you can find the script to generate training data, ``gen_test.sh``. The program takes in DIA-NN report (.tsv file) and spectrum files (.mzML files) and produce our internal data (.csv files for the header, as well as .msgp files for the data).

The format to run the code is 

```
sh parallel_preprocessing.sh [num_worker1] [num_worker2] \ # num_worker1 refers to how many processes to divide up all the mzml files, and num_worker2 refers to how many processes per mzml file.
[mzml folder dir] \
[diann report path] \
[diann report path for the whole dataset] \
[output dir]
```

It takes around 1.6 seconds to process a single peptide, but this process is parallelized.

# How to train the model

Please refer to ``train_optimalpath_debug.sh`` and ``train_seqgen_debug.sh`` to see how to train the two models. 

The format for training the optimal path model is
```
python -u main.py mode=train device=gpu task=optimal_path serialized_model_path=save/ckpt \
gnova_file_name=GNova-Astral_[pretain model name] \
train_spec_header_path=[train spec header path] \
eval_spec_header_path=[valid spec header path] \
train_dataset_dir=[train msgp file dir] \
eval_dataset_dir=[valid msgp file dir] \
sample=4090_24g_light wandb.project=RNova-Astral wandb.name=[model name]
```

For the sequence generation model, the format is
```
python -u main.py mode=train device=gpu task=sequence_generation serialized_model_path=save/ckpt \
gnova_file_name=GNova-Astral_[pretain model name] \
train_spec_header_path=[train spec header path] \
eval_spec_header_path=[valid spec header path] \
train_dataset_dir=[train msgp file dir] \
eval_dataset_dir=[valid msgp file dir] \
sample=4090_24g_light wandb.project=RNova-Astral wandb.name=[model name]
```

# How to run on test data

Please make sure that ``knapsack.npy`` is stored under ``[project_dir]/main_model/save/knapsack/``, and model checkpoints are stored under ``[project_dir]/main_model/save/ckpt/``.

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

The model will generate predicted result, as well as report amino acid and peptide recall after the de novo is done.

The sequence generation decoding will generate final de novo result in a .csv files, with the following columns
```
'graph_idx, 'pred_seq', 'pred_prob', 'pred_path', 'label_seq'
```

It takes around 1.5 seconds for one peptide to go through either optimal path decoding or sequence generation decoding, but you can parallelize peptides over multiple GPUs.