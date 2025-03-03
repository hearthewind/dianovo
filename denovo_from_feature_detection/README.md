# De Novo from Feature Detection

# How to generate de novo data

Please refer to DIA-Umpire for the feature detection, you will need DIA-Umpire's signal extraction module. The feature detection result is stored in the PeakCluster.csv file.
For more information, refer to https://github.com/Nesvilab/DIA-Umpire.

Under the utils folder, you can find the script to generate de novo data, ``gen_test.sh``. The program takes in DIAUmpire feature detecvtion file (PeakCluster.csv file) and spectrum files (.mzML files) and produce our internal data (.csv files for the header, as well as .msgp files for the data). We have also included a sample of PeakCluster.csv in the data folder.

The format to run the code is 

```
sh parallel_preprocessing.sh [num_worker] \ 
[mzml file path] \
[DIAUmpire PeakCluster path] \
[output dir]
```

It takes around 1.6 seconds to process a single peptide, but this process is parallelized.

# How to perform de novo sequencing

Please refer to 
``
./test_op_386on453.sh
``
and
``
./test_sg_386on453.sh
``
to perform de novo sequencing on the any dataset directly from feature detection result,

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

The model will generate predicted result, this time no performance evaluation will be performed, due to absence of ground truth label.

The sequence generation decoding will generate final de novo result in a .csv files, with the following columns
```
'graph_idx, 'pred_seq', 'pred_prob', 'pred_path'
```

It again takes around 1.5 seconds for one peptide to go through either optimal path decoding or sequence generation decoding.