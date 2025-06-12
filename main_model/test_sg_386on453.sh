#!/bin/bash

python -u main.py mode=inference task=sequence_generation device=gpu infer=sequence_generation_inference \
serialized_model_path=save/ckpt \
gnova_file_name=GNova-Astral_newpretrain-453 \
wandb.project=RGNova-Astral wandb.name=newsg-453 \
infer.optimal_path_file=prediction/optimal_path/386on453.csv \
infer.output_file=prediction/sequence_generation/386on453.csv \
infer.dataset_dir=/home/m/data/Dropbox/shared/dianovo_data/processed_data_46386/ \
infer.test_spec_header_filename=46386.csv
