python -u main.py mode=inference task=optimal_path device=gpu infer=optimal_path_inference \
serialized_model_path=save/ckpt \
gnova_file_name=GNova-Astral_pretrain-453 \
wandb.project=RNova-Astral wandb.name=op-453 \
infer.optimal_path_file=prediction/optimal_path/386on453.csv \
infer.dataset_dir=/home/m/data/Dropbox/shared/dianovo_data/generated_data_46386/ \
infer.test_spec_header_filename=02052023_Yeast_KO_3Th_2p5ms_KO1_rep01_20230503212731_2.csv
