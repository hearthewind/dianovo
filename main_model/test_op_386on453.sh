python -u main.py mode=inference task=optimal_path device=gpu infer=optimal_path_inference \
serialized_model_path=save/ckpt \
gnova_file_name=GNova-Astral_pretrain-ce-46453-smalllr \
wandb.project=RGNova-Astral wandb.name=rgce-op1-46453 \
infer.optimal_path_file=prediction/optimal_path/386on453_rg_imnprovement_test.csv \
infer.dataset_dir=/home/m/data3/Astral_mar28/46386/1-5/ \
infer.test_spec_header_filename=02052023_Yeast_KO_3Th_2p5ms_KO1_rep01_20230503212731_1.csv.shuffled
