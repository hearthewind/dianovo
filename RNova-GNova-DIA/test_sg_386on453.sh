python -u main.py mode=inference task=sequence_generation device=gpu infer=sequence_generation_inference \
serialized_model_path=save/ckpt \
gnova_file_name=GNova-Astral_pretrain-ce-46453-smalllr \
wandb.project=RGNova-Astral wandb.name=rgce-sg1-46453 \
infer.optimal_path_file=prediction/optimal_path/386on453_rg_imnprovement.csv \
infer.output_file=prediction/sequence_generation/386on453_rg_imnprovement.csv \
infer.dataset_dir=/home/m/data3/Astral_mar28/46386/1-5/ \
infer.test_spec_header_filename=02052023_Yeast_KO_3Th_2p5ms_KO1_rep01_20230503212731_1.csv.shuffled
