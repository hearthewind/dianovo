python -u main.py mode=inference task=optimal_path device=gpu infer=optimal_path_inference \
serialized_model_path=save/ckpt \
gnova_file_name=GNova-Astral_pretrain-celarge-46386 \
wandb.project=RGNova-Astral wandb.name=rgce-op1-46386 \
infer.optimal_path_file=prediction/optimal_path/astral453_ionopt_rg386large.csv \
infer.dataset_dir=/home/m/data3/Astral_mar28/46453/2p5ms/ \
infer.test_spec_header_filename=20230320_OLEP08_1000ngHeK_uPAC_180k-30min_MontBlanc_2p5ms_0p5_2Th_01_1.csv.shuffled
