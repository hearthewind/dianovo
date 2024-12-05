python -u main.py mode=inference task=sequence_generation device=gpu infer=sequence_generation_inference \
serialized_model_path=save/ckpt \
wandb.project=RGNova-Astral wandb.name=rg-sg1-46453\
infer.optimal_path_file=prediction/optimal_path/astral471_ionopt_rg453.csv \
infer.output_file=prediction/sequence_generation/astral471_ionopt_rg453.csv \
infer.dataset_dir=/home/m/data3/Astral_mar28/46471_test/1/ \
infer.test_spec_header_filename=2023115_Astral03_Evo5_UHG_SA_DIA_MSA_RR-073_250ng_01_1.csv.shuffled
