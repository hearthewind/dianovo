python -u main.py mode=inference task=optimal_path device=gpu infer=optimal_path_inference \
serialized_model_path=save/ckpt wandb.project=RNova-Multi-DIA wandb.name=OptimalPath-a100 \
infer.optimal_path_file=prediction/optimal_path/oc_tiny_beam20_multiply.csv \
infer.dataset_dir=/home/m/data2/RNova-DIA-Multi-Data/oc/unique/ \
infer.test_spec_header_filename=1_tiny.csv
