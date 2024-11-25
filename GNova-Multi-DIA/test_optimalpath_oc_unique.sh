python -u main.py mode=inference task=optimal_path infer=optimal_path_inference \
serialized_model_path=save/ckpt wandb.project=GNova-Multi-DIA wandb.name=OptimalPath-pain19777 \
infer.optimal_path_file=prediction/optimal_path/oc_unique_beam20_multiply.csv \
infer.dataset_dir=/home/m/data2/RNova-DIA-Multi-Data/oc/unique/ \
infer.test_spec_header_filename=1.csv
