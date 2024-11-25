python -u main.py mode=inference task=sequence_generation device=gpu infer=sequence_generation_inference \
serialized_model_path=save/ckpt wandb.project=RNova-Multi-DIA wandb.name=SeqGen-a100 \
infer.optimal_path_file=prediction/optimal_path/oc_tiny_beam20_sum.csv \
infer.output_file=prediction/sequence_generation/oc_tiny_beam20_sum.csv \
infer.dataset_dir=/home/m/data2/RNova-DIA-Multi-Data/oc/unique/ \
infer.test_spec_header_filename=1_tiny.csv
