python -u main.py mode=inference task=sequence_generation infer=simple_decoder_inference \
serialized_model_path=save/ckpt wandb.project=GNova-Multi-DIA wandb.name=SeqGen-debug \
infer.optimal_path_file=prediction/optimal_path/oc_unique_beam20_multiply.csv \
infer.output_file=prediction/sequence_generation/oc_unique_beam20_sum.csv \
infer.dataset_dir=/home/m/data2/RNova-DIA-Multi-Data/oc/unique/ \
infer.test_spec_header_filename=1.csv
