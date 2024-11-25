python -u main.py mode=inference task=pretrain infer=pretrain_inference \
serialized_model_path=save/ckpt wandb.project=GNova-Multi-DIA wandb.name=pretrain-2loss-debug-gradaccu100-largelr \
infer.output_path=prediction/pretrain/pain19777_03179/ \
infer.output_filename=pain19777_03179.csv \
infer.dataset_dir=/home/m/data2/RNova-DIA-Multi-Data/pain19777_03179_train_valid/train/ \
infer.test_spec_header_filename=combined.csv
