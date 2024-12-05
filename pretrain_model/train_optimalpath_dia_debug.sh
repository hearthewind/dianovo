CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
python -u main.py mode=train device=gpu task=optimal_path serialized_model_path=save/ckpt \
train_spec_header_path=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/train/combined.csv \
eval_spec_header_path=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/valid/1.csv \
train_dataset_dir=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/train/ \
eval_dataset_dir=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/valid/ \
sample=v100_16g_light wandb.project=GNova-Multi-DIA wandb.name=OptimalPath-debug \
train.detect_period=4 train.eval_period=20