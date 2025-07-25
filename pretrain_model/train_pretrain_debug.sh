CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
python -u main.py mode=train device=gpu task=pretrain serialized_model_path=save/ckpt \
train_spec_header_path=~/data/Dropbox/shared/dianovo_data/processed_data_46386/46386.csv \
eval_spec_header_path=~/data/Dropbox/shared/dianovo_data/processed_data_46386/46386.csv \
train_dataset_dir=~/data/Dropbox/shared/dianovo_data/processed_data_46386/ \
eval_dataset_dir=~/data/Dropbox/shared/dianovo_data/processed_data_46386/ \
sample=4090_24g_light wandb.project=GNova-Astral wandb.name=pretrain-debug \
train.detect_period=10 train.eval_period=100
