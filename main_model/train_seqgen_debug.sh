CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
python -u main.py mode=train device=gpu task=sequence_generation serialized_model_path=save/ckpt \
gnova_file_name=GNova-Astral_pretrain-453 \
train_spec_header_path=~/data/Dropbox/shared/dianovo_data/processed_data_46386/46386.csv \
eval_spec_header_path=~/data/Dropbox/shared/dianovo_data/processed_data_46386/46386.csv \
train_dataset_dir=~/data/Dropbox/shared/dianovo_data/processed_data_46386/ \
eval_dataset_dir=~/data/Dropbox/shared/dianovo_data/processed_data_46386/ \
sample=4090_24g_light wandb.project=RGNova-Astral wandb.name=SeqGen-debug \
train.eval_period=100 train.detect_period=10
