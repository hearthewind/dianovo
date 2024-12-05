CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python -u main.py mode=train device=gpu task=sequence_generation serialized_model_path=save/ckpt \
gnova_file_name=GNova-Multi-DIA_pretrain-2loss-debug-gradaccu100-largelr \
train_spec_header_path=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/train/combined.csv \
eval_spec_header_path=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/valid/1.csv \
train_dataset_dir=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/train/ \
eval_dataset_dir=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/valid/ \
sample=4090_24g_light wandb.project=RNova-GNova-DIA wandb.name=SeqGen-debug \
train.eval_period=400 train.detect_period=20

#train_spec_header_path=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/train/combined.csv \
#eval_spec_header_path=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/valid/1.csv \
#train_dataset_dir=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/train/ \
#eval_dataset_dir=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/valid/ \