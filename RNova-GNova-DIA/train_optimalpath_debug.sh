CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python -u main.py mode=train device=gpu task=optimal_path serialized_model_path=save/ckpt \
gnova_file_name=GNova-Astral_pretraince_453_try \
train_spec_header_path=~/data3/Astral_mar28/46453_train_valid/valid1.csv \
eval_spec_header_path=~/data3/Astral_mar28/46453_train_valid/valid1.csv \
train_dataset_dir=~/data3/Astral_mar28/46453_train_valid/ \
eval_dataset_dir=~/data3/Astral_mar28/46453_train_valid/ \
sample=4090_24g_light wandb.project=RNova-GNova-Astral wandb.name=OptimalPath-debug \
train.eval_period=400 train.detect_period=20

#train_spec_header_path=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/train/combined.csv \
#eval_spec_header_path=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/valid/1.csv \
#train_dataset_dir=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/train/ \
#eval_dataset_dir=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/valid/ \