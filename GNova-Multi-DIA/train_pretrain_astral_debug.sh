CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
python -u main.py mode=train device=gpu task=pretrain serialized_model_path=save/ckpt \
train_spec_header_path=~/data3/Astral_mar28/46453_train_valid/train1.csv \
eval_spec_header_path=~/data3/Astral_mar28/46453_train_valid/valid1.csv \
train_dataset_dir=~/data3/Astral_mar28/46453_train_valid/ \
eval_dataset_dir=~/data3/Astral_mar28/46453_train_valid/ \
sample=4090_24g_light wandb.project=GNova-Astral wandb.name=pretrain-2loss-453-large \
train.detect_period=100 train.eval_period=1000

#
#train_spec_header_path=~/data2/RNova-DIA-Multi-Data/oc/unique/1.csv \
#eval_spec_header_path=~/data2/RNova-DIA-Multi-Data/oc/unique/1.csv \
#train_dataset_dir=~/data2/RNova-DIA-Multi-Data/oc/unique/ \
#eval_dataset_dir=~/data2/RNova-DIA-Multi-Data/oc/unique/ \

#train_spec_header_path=~/data2/RNova-DIA-Multi-Data/pain19777_03179_train_valid/train/03179.csv \
#eval_spec_header_path=~/data2/RNova-DIA-Multi-Data/pain19777_03179_train_valid/valid/1.csv \
#train_dataset_dir=~/data2/RNova-DIA-Multi-Data/pain19777_03179_train_valid/train/ \
#eval_dataset_dir=~/data2/RNova-DIA-Multi-Data/pain19777_03179_train_valid/valid/ \