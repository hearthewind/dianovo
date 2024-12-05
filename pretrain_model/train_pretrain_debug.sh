CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
python -u main.py mode=train device=gpu task=pretrain serialized_model_path=save/ckpt \
train_spec_header_path=~/data2/RNova-DIA-Multi-Data/pain19777_03179_train_valid/train/03179.csv \
eval_spec_header_path=~/data2/RNova-DIA-Multi-Data/pain19777_03179_train_valid/valid/1.csv \
train_dataset_dir=~/data2/RNova-DIA-Multi-Data/pain19777_03179_train_valid/train/ \
eval_dataset_dir=~/data2/RNova-DIA-Multi-Data/pain19777_03179_train_valid/valid/ \
sample=4090_24g_light wandb.project=GNova-Multi-DIA wandb.name=pretrain-2loss-debug \
train.detect_period=400 train.eval_period=2000

#
#train_spec_header_path=~/data2/RNova-DIA-Multi-Data/oc/unique/1.csv \
#eval_spec_header_path=~/data2/RNova-DIA-Multi-Data/oc/unique/1.csv \
#train_dataset_dir=~/data2/RNova-DIA-Multi-Data/oc/unique/ \
#eval_dataset_dir=~/data2/RNova-DIA-Multi-Data/oc/unique/ \

#train_spec_header_path=~/data2/RNova-DIA-Multi-Data/pain19777_03179_train_valid/train/03179.csv \
#eval_spec_header_path=~/data2/RNova-DIA-Multi-Data/pain19777_03179_train_valid/valid/1.csv \
#train_dataset_dir=~/data2/RNova-DIA-Multi-Data/pain19777_03179_train_valid/train/ \
#eval_dataset_dir=~/data2/RNova-DIA-Multi-Data/pain19777_03179_train_valid/valid/ \