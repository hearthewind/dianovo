CUDA_VISIBLE_DEVICES=0 python -u main.py mode=train device=gpu task=node_classification serialized_model_path=save/ckpt \
train_spec_header_path=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/train/combined.csv \
eval_spec_header_path=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/valid/1.csv \
train_dataset_dir=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/train/ \
eval_dataset_dir=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/valid/ \
sample=4090_24g_light wandb.project=RNova-Multi-DIA wandb.name=NodeClass-pain19777-type_main_loss-debug \
train.eval_period=40 train.detect_period=4

#train_spec_header_path=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/train/combined.csv \
#eval_spec_header_path=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/valid/1.csv \
#train_dataset_dir=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/train/ \
#eval_dataset_dir=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/valid/ \

#train_spec_header_path=~/data2/RNova-DIA-Multi-Data/oc/unique/1.csv \
#eval_spec_header_path=~/data2/RNova-DIA-Multi-Data/oc/unique/1.csv \
#train_dataset_dir=~/data2/RNova-DIA-Multi-Data/oc/unique/ \
#eval_dataset_dir=~/data2/RNova-DIA-Multi-Data/oc/unique/ \

#train_spec_header_path=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/valid/1.csv \
#eval_spec_header_path=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/valid/1.csv \
#train_dataset_dir=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/valid/ \
#eval_dataset_dir=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/valid/ \

# CUDA_LAUNCH_BLOCKING=1