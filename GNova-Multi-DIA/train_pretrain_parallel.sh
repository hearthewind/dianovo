CUDA_VISIBLE_DEVICES=3,1,0,2 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:12148 --nnodes=1 --nproc_per_node=4 main.py \
mode=train task=pretrain serialized_model_path=save/ckpt \
train_spec_header_path=~/data/RNova-DIA-Multi-Data/pain19777_03179_train_valid/train/03179.csv \
eval_spec_header_path=~/data/RNova-DIA-Multi-Data/pain19777_03179_train_valid/valid/1.csv \
train_dataset_dir=~/data/RNova-DIA-Multi-Data/pain19777_03179_train_valid/train/ \
eval_dataset_dir=~/data/RNova-DIA-Multi-Data/pain19777_03179_train_valid/valid/ \
sample=a100_80g_light wandb.project=GNova-Multi-DIA wandb.name=Pretrain-03179-iontype
