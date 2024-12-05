CUDA_VISIBLE_DEVICES=3,1,0,2 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:19928 --nnodes=1 --nproc_per_node=4 main.py \
mode=train device=gpu task=node_classification serialized_model_path=save/ckpt \
train_spec_header_path=~/data/RNova-DIA-Multi-Data/pain19777_03179_train_valid/train/combined.csv \
eval_spec_header_path=~/data/RNova-DIA-Multi-Data/pain19777_03179_train_valid/valid/1.csv \
train_dataset_dir=~/data/RNova-DIA-Multi-Data/pain19777_03179_train_valid/train/ \
eval_dataset_dir=~/data/RNova-DIA-Multi-Data/pain19777_03179_train_valid/valid/ \
sample=a100_40g_light wandb.project=RNova-Multi-DIA wandb.name=NodeClass-pain19777-type_main_loss_add \
