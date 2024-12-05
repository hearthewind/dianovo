CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:12378 --nnodes=1 --nproc_per_node=4 main.py \
mode=train task=peak_classification_multiclass serialized_model_path=save/ckpt \
train_spec_header_path=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/train/combined.csv \
eval_spec_header_path=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/valid/1.csv \
train_dataset_dir=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/train/ \
eval_dataset_dir=~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/valid/ \
sample=a100_80g_light wandb.project=GNova-Multi-DIA wandb.name=PeakClass-pain19777
