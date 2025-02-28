# Pretrain Model

# How to generate training data

Please refer to the ``main_model`` folder for the detailed instruction on how to generate training data. These two models utilize identical training data.

# How to train the model

Please refer to ``train_pretrain_debug.sh`` to see how to train the model. 
You will need to provide a ``train_spec_header`` and a ``valid_spec_header``, which are generated from raw data.

The format for training the model is 
```
python -u main.py mode=train device=gpu task=pretrain serialized_model_path=save/ckpt \
train_spec_header_path=[train spec header path] \
eval_spec_header_path=[valid spec header path] \
train_dataset_dir=[train msgp file dir] \
eval_dataset_dir=[valid msgp file dir] \
sample=4090_24g_light wandb.project=GNova-Astral wandb.name=[model name]
```