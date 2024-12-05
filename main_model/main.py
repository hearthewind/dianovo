import os

import hydra
from hydra.utils import get_original_cwd

import wandb

import pandas as pd
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from rnova.task import Task


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig)->None:
    if cfg.mode == 'train':
        train_spec_header = pd.read_csv(cfg.train_spec_header_path,index_col='Spec Index')
        eval_spec_header = pd.read_csv(cfg.eval_spec_header_path,index_col='Spec Index')

        task = Task(cfg,serialized_model_path=cfg.serialized_model_path)
        task.initialize(train_spec_header=train_spec_header,train_dataset_dir=cfg.train_dataset_dir,val_spec_header=eval_spec_header,val_dataset_dir=cfg.eval_dataset_dir)

        if dist.is_initialized() and dist.get_rank()!=0: pass
        else:
            run = wandb.init(entity=cfg.wandb.entity,
                            project=cfg.wandb.project, name=cfg.wandb.name,
                            config=OmegaConf.to_container(cfg))
            wandb.watch(task.model, log='all', log_freq=cfg.train.eval_period*cfg.train.detect_period, log_graph=True)

        best_loss = float('inf')
        if cfg.task == 'node_classification' or cfg.task == 'optimal_path':
            for loss_train, recall_train, precision_train, accuracy_train, total_step, epoch in task.train():
                if total_step%cfg.train.eval_period==0:
                    loss_eval, recall_eval, precision_eval, accuracy_eval = task.eval()
                    if (dist.is_initialized() and dist.get_rank()==0) or (not dist.is_initialized()):
                        print(f'epoch:{epoch}, \
step:{total_step}, \
train loss:{round(loss_train, 8)}, \
eval loss:{round(loss_eval, 8)}, \
lr: {task.scheduler.get_last_lr()[0]}, \
train accuracy: {round(accuracy_train, 5)}, \
eval accuracy {round(accuracy_eval, 5)}, \
train recall {round(recall_train, 5)}, \
eval recall: {round(recall_eval, 5)}, \
train precision: {round(precision_train, 5)}, \
eval precision {round(precision_eval, 5)}')
                        wandb.log({'train_loss': loss_train, 'eval_loss': loss_eval, \
                                   'train_accuracy': accuracy_train, 'train_recall': recall_train, 'train_precision': precision_train, \
                                   'eval_accuracy': accuracy_eval, 'eval_recall': recall_eval, 'eval_precision': precision_eval, \
                                   'epoch': epoch}, step=total_step)
                        if best_loss > loss_eval:
                            best_loss = loss_eval
                            task.model_save()
                elif (dist.is_initialized() and dist.get_rank()==0) or (not dist.is_initialized()):
                    print(f'epoch:{epoch}, \
step: {total_step}, \
train_loss: {round(loss_train, 8)}, \
lr: {task.scheduler.get_last_lr()[0]}, \
train accuracy: {round(accuracy_train, 5)}, \
train recall: {round(recall_train, 5)}, \
train precision: {round(precision_train, 5)}')
                    wandb.log({'train_loss': loss_train, 'lr': task.scheduler.get_last_lr()[0], \
                               'train_accuracy': accuracy_train, 'train_recall': recall_train, 'train_precision': precision_train, \
                               'epoch': epoch}, step=total_step)
        elif cfg.task == 'sequence_generation':
            for loss_train, total_step, epoch in task.train():
                if total_step%cfg.train.eval_period==0:
                    loss_eval = task.eval()
                    if (dist.is_initialized() and dist.get_rank()==0) or (not dist.is_initialized()):
                        print(f'epoch:{epoch}, \
step:{total_step}, \
train loss:{round(loss_train, 8)}, \
eval loss:{round(loss_eval, 8)}, \
lr: {task.scheduler.get_last_lr()[0]}')
                        wandb.log({'train_loss': loss_train, 'eval_loss': loss_eval, \
                                   'epoch': epoch}, step=total_step)
                        if best_loss > loss_eval:
                            best_loss = loss_eval
                            task.model_save()
                elif (dist.is_initialized() and dist.get_rank()==0) or (not dist.is_initialized()):
                    print(f'epoch:{epoch}, \
step: {total_step}, \
train_loss: {round(loss_train, 8)}, \
lr: {task.scheduler.get_last_lr()[0]}')
                    wandb.log({'train_loss': loss_train, 'lr': task.scheduler.get_last_lr()[0], \
                               'epoch': epoch}, step=total_step)
        else:
            raise NotImplementedError

        if (not dist.is_initialized()) or (dist.get_rank()==0): run.finish()
    elif cfg.mode == 'inference':
        dataset_dir = cfg.infer.dataset_dir
        print('dataset_dir: ', dataset_dir)
        spec_header = pd.read_csv(os.path.join(dataset_dir, cfg.infer.test_spec_header_filename), index_col='Spec Index')
        print('Full dataset shape: ', spec_header.shape)
        print('MSGP File Name list: ', set(spec_header['MSGP File Name'].tolist()))

        task = Task(cfg, serialized_model_path=cfg.serialized_model_path)
        task.test_initialize(test_spec_header=spec_header, test_dataset_dir=cfg.infer.dataset_dir)
        task.inference()
    else:
        raise NotImplementedError


if __name__=='__main__':
    main()
