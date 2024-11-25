import os

import hydra
import wandb
from hydra.utils import get_original_cwd

import gnova
import pandas as pd
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from gnova.loss_func.multitask_loss import get_multitask_loss_v2


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig)->None:
    if cfg.mode == 'train':
        train_spec_header = pd.read_csv(cfg.train_spec_header_path,index_col='Spec Index')
        eval_spec_header = pd.read_csv(cfg.eval_spec_header_path,index_col='Spec Index')

        task = gnova.task.Task(cfg,serialized_model_path=cfg.serialized_model_path)
        task.initialize(train_spec_header=train_spec_header,train_dataset_dir=cfg.train_dataset_dir,val_spec_header=eval_spec_header,val_dataset_dir=cfg.eval_dataset_dir)

        if dist.is_initialized() and dist.get_rank()!=0: pass
        else:
            run = wandb.init(entity=cfg.wandb.entity,
                            project=cfg.wandb.project, name=cfg.wandb.name,
                            config=OmegaConf.to_container(cfg))
            wandb.watch(task.model, log='all', log_freq=cfg.train.eval_period*cfg.train.detect_period, log_graph=True)

        best_loss = float('inf')
        if cfg.task == 'pretrain':
            for iontype_loss_train, ionsource_loss_train, total_step, epoch, train_accu, train_pred_noise, train_pos_recall, train_pos_precision in task.train():
                if total_step % cfg.train.eval_period == 0:
                    iontype_loss_eval, ionsource_loss_eval, eval_accu, eval_pred_noise, eval_pos_recall, eval_pos_precision = task.eval()
                    if (dist.is_initialized() and dist.get_rank() == 0) or (not dist.is_initialized()):
                        print(f'epoch: {epoch}, \
step: {total_step}, \
iontype train loss: {round(iontype_loss_train, 8)}, \
iontyoe train accuracy: {round(train_accu, 8)}, \
iontype eval loss: {round(iontype_loss_eval, 8)}, \
iontype eval accuracy: {round(eval_accu, 8)}, \
ionsource train loss: {round(ionsource_loss_train, 8)}, \
ionsource eval loss: {round(ionsource_loss_eval, 8)}, \
train pred noise: {round(train_pred_noise, 8)}, \
eval pred noise: {round(eval_pred_noise, 8)}, \
train pos recall: {round(train_pos_recall, 8)}, \
eval pos recall: {round(eval_pos_recall, 8)}, \
train pos precision: {round(train_pos_precision, 8)}, \
eval pos precision: {round(eval_pos_precision, 8)}, \
lr: {task.scheduler.get_last_lr()[0]}')
                        wandb.log({'lr': task.scheduler.get_last_lr()[0], \
                                   'iontype_accuracy_train': train_accu, 'iontype_accuracy_eval': eval_accu, \
                                   'iontype_train_loss': iontype_loss_train, 'iontype_eval_loss': iontype_loss_eval, \
                                   'ionsource_train_loss': ionsource_loss_train, \
                                   'ionsource_eval_loss': ionsource_loss_eval, \
                                   'train_pred_noise': train_pred_noise, 'eval_pred_noise': eval_pred_noise, \
                                   'train_pos_recall': train_pos_recall, 'eval_pos_recall': eval_pos_recall, \
                                   'train_pos_precision': train_pos_precision, 'eval_pos_precision': eval_pos_precision, \
                                   'epoch': epoch}, step=total_step)

                        loss_eval = get_multitask_loss_v2(iontype_loss_eval, ionsource_loss_eval)
                        if best_loss > loss_eval:
                            best_loss = loss_eval
                            task.model_save()
                elif (dist.is_initialized() and dist.get_rank() == 0) or (not dist.is_initialized()):
                    print(f'epoch:{epoch}, \
step: {total_step}, \
iontype train loss:{iontype_loss_train}, \
iontype train accuracy: {train_accu}, \
ionsource train loss:{ionsource_loss_train}, \
train pred noise: {train_pred_noise}, \
train pos recall: {train_pos_recall}, \
train pos precision: {train_pos_precision}, \
lr: {task.scheduler.get_last_lr()[0]}')
                    wandb.log({'lr': task.scheduler.get_last_lr()[0], \
                               'iontype_accuracy_train': train_accu, \
                               'iontype_train_loss': iontype_loss_train, 'ionsource_train_loss': ionsource_loss_train, \
                               'train_pred_noise': train_pred_noise, \
                               'train_pos_recall': train_pos_recall, 'train_pos_precision': train_pos_precision, \
                               'epoch': epoch}, step=total_step)
        elif cfg.task == 'peak_classification':
            for loss_train, accuracy_train, train_pred_noise, total_step, epoch, train_pos_recall, train_pos_precision in task.train():
                if total_step % cfg.train.eval_period == 0:
                    loss_eval, accuracy_eval, pred_noise_eval, eval_pos_recall, eval_pos_precision = task.eval()
                    if (dist.is_initialized() and dist.get_rank() == 0) or (not dist.is_initialized()):
                        print(f'epoch:{epoch}, \
step:{total_step}, \
train loss:{round(loss_train, 8)}, \
eval loss:{round(loss_eval, 8)}, \
lr: {task.scheduler.get_last_lr()[0]}, \
train accuracy: {round(accuracy_train, 5)}, \
eval accuracy: {round(accuracy_eval, 5)}, \
train pred noise: {round(train_pred_noise, 5)}, \
eval pred noise: {round(pred_noise_eval, 5)}, \
train pos recall: {round(train_pos_recall, 5)}, \
eval pos recall: {round(eval_pos_recall, 5)}, \
train pos precision: {round(train_pos_precision, 5)}, \
eval pos precision: {round(eval_pos_precision, 5)}')
                        wandb.log({'train_loss': loss_train, 'eval_loss': loss_eval, \
                                   'train_accuracy': accuracy_train, 'train_pred_noise': train_pred_noise, \
                                   'eval_accuracy': accuracy_eval, 'eval_pred_noise': pred_noise_eval, \
                                   'train_pos_recall': train_pos_recall, 'eval_pos_recall': eval_pos_recall, \
                                   'train_pos_precision': train_pos_precision, 'eval_pos_precision': eval_pos_precision, \
                                   'epoch': epoch}, step=total_step)
                        if best_loss > loss_eval:
                            best_loss = loss_eval
                            task.model_save()
                elif (dist.is_initialized() and dist.get_rank() == 0) or (not dist.is_initialized()):
                    print(f'epoch:{epoch}, \
step: {total_step}, \
train_loss: {round(loss_train, 8)}, \
lr: {task.scheduler.get_last_lr()[0]}, \
train accuracy: {round(accuracy_train, 5)}, \
train pred noise: {round(train_pred_noise, 5)} \
train pos recall: {round(train_pos_recall, 5)}, \
train pos precision: {round(train_pos_precision, 5)}')
                    wandb.log({'train_loss': loss_train, 'lr': task.scheduler.get_last_lr()[0], \
                               'train_accuracy': accuracy_train, 'train_pred_noise': train_pred_noise, \
                               'train_pos_recall': train_pos_recall, 'train_pos_precision': train_pos_precision, \
                               'epoch': epoch}, step=total_step)
        elif cfg.task == 'optimal_path':
            for iontype_loss_train, ionsource_loss_train, loss_train, recall_train, precision_train, accuracy_train, total_step, epoch in task.train():
                if total_step%cfg.train.eval_period==0:
                    iontype_loss_eval, ionsource_loss_eval, loss_eval, recall_eval, precision_eval, accuracy_eval = task.eval()
                    if (dist.is_initialized() and dist.get_rank()==0) or (not dist.is_initialized()):
                        print(f'epoch:{epoch}, \
step:{total_step}, \
iontype train loss:{round(iontype_loss_train, 8)}, \
iontype eval loss:{round(iontype_loss_eval, 8)}, \
ionsource train loss:{round(ionsource_loss_train, 8)}, \
ionsource eval loss {round(ionsource_loss_eval, 8)}, \
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
                                   'train_accuracy': accuracy_train, 'train_recall': recall_train, \
                                   'train_precision': precision_train, \
                                   'eval_accuracy': accuracy_eval, 'eval_recall': recall_eval, \
                                   'eval_precision': precision_eval, \
                                   'iontype_train_loss': iontype_loss_train, 'iontype_eval_loss': iontype_loss_eval, \
                                   'ionsource_train_loss': ionsource_loss_train, \
                                   'ionsource_eval_loss': ionsource_loss_eval, \
                                   'epoch': epoch}, step=total_step)
                        if best_loss > loss_eval:
                            best_loss = loss_eval
                            task.model_save()
                elif (dist.is_initialized() and dist.get_rank()==0) or (not dist.is_initialized()):
                    print(f'epoch:{epoch}, \
step: {total_step}, \
iontype train loss:{iontype_loss_train}, \
ionsource train loss:{ionsource_loss_train}, \
train_loss: {round(loss_train, 8)}, \
lr: {task.scheduler.get_last_lr()[0]}, \
train accuracy: {round(accuracy_train, 5)}, \
train recall: {round(recall_train, 5)}, \
train precision: {round(precision_train, 5)}')
                    wandb.log({'train_loss': loss_train, 'lr': task.scheduler.get_last_lr()[0], \
                               'train_accuracy': accuracy_train, 'train_recall': recall_train,
                               'train_precision': precision_train, \
                               'iontype_train_loss': iontype_loss_train, 'ionsource_train_loss': ionsource_loss_train, \
                               'epoch': epoch}, step=total_step)
        elif cfg.task == 'sequence_generation':
            for iontype_loss_train, ionsource_loss_train, loss_train, total_step, epoch in task.train():
                if total_step % cfg.train.eval_period == 0:
                    iontype_loss_eval, ionsource_loss_eval, loss_eval = task.eval()
                    if (dist.is_initialized() and dist.get_rank() == 0) or (not dist.is_initialized()):
                        print(f'epoch:{epoch}, \
step:{total_step}, \
iontype train loss:{iontype_loss_train}, \
iontype eval loss:{iontype_loss_eval}, \
ionsource train loss:{ionsource_loss_train}, \
ionsource eval loss {ionsource_loss_eval}, \
train loss:{round(loss_train, 8)}, \
eval loss:{round(loss_eval, 8)}, \
lr: {task.scheduler.get_last_lr()[0]}')
                        wandb.log({'train_loss': loss_train, 'eval_loss': loss_eval, \
                                   'iontype_train_loss': iontype_loss_train, 'iontype_eval_loss': iontype_loss_eval, \
                                   'ionsource_train_loss': ionsource_loss_train,
                                   'ionsource_eval_loss': ionsource_loss_eval, \
                                   'epoch': epoch}, step=total_step)
                        if best_loss > loss_eval:
                            best_loss = loss_eval
                            task.model_save()
                elif (dist.is_initialized() and dist.get_rank() == 0) or (not dist.is_initialized()):
                    print(f'epoch:{epoch}, \
step: {total_step}, \
iontype train loss:{iontype_loss_train}, \
ionsource train loss:{ionsource_loss_train}, \
train_loss: {round(loss_train, 8)}, \
lr: {task.scheduler.get_last_lr()[0]}')
                    wandb.log({'train_loss': loss_train, 'lr': task.scheduler.get_last_lr()[0], \
                               'iontype_train_loss': iontype_loss_train, 'ionsource_train_loss': ionsource_loss_train, \
                               'epoch': epoch}, step=total_step)
        else:
            raise NotImplementedError

        if (not dist.is_initialized()) or (dist.get_rank()==0): run.finish()

    elif cfg.mode == 'inference':
        graphnovo_dir = get_original_cwd()
        print('graphnovo_dir: ', graphnovo_dir)
        dataset_dir = cfg.infer.dataset_dir
        print('dataset_dir: ', dataset_dir)
        spec_header = pd.read_csv(os.path.join(dataset_dir, cfg.infer.test_spec_header_filename), index_col='Spec Index')
        print('Full dataset shape: ', spec_header.shape)
        print('MSGP File Name list: ', set(spec_header['MSGP File Name'].tolist()))

        task = gnova.task.Task(cfg, os.path.join(graphnovo_dir, cfg.serialized_model_path))
        task.test_initialize(test_spec_header=spec_header, test_dataset_dir=cfg.infer.dataset_dir)
        task.inference()


if __name__=='__main__':
    main()
