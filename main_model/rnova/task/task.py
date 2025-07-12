import os
import torch
from torch import nn, optim

from rnova.data import RNovaDataset, RNovaBucketBatchSampler, DataPrefetcher, RNovaCollator, RNovaSequentialSampler
from rnova.loss_func.consistency_loss import noble_loss, ClipLoss, infonce_loss
from rnova.loss_func.focal_loss import FocalLossWithLogits, MultiClassFocalLossWithLogits
from rnova.loss_func.multitask_loss import get_multitask_loss, get_multitask_loss_v2
from rnova.model.rnova import RNova
from rnova.model_gnova import GNova
from rnova.optimizer import Lion
from rnova.optimizer.warmup_scheduler import get_lr_scheduler
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from functools import partial

from rnova.task.init_distributed import initialize_distributed
from rnova.task.optimal_path_inference_classic import OptimalPathInference
from rnova.task.sequence_generation_inference import sequence_generation_inference

float16_type = torch.bfloat16
# float16_type = torch.float

class Task:
    def __init__(self, cfg, serialized_model_path):
        self.cfg = cfg
        self.serialized_model_path = serialized_model_path

        if cfg.mode == 'train':
            try:
                self.local_rank = initialize_distributed(cfg.server)
                self.device = torch.device("cuda", self.local_rank)

                self.distributed = True
            except ValueError as _:
                self.local_rank = 0
                if self.cfg.device == 'cpu':
                    self.device = torch.device("cpu")
                elif self.cfg.device == 'gpu':
                    self.device = torch.device("cuda", self.local_rank)
                    torch.cuda.set_device(self.local_rank)
                self.distributed = False
        else:
            if self.cfg.device == 'gpu':
                self.device = torch.device("cuda", 0)
                torch.cuda.set_device(0)
            elif self.cfg.device == 'cpu':
                self.device = torch.device("cpu")
            self.distributed = False

    def initialize(self, *, train_spec_header,train_dataset_dir,val_spec_header,val_dataset_dir):
        self.model = RNova(self.cfg, float16_type).to(self.device)
        self.model_gnova = GNova(self.cfg, float16_type).to(self.device)
        self.model_gnova.eval()

        if self.cfg.task == 'node_classification' or self.cfg.task == 'optimal_path':
            # self.train_loss_fn = nn.BCEWithLogitsLoss()
            # self.eval_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

            focal_loss = FocalLossWithLogits(gamma=1)
            self.train_loss_fn = partial(focal_loss, reduction='mean')
            self.eval_loss_fn = partial(focal_loss, reduction='sum')
        elif self.cfg.task == 'sequence_generation':
            self.train_loss_fn = nn.CrossEntropyLoss(reduction='mean')
            self.eval_loss_fn = nn.CrossEntropyLoss(reduction='sum')
        else:
            raise NotImplementedError

        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            self.model_gnova = DDP(self.model_gnova, device_ids=[self.local_rank], output_device=self.local_rank)

        # self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.train.lr)
        self.optimizer = Lion(self.model.parameters(), lr=self.cfg.train.lr)
        self.scheduler = get_lr_scheduler(self.optimizer, self.cfg.train.warmup_steps)
        self.scaler = GradScaler()

        self.gnova_model_file = os.path.join(self.serialized_model_path, self.cfg.gnova_file_name) + '.pt'
        if os.path.exists(self.gnova_model_file):
            checkpoint_gnova = torch.load(self.gnova_model_file, map_location={f'cuda:{0}': f'cuda:{self.local_rank}'})
            if self.distributed: self.model_gnova.module.load_state_dict(checkpoint_gnova['model_state_dict'])
            else: self.model_gnova.load_state_dict(checkpoint_gnova['model_state_dict'])
            print(f'load gnova model from {self.gnova_model_file}')
        else:
            raise FileNotFoundError('No Gnova Checkpoint')

        self.persistent_file_name = os.path.join(self.serialized_model_path, self.cfg.wandb.project + '_' + self.cfg.wandb.name + '.pt')
        if os.path.exists(self.persistent_file_name):
            checkpoint = torch.load(self.persistent_file_name,map_location={f'cuda:{0}': f'cuda:{self.local_rank}'})
            if self.distributed: self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else: self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f'load model from {self.persistent_file_name}')
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print('no model detected, start from scratch')


        self.train_dl = self.train_loader(train_spec_header,train_dataset_dir)
        self.eval_dl = self.eval_loader(val_spec_header,val_dataset_dir)

    def test_initialize(self, *, test_spec_header=None,test_dataset_dir=None):
        assert not self.distributed

        self.model = RNova(self.cfg, float16_type).to(self.device)
        self.persistent_file_name = os.path.join(self.serialized_model_path, self.cfg.wandb.project + '_' + self.cfg.wandb.name + '.pt')
        print('checkpoint: ', self.persistent_file_name)
        assert os.path.exists(self.persistent_file_name)
        if self.cfg.device == 'gpu':
            checkpoint = torch.load(self.persistent_file_name)
        else:
            checkpoint = torch.load(self.persistent_file_name, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.model_gnova = GNova(self.cfg, float16_type).to(self.device)
        self.model_gnova.eval()
        self.gnova_model_file = os.path.join(self.serialized_model_path, self.cfg.gnova_file_name) + '.pt'
        if os.path.exists(self.gnova_model_file):
            if self.cfg.device == 'gpu':
                checkpoint_gnova = torch.load(self.gnova_model_file)
            else:
                checkpoint_gnova = torch.load(self.gnova_model_file, map_location='cpu')
            self.model_gnova.load_state_dict(checkpoint_gnova['model_state_dict'])
            print(f'load gnova model from {self.gnova_model_file}')
        else:
            raise FileNotFoundError('No Gnova Checkpoint')

        self.test_dl = self.test_loader(test_spec_header, test_dataset_dir)
        self.test_spec_header = test_spec_header

    def train_loader(self,train_spec_header,train_dataset_dir):
        ds = RNovaDataset(self.cfg, spec_header=train_spec_header,dataset_dir_path=train_dataset_dir)
        sampler = RNovaBucketBatchSampler(self.cfg,train_dataset_dir,train_spec_header,self.cfg.sample.train_bin_boarders,self.cfg.sample.train_bin_batch_size,shuffle=True, drop_last=True)
        train_dl = DataLoader(ds,batch_sampler=sampler,collate_fn=RNovaCollator(self.cfg),pin_memory=True,num_workers=4)
        train_dl = DataPrefetcher(train_dl,self.device)
        return train_dl

    def eval_loader(self,val_spec_header,val_dataset_dir):
        ds = RNovaDataset(self.cfg, spec_header=val_spec_header,dataset_dir_path=val_dataset_dir)
        sampler = RNovaBucketBatchSampler(self.cfg,val_dataset_dir,val_spec_header,self.cfg.sample.eval_bin_boarders,self.cfg.sample.eval_bin_batch_size,shuffle=False, drop_last=False)
        collate_fn = RNovaCollator(self.cfg)
        if self.distributed:
            eval_dl = DataLoader(ds,batch_sampler=sampler,collate_fn=collate_fn,pin_memory=True,num_workers=4)
        else:
            eval_dl = DataLoader(ds,batch_sampler=sampler,collate_fn=collate_fn,pin_memory=True)
        eval_dl = DataPrefetcher(eval_dl,self.device)
        return eval_dl

    def test_loader(self,test_spec_header,test_dataset_dir):
        ds = RNovaDataset(self.cfg, spec_header=test_spec_header,dataset_dir_path=test_dataset_dir)
        sampler = RNovaSequentialSampler(self.cfg, test_dataset_dir, test_spec_header)
        collate_fn = RNovaCollator(self.cfg)
        test_dl = DataLoader(ds,batch_sampler=sampler,collate_fn=collate_fn,pin_memory=True)
        if self.cfg.device == 'gpu':
            test_dl = DataPrefetcher(test_dl,self.device)
        return test_dl
    
    def model_save(self):
        if self.distributed:
            torch.save({'model_state_dict':self.model.module.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
            'scheduler_state_dict':self.scheduler.state_dict()},self.persistent_file_name)
        else:
            torch.save({'model_state_dict':self.model.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
            'scheduler_state_dict':self.scheduler.state_dict()},self.persistent_file_name)

    def train(self):
        total_step = 0
        loss_cum = 0
        true_positive = 0
        total_true = 0
        total_positive = 0
        total_seq_len = 0
        total_match = 0
        if self.cfg.task == 'node_classification':
            for epoch in range(0, self.cfg.train.total_epoch):
                for i, d in enumerate(self.train_dl,start=1):
                    self.model.train()

                    encoder_input, label, label_mask = d['rnova']
                    gnova_encoder_input, _ = d['gnova']
                    meta_info_list = d['meta_info']

                    total_step += 1
                    # self.optimizer.zero_grad(set_to_none=True) #TODO(m) for gradient accumulation
                    if total_step%self.cfg.train.detect_period == 1:
                        loss_cum = 0
                        true_positive = 0
                        total_true = 0
                        total_positive = 0
                        total_seq_len = 0
                        total_match = 0
                    if self.cfg.device == 'gpu':
                        with autocast(dtype=float16_type):
                            pred = self.model(self.model_gnova, encoder_input, gnova_encoder_input, meta_info_list)

                            pred = pred[label_mask].flatten()
                            label = label[label_mask].flatten()
                            pred_loss = self.train_loss_fn(pred,label)

                            loss = pred_loss
                    elif self.cfg.device == 'cpu':
                        pred = self.model(self.model_gnova, encoder_input, gnova_encoder_input, meta_info_list)

                        pred = pred[label_mask].flatten()
                        label = label[label_mask].flatten()
                        pred_loss = self.train_loss_fn(pred, label)

                        loss = pred_loss

                    if self.cfg.device == 'gpu':
                        self.scaler.scale(loss).backward()
                        if (i + 1) % self.cfg.train.gradient_accumulation_step == 0:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad(set_to_none=True)
                    elif self.cfg.device == 'cpu':
                        loss.backward()
                        self.optimizer.step()

                    self.scheduler.step()

                    pred = (pred > 0.0).bool()
                    loss_cum += pred_loss.item()
                    true_positive += (pred[label.bool()] > 0).sum()
                    total_true += label.sum()
                    total_positive += (pred > 0).sum()
                    total_seq_len += label_mask.sum()
                    total_match += (pred == label).sum()

                    report_loss = loss_cum / self.cfg.train.detect_period
                    recall = (true_positive / total_true).item()
                    precision = (true_positive / total_positive).item()
                    accuracy = (total_match / total_seq_len).item()

                    if total_step%self.cfg.train.detect_period == 0:
                        yield report_loss, recall, precision, accuracy, total_step, epoch
        elif self.cfg.task == 'optimal_path':
            for epoch in range(0, self.cfg.train.total_epoch):
                for i, d in enumerate(self.train_dl, start=1):
                    self.model.train()

                    encoder_input, decoder_input, label, label_mask, _ = d['rnova']
                    gnova_encoder_input, _ = d['gnova']
                    meta_info_list = d['meta_info']

                    total_step += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    if total_step % self.cfg.train.detect_period == 1:
                        loss_cum = 0
                        true_positive = 0
                        total_true = 0
                        total_positive = 0
                        total_seq_len = 0
                        total_match = 0
                    if self.cfg.device == 'gpu':
                        with autocast(dtype=float16_type):
                            pred = self.model(self.model_gnova, encoder_input, gnova_encoder_input, meta_info_list, decoder_input=decoder_input)

                            pred = pred[label_mask].flatten()
                            label = label[label_mask].flatten()
                            pred_loss = self.train_loss_fn(pred, label)

                            loss = pred_loss
                    elif self.cfg.device == 'cpu':
                        pred = self.model(self.model_gnova, encoder_input, gnova_encoder_input, meta_info_list, decoder_input=decoder_input)

                        pred = pred[label_mask].flatten()
                        label = label[label_mask].flatten()
                        pred_loss = self.train_loss_fn(pred, label)

                        loss = pred_loss

                    if self.cfg.device == 'gpu':
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    elif self.cfg.device == 'cpu':
                        loss.backward()
                        self.optimizer.step()

                    self.scheduler.step()

                    pred = (pred > 0.0).bool()
                    loss_cum += pred_loss.item()
                    true_positive += (pred[label.bool()] > 0).sum()
                    total_true += label.sum()
                    total_positive += (pred > 0).sum()
                    total_seq_len += label_mask.sum()
                    total_match += (pred == label).sum()

                    report_loss = loss_cum / self.cfg.train.detect_period
                    recall = (true_positive / total_true).item()
                    precision = (true_positive / total_positive).item()
                    accuracy = (total_match / total_seq_len).item()

                    if total_step % self.cfg.train.detect_period == 0:
                        yield report_loss, recall, precision, accuracy, total_step, epoch
        elif self.cfg.task == 'sequence_generation':
            for epoch in range(0, self.cfg.train.total_epoch):
                for i, d in enumerate(self.train_dl, start=1):
                    self.model.train()

                    encoder_input, decoder_input, label, label_mask, _ = d['rnova']
                    gnova_encoder_input, _ = d['gnova']
                    meta_info_list = d['meta_info']

                    total_step += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    if total_step % self.cfg.train.detect_period == 1:
                        loss_cum = 0
                    if self.cfg.device == 'gpu':
                        with autocast(dtype=float16_type):
                            pred = self.model(self.model_gnova, encoder_input, gnova_encoder_input, meta_info_list, decoder_input=decoder_input)

                            pred = pred[label_mask]
                            label = label[label_mask]
                            pred_loss = self.train_loss_fn(pred, label)

                            loss = pred_loss
                    elif self.cfg.device == 'cpu':
                        pred = self.model(self.model_gnova, encoder_input, gnova_encoder_input, meta_info_list, decoder_input=decoder_input)

                        pred = pred[label_mask]
                        label = label[label_mask]
                        pred_loss = self.train_loss_fn(pred, label)

                        loss = pred_loss

                    loss_cum += pred_loss.item()

                    if self.cfg.device == 'gpu':
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    elif self.cfg.device == 'cpu':
                        loss.backward()
                        self.optimizer.step()

                    self.scheduler.step()

                    report_loss = loss_cum / self.cfg.train.detect_period
                    if total_step % self.cfg.train.detect_period == 0:
                        yield report_loss, total_step, epoch
        else:
            raise NotImplementedError

    def eval(self):
        self.model.eval()

        if self.cfg.task == 'node_classification':
            loss_cum = torch.Tensor([0]).to(self.device)
            total_seq_len = torch.Tensor([0]).to(self.device)

            total_match = torch.Tensor([0]).to(self.device)
            true_positive = torch.Tensor([0]).to(self.device)
            total_positive = torch.Tensor([0]).to(self.device)
            total_true = torch.Tensor([0]).to(self.device)

            for i, d in enumerate(self.eval_dl, start=1):
                encoder_input, label, label_mask = d['rnova']
                gnova_encoder_input, _ = d['gnova']
                meta_info_list = d['meta_info']

                with torch.no_grad():
                    if self.cfg.device == 'gpu':
                        with autocast(dtype=float16_type):
                            pred = self.model(self.model_gnova, encoder_input, gnova_encoder_input, meta_info_list)

                            pred = pred[label_mask].flatten()
                            label = label[label_mask].flatten()
                            pred_loss = self.eval_loss_fn(pred, label)

                    elif self.cfg.device == 'cpu':
                        pred = self.model(self.model_gnova, encoder_input, gnova_encoder_input, meta_info_list)

                        pred = pred[label_mask].flatten()
                        label = label[label_mask].flatten()
                        pred_loss = self.eval_loss_fn(pred, label)

                    pred = (pred>0.0).bool()

                    loss_cum += pred_loss

                    true_positive += (pred[label.bool()] > 0).sum()
                    total_true += label.sum()
                    total_positive += (pred > 0).sum()
                    total_seq_len += label_mask.sum()
                    total_match += (pred == label).sum()
            if self.distributed:
                dist.barrier()
                dist.all_reduce(loss_cum)
                dist.all_reduce(total_seq_len)
                dist.all_reduce(total_match)
                dist.all_reduce(true_positive)
                dist.all_reduce(total_positive)
                dist.all_reduce(total_true)

            report_loss = (loss_cum / total_seq_len).item()
            recall = (true_positive / total_true).item()
            precision = (true_positive / total_positive).item()
            accuracy = (total_match / total_seq_len).item()

            return report_loss, recall, precision, accuracy
        elif self.cfg.task == 'optimal_path':
            loss_cum = torch.Tensor([0]).to(self.device)
            total_seq_len = torch.Tensor([0]).to(self.device)

            total_match = torch.Tensor([0]).to(self.device)
            true_positive = torch.Tensor([0]).to(self.device)
            total_positive = torch.Tensor([0]).to(self.device)
            total_true = torch.Tensor([0]).to(self.device)

            for i, d in enumerate(self.eval_dl, start=1):
                encoder_input, decoder_input, label, label_mask, _ = d['rnova']
                gnova_encoder_input, _ = d['gnova']
                meta_info_list = d['meta_info']
                
                with torch.no_grad():
                    if self.cfg.device == 'gpu':
                        with autocast(dtype=float16_type):
                            pred = self.model(self.model_gnova, encoder_input, gnova_encoder_input, meta_info_list, decoder_input=decoder_input)

                            pred = pred[label_mask].flatten()
                            label = label[label_mask].flatten()
                            pred_loss = self.eval_loss_fn(pred, label)
                    elif self.cfg.device == 'cpu':
                        pred = self.model(self.model_gnova, encoder_input, gnova_encoder_input, meta_info_list, decoder_input=decoder_input)

                        pred = pred[label_mask].flatten()
                        label = label[label_mask].flatten()
                        pred_loss = self.eval_loss_fn(pred, label)

                    pred = (pred > 0.0).bool()
                    loss_cum += pred_loss

                    true_positive += (pred[label.bool()] > 0).sum()
                    total_true += label.sum()
                    total_positive += (pred > 0).sum()
                    total_seq_len += label_mask.sum()
                    total_match += (pred == label).sum()
            if self.distributed:
                dist.barrier()
                dist.all_reduce(loss_cum)
                dist.all_reduce(total_seq_len)
                dist.all_reduce(total_match)
                dist.all_reduce(true_positive)
                dist.all_reduce(total_positive)
                dist.all_reduce(total_true)

            report_loss = (loss_cum / total_seq_len).item()
            recall = (true_positive / total_true).item()
            precision = (true_positive / total_positive).item()
            accuracy = (total_match / total_seq_len).item()

            return report_loss, recall, precision, accuracy
        elif self.cfg.task == 'sequence_generation':
            loss_cum = torch.Tensor([0]).to(self.device)
            total_seq_len = torch.Tensor([0]).to(self.device)

            for i, d in enumerate(self.eval_dl, start=1):
                encoder_input, decoder_input, label, label_mask, _ = d['rnova']
                gnova_encoder_input, _ = d['gnova']
                meta_info_list = d['meta_info']

                with torch.no_grad():
                    if self.cfg.device == 'gpu':
                        with autocast(dtype=float16_type):
                            pred = self.model(self.model_gnova, encoder_input, gnova_encoder_input, meta_info_list, decoder_input=decoder_input)

                            pred = pred[label_mask]
                            label = label[label_mask]
                            pred_loss = self.eval_loss_fn(pred, label)
                    elif self.cfg.device == 'cpu':
                        pred = self.model(self.model_gnova, encoder_input, gnova_encoder_input, meta_info_list, decoder_input=decoder_input)

                        pred = pred[label_mask]
                        label = label[label_mask]
                        pred_loss = self.eval_loss_fn(pred, label)

                    loss_cum += pred_loss
                    total_seq_len += label_mask.sum()
            if self.distributed:
                dist.barrier()
                dist.all_reduce(loss_cum)
                dist.all_reduce(total_seq_len)

            report_loss = (loss_cum / total_seq_len).item()

            return report_loss
        else:
            raise NotImplementedError

    def inference(self):
        if self.cfg.task == 'optimal_path':
            opt_path = OptimalPathInference(self.cfg, self.device, self.test_spec_header, self.model, self.model_gnova, self.test_dl)
            opt_path.inference()
        elif self.cfg.task == 'sequence_generation':
            sequence_generation_inference(self.cfg, self.test_spec_header, self.test_dl, self.model, self.model_gnova, self.device)
        else:
            raise NotImplementedError
