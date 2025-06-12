import os
import torch
import gnova
from torch import nn
from torch import optim

from gnova.loss_func.consistency_loss import ClipLoss, noble_loss, corrected_simclr_loss
from gnova.loss_func.focal_loss import FocalLossWithLogits, MultiClassFocalLossWithLogits
from gnova.loss_func.multitask_loss import get_multitask_loss, get_multitask_loss_v2
from gnova.optimizer import Lion
from gnova.optimizer.warmup_scheduler import get_lr_scheduler
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from gnova.task.init_distributed import initialize_distributed
from gnova.task.optimal_path_inference import OptimalPathInference
from functools import partial

from gnova.task.pretrain_inference import pretrain_infer
from gnova.task.sequence_generation_inference import sequence_generation_inference

from utils.cofragment_peptide_processing import label_types

float16_type = torch.bfloat16

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
            if isinstance(cfg.infer.device, int):
                self.device = torch.device("cuda", cfg.infer.device)
                torch.cuda.set_device(cfg.infer.device)
            else:
                self.device = torch.device("cpu")
            self.distributed = False

    def initialize(self, *, train_spec_header,train_dataset_dir,val_spec_header,val_dataset_dir):
        self.model = gnova.models.GNova(self.cfg, float16_type).to(self.device)

        # iontype_focal_loss = MultiClassFocalLossWithLogits(device=self.device, alpha=0.25, gamma=2)
        # self.iontype_train_loss_fn = partial(iontype_focal_loss, reduction='mean')
        # self.iontype_eval_loss_fn = partial(iontype_focal_loss, reduction='sum')

        self.iontype_train_loss_fn = nn.CrossEntropyLoss()
        self.iontype_eval_loss_fn = nn.CrossEntropyLoss(reduction='sum')

        ionsource_loss = corrected_simclr_loss
        self.ionsource_train_loss_fn = partial(ionsource_loss, reduction='mean')
        self.ionsource_eval_loss_fn = partial(ionsource_loss, reduction='sum')

        if self.cfg.task == 'pretrain':
            pass
        elif self.cfg.task == 'peak_classification':
            # self.train_loss_fn = nn.BCEWithLogitsLoss()
            # self.eval_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

            # focal_loss = FocalLossWithLogits(gamma=1)
            # self.train_loss_fn = partial(focal_loss, reduction='mean')
            # self.eval_loss_fn = partial(focal_loss, reduction='sum')

            self.train_loss_fn = nn.CrossEntropyLoss()
            self.eval_loss_fn = nn.CrossEntropyLoss(reduction='sum')
        elif self.cfg.task == "optimal_path":
            focal_loss = FocalLossWithLogits(gamma=1)
            self.train_loss_fn = partial(focal_loss, reduction='mean')
            self.eval_loss_fn = partial(focal_loss, reduction='sum')
        elif self.cfg.task == 'sequence_generation':
            self.train_loss_fn = nn.CrossEntropyLoss(reduction='mean')
            self.eval_loss_fn = nn.CrossEntropyLoss(reduction='sum')
        else:
            raise NotImplementedError

        if self.distributed: self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        # self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.train.lr)
        self.optimizer = Lion(self.model.parameters(), lr=self.cfg.train.lr)
        self.scheduler = get_lr_scheduler(self.optimizer, self.cfg.train.warmup_steps)

        self.scaler = GradScaler()
        self.persistent_file_name = os.path.join(self.serialized_model_path,self.cfg.wandb.project+'_'+self.cfg.wandb.name+'.pt')
        if os.path.exists(self.persistent_file_name):
            checkpoint = torch.load(self.persistent_file_name,map_location={f'cuda:{0}': f'cuda:{self.local_rank}'})
            if self.distributed: self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else: self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f'load model from {self.persistent_file_name}')
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else: print('no model detected, start from scratch')

        self.train_dl = self.train_loader(train_spec_header,train_dataset_dir)
        self.eval_dl = self.eval_loader(val_spec_header,val_dataset_dir)

    def test_initialize(self, *, test_spec_header=None,test_dataset_dir=None):
        assert not self.distributed

        self.model = gnova.models.GNova(self.cfg, float16_type).to(self.device)
        self.persistent_file_name = os.path.join(self.serialized_model_path, self.cfg.wandb.project + '_' + self.cfg.wandb.name + '.pt')
        print('checkpoint: ', self.persistent_file_name)
        assert os.path.exists(self.persistent_file_name)
        if isinstance(self.cfg.infer.device, int):
            checkpoint = torch.load(self.persistent_file_name)
        else:
            checkpoint = torch.load(self.persistent_file_name, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.test_dl = self.test_loader(test_spec_header, test_dataset_dir)
        self.test_spec_header = test_spec_header

    def train_loader(self,train_spec_header,train_dataset_dir):
        ds = gnova.data.GenovaDataset(self.cfg, spec_header=train_spec_header,dataset_dir_path=train_dataset_dir)
        sampler = gnova.data.GNovaBucketBatchSampler(self.cfg,train_spec_header,self.cfg.sample.train_bin_boarders,self.cfg.sample.train_bin_batch_size,shuffle=True, drop_last=True)
        train_dl = DataLoader(ds,batch_sampler=sampler,collate_fn=gnova.data.GenovaCollator(self.cfg),pin_memory=True,num_workers=4)
        train_dl = gnova.data.DataPrefetcher(train_dl,self.device)
        return train_dl

    def eval_loader(self,val_spec_header,val_dataset_dir):
        ds = gnova.data.GenovaDataset(self.cfg, spec_header=val_spec_header,dataset_dir_path=val_dataset_dir)
        sampler = gnova.data.GNovaBucketBatchSampler(self.cfg,val_spec_header,self.cfg.sample.eval_bin_boarders,self.cfg.sample.eval_bin_batch_size,shuffle=False, drop_last=False)
        collate_fn = gnova.data.GenovaCollator(self.cfg)
        eval_dl = DataLoader(ds,batch_sampler=sampler,collate_fn=collate_fn,pin_memory=True,num_workers=4)
        eval_dl = gnova.data.DataPrefetcher(eval_dl,self.device)
        return eval_dl

    def test_loader(self,test_spec_header,test_dataset_dir):
        ds = gnova.data.GenovaDataset(self.cfg, spec_header=test_spec_header,dataset_dir_path=test_dataset_dir)
        sampler = gnova.data.GNovaSequentialSampler(test_spec_header)
        collate_fn = gnova.data.GenovaCollator(self.cfg)
        test_dl = DataLoader(ds,batch_sampler=sampler,collate_fn=collate_fn,pin_memory=True)
        if isinstance(self.cfg.infer.device, int):
            test_dl = gnova.data.DataPrefetcher(test_dl,self.device)
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
        self.model.train()
        total_step = 0

        iontype_loss_cum = 0
        ionsource_loss_cum = 0
        loss_cum = 0

        num_correct_cum = 0
        num_noise_cum = 0
        num_total_cum = 0

        num_pred_positive_cum = 0
        num_label_positive_cum = 0
        num_pos_correct_cum = 0

        true_positive = 0
        total_true = 0
        total_positive = 0
        total_seq_len = 0
        total_match = 0
        if self.cfg.task == 'pretrain':
            for epoch in range(0, self.cfg.train.total_epoch):
                self.optimizer.zero_grad(set_to_none=True)
                for i, (encoder_input, cofragment_label, _) in enumerate(self.train_dl, start=1):
                    iontype_label = cofragment_label['ion_type']
                    ionsource_label = cofragment_label['ion_source']

                    total_step += 1
                    if total_step%self.cfg.train.detect_period == 1:
                        iontype_loss_cum = 0
                        ionsource_loss_cum = 0

                        num_pred_positive_cum = 0
                        num_label_positive_cum = 0
                        num_pos_correct_cum = 0

                        num_correct_cum = 0
                        num_noise_cum = 0
                        num_total_cum = 0
                    if self.cfg.device == 'gpu':
                        with autocast(dtype=float16_type):
                            iontype_pred, ionsource_emb = self.model(encoder_input)

                            iontype_pred_flatten = iontype_pred.view(-1, len(label_types) * 2 - 1)
                            iontype_label_flatten = iontype_label.flatten().long()

                            iontype_loss = self.iontype_train_loss_fn(iontype_pred_flatten, iontype_label_flatten)
                            # ionsource_loss = self.ionsource_train_loss_fn(ionsource_emb, ionsource_label)

                            # loss = get_multitask_loss_v2(iontype_loss, ionsource_loss)
                            loss = iontype_loss #TODO(m) for single loss
                            loss = loss / self.cfg.train.gradient_accumulation_step
                    elif self.cfg.device == 'cpu':
                        iontype_pred, ionsource_emb = self.model(encoder_input)

                        iontype_pred_flatten = iontype_pred.view(-1, len(label_types) * 2 - 1)
                        iontype_label_flatten = iontype_label.flatten().long()

                        iontype_loss = self.iontype_train_loss_fn(iontype_pred_flatten, iontype_label_flatten)
                        ionsource_loss = self.ionsource_train_loss_fn(ionsource_emb, ionsource_label)

                        loss = get_multitask_loss_v2(iontype_loss, ionsource_loss)
                        loss = loss / self.cfg.train.gradient_accumulation_step

                    predicted_type = torch.argmax(iontype_pred_flatten, dim=1) #TODO(m) for all zero prediction
                    # predicted_type = torch.zeros_like(iontype_label_flatten)

                    # positive recall and precision
                    predicted_positive = (predicted_type > 0).sum().item()
                    label_positive = (iontype_label_flatten > 0).sum().item()
                    positive_correct = (predicted_type == iontype_label_flatten)[predicted_type > 0].sum().item()
                    num_pred_positive_cum += predicted_positive
                    num_label_positive_cum += label_positive
                    num_pos_correct_cum += positive_correct

                    correct_predicition = (predicted_type == iontype_label_flatten).sum().item()
                    total_prediction = len(iontype_label_flatten)
                    num_correct_cum += correct_predicition
                    num_noise_cum += (predicted_type == 0).sum().item()
                    num_total_cum += total_prediction

                    iontype_loss_cum += iontype_loss.item()
                    # ionsource_loss_cum += ionsource_loss.item()
                    ionsource_loss_cum += 0.0 #TODO(m) for single loss

                    iontype_loss = iontype_loss_cum / self.cfg.train.detect_period
                    ionsource_loss = ionsource_loss_cum / self.cfg.train.detect_period

                    iontype_accuracy = num_correct_cum / num_total_cum
                    iontype_pred_noise = num_noise_cum / num_total_cum

                    iontype_pos_recall = num_pos_correct_cum / num_label_positive_cum
                    try:
                        iontype_pos_precision = num_pos_correct_cum / num_pred_positive_cum
                    except ZeroDivisionError:
                        iontype_pos_precision = 0.0

                    if self.cfg.device == 'gpu':
                        self.scaler.scale(loss).backward()
                        if (i + 1) % self.cfg.train.gradient_accumulation_step == 0:
                            # 1. Un‐scale the gradients in the optimizer’s parameter groups
                            self.scaler.unscale_(self.optimizer)
                            # 2. Clip them (choose max_norm to taste, often 1.0)
                            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad(set_to_none=True)
                    elif self.cfg.device == 'cpu':
                        loss.backward()
                        if (i + 1) % self.cfg.train.gradient_accumulation_step == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)

                    self.scheduler.step()
                    if total_step%self.cfg.train.detect_period == 0:
                        yield iontype_loss, ionsource_loss, total_step, epoch, iontype_accuracy, iontype_pred_noise, iontype_pos_recall, iontype_pos_precision
        elif self.cfg.task == 'peak_classification':
            for epoch in range(0, self.cfg.train.total_epoch):
                for encoder_input, _, label in self.train_dl:

                    total_step += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    if total_step%self.cfg.train.detect_period == 1:
                        loss_cum = 0

                        num_pred_positive_cum = 0
                        num_label_positive_cum = 0
                        num_pos_correct_cum = 0

                        num_correct_cum = 0
                        num_noise_cum = 0
                        num_total_cum = 0

                    if self.cfg.device == 'gpu':
                        with autocast(dtype=float16_type):
                            pred = self.model(encoder_input)

                            pred_flatten = pred.view(-1, len(label_types))
                            label_flatten = label.flatten().long()

                            pred_loss = self.train_loss_fn(pred_flatten, label_flatten)

                            loss = pred_loss
                    elif self.cfg.device == 'cpu':
                        pred = self.model(encoder_input)

                        pred_flatten = pred.view(-1, len(label_types))
                        label_flatten = label.flatten().long()

                        pred_loss = self.train_loss_fn(pred_flatten, label_flatten)

                        loss = pred_loss

                    pred_flatten = pred_flatten.argmax(dim=1)
                    correct_predicition = (pred_flatten == label_flatten).sum().item()
                    total_prediction = len(label_flatten)

                    loss_cum += loss.item()
                    report_loss = loss_cum / self.cfg.train.detect_period

                    num_correct_cum += correct_predicition
                    num_noise_cum += (pred_flatten == 0).sum().item()
                    num_total_cum += total_prediction

                    # positive recall and precision
                    predicted_positive = (pred_flatten > 0).sum().item()
                    label_positive = (label_flatten > 0).sum().item()
                    positive_correct = (pred_flatten == label_flatten)[pred_flatten > 0].sum().item()
                    num_pred_positive_cum += predicted_positive
                    num_label_positive_cum += label_positive
                    num_pos_correct_cum += positive_correct

                    pos_recall = num_pos_correct_cum / num_label_positive_cum
                    try:
                        pos_precision = num_pos_correct_cum / num_pred_positive_cum
                    except ZeroDivisionError:
                        pos_precision = 0.0

                    accuracy = num_correct_cum / num_total_cum
                    pred_noise = num_noise_cum / num_total_cum

                    if self.cfg.device == 'gpu':
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    elif self.cfg.device == 'cpu':
                        loss.backward()
                        self.optimizer.step()

                    self.scheduler.step()
                    if total_step%self.cfg.train.detect_period == 0:
                        yield report_loss, accuracy, pred_noise, total_step, epoch, pos_recall, pos_precision
        elif self.cfg.task == 'optimal_path':
            for epoch in range(0, self.cfg.train.total_epoch):
                for encoder_input, cofragment_label, decoder_input, label, label_mask, _ in self.train_dl:
                    iontype_label = cofragment_label['ion_type']
                    ionsource_label = cofragment_label['ion_source']

                    total_step += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    if total_step % self.cfg.train.detect_period == 1:
                        iontype_loss_cum = 0
                        ionsource_loss_cum = 0
                        loss_cum = 0

                        true_positive = 0
                        total_true = 0
                        total_positive = 0
                        total_seq_len = 0
                        total_match = 0
                    if self.cfg.device == 'gpu':
                        with autocast(dtype=float16_type):
                            iontype_pred, ionsource_emb, pred = self.model(encoder_input, decoder_input=decoder_input)

                            iontype_loss = self.iontype_train_loss_fn(iontype_pred.view(-1, len(label_types)), iontype_label.flatten())
                            ionsource_loss = self.ionsource_train_loss_fn(ionsource_emb, ionsource_label)

                            pred = pred[label_mask].flatten()
                            label = label[label_mask].flatten()
                            pred_loss = self.train_loss_fn(pred, label)

                            loss = get_multitask_loss(iontype_loss, ionsource_loss, pred_loss)
                    elif self.cfg.device == 'cpu':
                        iontype_pred, ionsource_emb, pred = self.model(encoder_input, decoder_input=decoder_input)

                        iontype_loss = self.iontype_train_loss_fn(iontype_pred.view(-1, len(label_types)),
                                                                  iontype_label.flatten())
                        ionsource_loss = self.ionsource_train_loss_fn(ionsource_emb, ionsource_label)

                        pred = pred[label_mask].flatten()
                        label = label[label_mask].flatten()
                        pred_loss = self.train_loss_fn(pred, label)

                        loss = get_multitask_loss(iontype_loss, ionsource_loss, pred_loss)

                    pred = (pred > 0.0).bool()

                    iontype_loss_cum += iontype_loss.item()
                    ionsource_loss_cum += ionsource_loss.item()
                    loss_cum += pred_loss.item()

                    true_positive += (pred[label.bool()] > 0).sum()
                    total_true += label.sum()
                    total_positive += (pred > 0).sum()
                    total_seq_len += label_mask.sum()
                    total_match += (pred == label).sum()

                    iontype_loss = iontype_loss_cum / self.cfg.train.detect_period
                    ionsource_loss = ionsource_loss_cum / self.cfg.train.detect_period
                    report_loss = loss_cum / self.cfg.train.detect_period
                    recall = (true_positive / total_true).item()
                    precision = (true_positive / total_positive).item()
                    accuracy = (total_match / total_seq_len).item()

                    if self.cfg.device == 'gpu':
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    elif self.cfg.device == 'cpu':
                        loss.backward()
                        self.optimizer.step()

                    self.scheduler.step()
                    if total_step % self.cfg.train.detect_period == 0:
                        yield iontype_loss, ionsource_loss, report_loss, recall, precision, accuracy, total_step, epoch
        elif self.cfg.task == 'sequence_generation':
            for epoch in range(0, self.cfg.train.total_epoch):
                for encoder_input, cofragment_label, decoder_input, label, label_mask, idxes in self.train_dl:
                    iontype_label = cofragment_label['ion_type']
                    ionsource_label = cofragment_label['ion_source']

                    total_step += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    if total_step%self.cfg.train.detect_period == 1:
                        iontype_loss_cum = 0
                        ionsource_loss_cum = 0
                        loss_cum = 0
                    if self.cfg.device == 'gpu':
                        with autocast(dtype=float16_type):
                            iontype_pred, ionsource_emb, pred = self.model(encoder_input, decoder_input=decoder_input)

                            iontype_loss = self.iontype_train_loss_fn(iontype_pred.view(-1, len(label_types)), iontype_label.flatten())
                            ionsource_loss = self.ionsource_train_loss_fn(ionsource_emb, ionsource_label)

                            pred = pred[label_mask]
                            label = label[label_mask]
                            pred_loss = self.train_loss_fn(pred, label)

                            loss = get_multitask_loss(iontype_loss, ionsource_loss, pred_loss)

                    elif self.cfg.device == 'cpu':
                        iontype_pred, ionsource_emb, pred = self.model(encoder_input, decoder_input=decoder_input)

                        iontype_loss = self.iontype_train_loss_fn(iontype_pred.view(-1, len(label_types)), iontype_label.flatten())
                        ionsource_loss = self.ionsource_train_loss_fn(ionsource_emb, ionsource_label)

                        pred = pred[label_mask]
                        label = label[label_mask]
                        pred_loss = self.train_loss_fn(pred, label)

                        loss = get_multitask_loss(iontype_loss, ionsource_loss, pred_loss)

                    iontype_loss_cum += iontype_loss.item()
                    ionsource_loss_cum += ionsource_loss.item()
                    loss_cum += pred_loss.item()

                    iontype_loss = iontype_loss_cum / self.cfg.train.detect_period
                    ionsource_loss = ionsource_loss_cum / self.cfg.train.detect_period
                    report_loss = loss_cum / self.cfg.train.detect_period

                    if self.cfg.device == 'gpu':
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    elif self.cfg.device == 'cpu':
                        loss.backward()
                        self.optimizer.step()

                    self.scheduler.step()
                    if total_step%self.cfg.train.detect_period == 0:
                        yield iontype_loss, ionsource_loss, report_loss, total_step, epoch
        else:
            raise NotImplementedError

    def eval(self):
        self.model.eval()

        if self.cfg.task == 'pretrain':
            iontype_loss_cum = torch.Tensor([0]).to(self.device)
            ionsource_loss_cum = torch.Tensor([0]).to(self.device)

            num_correct_cum = torch.Tensor([0]).to(self.device)
            num_noise_cum = torch.Tensor([0]).to(self.device)

            num_pred_positive_cum = torch.Tensor([0]).to(self.device)
            num_label_positive_cum = torch.Tensor([0]).to(self.device)
            num_pos_correct_cum = torch.Tensor([0]).to(self.device)

            total_wseq_len = torch.Tensor([0]).to(self.device)
            total_wseq2_len = torch.Tensor([0]).to(self.device)

            for encoder_input, cofragment_label, _ in self.eval_dl:
                iontype_label = cofragment_label['ion_type']
                ionsource_label = cofragment_label['ion_source']

                with torch.no_grad():
                    if self.cfg.device == 'gpu':
                        with autocast(dtype=float16_type):
                            iontype_pred, ionsource_emb = self.model(encoder_input)

                            iontype_pred_flatten = iontype_pred.view(-1, len(label_types) * 2 - 1)
                            iontype_label_flatten = iontype_label.flatten().long()

                            iontype_loss = self.iontype_eval_loss_fn(iontype_pred_flatten, iontype_label_flatten)
                            # ionsource_loss = self.ionsource_eval_loss_fn(ionsource_emb, ionsource_label) #TODO(m) for single loss
                    elif self.cfg.device == 'cpu':
                        iontype_pred, ionsource_emb = self.model(encoder_input)

                        iontype_pred_flatten = iontype_pred.view(-1, len(label_types) * 2 - 1)
                        iontype_label_flatten = iontype_label.flatten().long()

                        iontype_loss = self.iontype_eval_loss_fn(iontype_pred_flatten, iontype_label_flatten)
                        ionsource_loss = self.ionsource_eval_loss_fn(ionsource_emb, ionsource_label)

                    predicted_type = torch.argmax(iontype_pred_flatten, dim=1) #TODO(m) for all zero prediction
                    # predicted_type = torch.zeros_like(iontype_label_flatten)
                    correct_predicition = (predicted_type == iontype_label_flatten).sum().item()
                    num_correct_cum += correct_predicition
                    num_noise_cum += (predicted_type == 0).sum().item()

                    # positive recall and precision
                    predicted_positive = (predicted_type > 0).sum().item()
                    label_positive = (iontype_label_flatten > 0).sum().item()
                    positive_correct = (predicted_type == iontype_label_flatten)[predicted_type > 0].sum().item()
                    num_pred_positive_cum += predicted_positive
                    num_label_positive_cum += label_positive
                    num_pos_correct_cum += positive_correct

                    iontype_loss_cum += iontype_loss
                    # ionsource_loss_cum += ionsource_loss  #TODO(m) for single loss
                    ionsource_loss_cum += 0.0

                    total_wseq_len += iontype_label.shape[0] * iontype_label.shape[1]
                    total_wseq2_len += iontype_label.shape[0] * iontype_label.shape[1] ** 2
            if self.distributed:
                dist.barrier()
                dist.all_reduce(iontype_loss_cum)
                dist.all_reduce(ionsource_loss_cum)
                dist.all_reduce(num_correct_cum)
                dist.all_reduce(num_noise_cum)

                dist.all_reduce(num_pred_positive_cum)
                dist.all_reduce(num_label_positive_cum)
                dist.all_reduce(num_pos_correct_cum)

                dist.all_reduce(total_wseq_len)
                dist.all_reduce(total_wseq2_len)

            iontype_loss = (iontype_loss_cum / total_wseq_len).item()
            ionsource_loss = (ionsource_loss_cum / total_wseq_len).item()

            iontype_accuracy = (num_correct_cum / total_wseq_len).item()
            iontype_pred_noise = (num_noise_cum / total_wseq_len).item()

            iontype_pos_recall = (num_pos_correct_cum / num_label_positive_cum).item()
            iontype_pos_precision = (num_pos_correct_cum / num_pred_positive_cum).item()
            if iontype_pos_precision == float('inf'):
                iontype_pos_precision = 0.0

            return iontype_loss, ionsource_loss, iontype_accuracy, iontype_pred_noise, iontype_pos_recall, iontype_pos_precision
        elif self.cfg.task == 'peak_classification':
            loss_cum = torch.Tensor([0]).to(self.device)

            num_correct_cum = torch.Tensor([0]).to(self.device)
            num_noise_cum = torch.Tensor([0]).to(self.device)

            num_pred_positive_cum = torch.Tensor([0]).to(self.device)
            num_label_positive_cum = torch.Tensor([0]).to(self.device)
            num_pos_correct_cum = torch.Tensor([0]).to(self.device)

            total_seq_len = torch.Tensor([0]).to(self.device)

            for encoder_input, _, label in self.eval_dl:

                with torch.no_grad():
                    if self.cfg.device == 'gpu':
                        with autocast(dtype=float16_type):
                            pred = self.model(encoder_input)

                            pred_flatten = pred.view(-1, len(label_types))
                            label_flatten = label.flatten().long()

                            pred_loss = self.eval_loss_fn(pred_flatten, label_flatten)
                    elif self.cfg.device == 'cpu':
                        pred = self.model(encoder_input)

                        pred_flatten = pred.view(-1, len(label_types))
                        label_flatten = label.flatten().long()

                        pred_loss = self.eval_loss_fn(pred_flatten, label_flatten)

                    pred_flatten = pred_flatten.argmax(dim=1)
                    correct_predicition = (pred_flatten == label_flatten).sum().item()
                    total_prediction = len(label_flatten)

                    loss_cum += pred_loss

                    num_correct_cum += correct_predicition
                    num_noise_cum += (pred_flatten == 0).sum().item()
                    total_seq_len += total_prediction

                    # positive recall and precision
                    predicted_positive = (pred_flatten > 0).sum().item()
                    label_positive = (label_flatten > 0).sum().item()
                    positive_correct = (pred_flatten == label_flatten)[pred_flatten > 0].sum().item()
                    num_pred_positive_cum += predicted_positive
                    num_label_positive_cum += label_positive
                    num_pos_correct_cum += positive_correct


            if self.distributed:
                dist.barrier()
                dist.all_reduce(loss_cum)
                dist.all_reduce(total_seq_len)
                dist.all_reduce(num_correct_cum)
                dist.all_reduce(num_noise_cum)

                dist.all_reduce(num_pred_positive_cum)
                dist.all_reduce(num_label_positive_cum)
                dist.all_reduce(num_pos_correct_cum)

            report_loss = (loss_cum / total_seq_len).item()
            accuracy = (num_correct_cum / total_seq_len).item()
            pred_noise = (num_noise_cum / total_seq_len).item()

            pos_recall = (num_pos_correct_cum / num_label_positive_cum).item()
            pos_precision = (num_pos_correct_cum / num_pred_positive_cum).item()
            if pos_precision == float('inf'):
                pos_precision = 0.0

            return report_loss, accuracy, pred_noise, pos_recall, pos_precision
        elif self.cfg.task == 'optimal_path':
            iontype_loss_cum = torch.Tensor([0]).to(self.device)
            ionsource_loss_cum = torch.Tensor([0]).to(self.device)
            loss_cum = torch.Tensor([0]).to(self.device)
            total_seq_len = torch.Tensor([0]).to(self.device)
            total_wseq_len = torch.Tensor([0]).to(self.device)
            total_wseq2_len = torch.Tensor([0]).to(self.device)

            total_match = torch.Tensor([0]).to(self.device)
            true_positive = torch.Tensor([0]).to(self.device)
            total_positive = torch.Tensor([0]).to(self.device)
            total_true = torch.Tensor([0]).to(self.device)

            for encoder_input, cofragment_label, decoder_input, label, label_mask, _ in self.eval_dl:
                iontype_label = cofragment_label['ion_type']
                ionsource_label = cofragment_label['ion_source']

                with torch.no_grad():
                    if self.cfg.device == 'gpu':
                        with autocast(dtype=float16_type):
                            iontype_pred, ionsource_emb, pred = self.model(encoder_input, decoder_input=decoder_input)

                            iontype_loss = self.iontype_eval_loss_fn(iontype_pred.view(-1, len(label_types)), iontype_label.flatten())
                            ionsource_loss = self.ionsource_eval_loss_fn(ionsource_emb, ionsource_label)

                            pred = pred[label_mask].flatten()
                            label = label[label_mask].flatten()
                            pred_loss = self.eval_loss_fn(pred, label)
                    elif self.cfg.device == 'cpu':
                        iontype_pred, ionsource_emb, pred = self.model(encoder_input, decoder_input=decoder_input)

                        iontype_loss = self.iontype_eval_loss_fn(iontype_pred.view(-1, len(label_types)), iontype_label.flatten())
                        ionsource_loss = self.ionsource_eval_loss_fn(ionsource_emb, ionsource_label)

                        pred = pred[label_mask].flatten()
                        label = label[label_mask].flatten()
                        pred_loss = self.eval_loss_fn(pred, label)

                    pred = (pred > 0.0).bool()
                    iontype_loss_cum += iontype_loss
                    ionsource_loss_cum += ionsource_loss
                    loss_cum += pred_loss

                    true_positive += (pred[label.bool()] > 0).sum()
                    total_true += label.sum()
                    total_positive += (pred > 0).sum()
                    total_seq_len += label_mask.sum()
                    total_wseq_len += iontype_label.shape[1]
                    total_wseq2_len += iontype_label.shape[1] ** 2
                    total_match += (pred == label).sum()
            if self.distributed:
                dist.barrier()
                dist.all_reduce(iontype_loss_cum)
                dist.all_reduce(ionsource_loss_cum)
                dist.all_reduce(loss_cum)
                dist.all_reduce(total_seq_len)
                dist.all_reduce(total_wseq_len)
                dist.all_reduce(total_wseq2_len)
                dist.all_reduce(total_match)
                dist.all_reduce(true_positive)
                dist.all_reduce(total_positive)
                dist.all_reduce(total_true)

            iontype_loss = (iontype_loss_cum / total_wseq_len).item()
            ionsource_loss = (ionsource_loss_cum / total_wseq2_len).item()
            report_loss = (loss_cum / total_seq_len).item()
            recall = (true_positive / total_true).item()
            precision = (true_positive / total_positive).item()
            accuracy = (total_match / total_seq_len).item()

            return iontype_loss, ionsource_loss, report_loss, recall, precision, accuracy
        elif self.cfg.task == 'sequence_generation':
            iontype_loss_cum = torch.Tensor([0]).to(self.device)
            ionsource_loss_cum = torch.Tensor([0]).to(self.device)
            loss_cum = torch.Tensor([0]).to(self.device)
            total_seq_len = torch.Tensor([0]).to(self.device)
            total_wseq_len = torch.Tensor([0]).to(self.device)
            total_wseq2_len = torch.Tensor([0]).to(self.device)

            for encoder_input, cofragment_label, decoder_input, label, label_mask, _ in self.eval_dl:
                iontype_label = cofragment_label['ion_type']
                ionsource_label = cofragment_label['ion_source']

                with torch.no_grad():
                    if self.cfg.device == 'gpu':
                        with autocast(dtype=float16_type):
                            iontype_pred, ionsource_emb, pred = self.model(encoder_input, decoder_input=decoder_input)

                            iontype_loss = self.iontype_eval_loss_fn(iontype_pred.view(-1, len(label_types)),
                                                                     iontype_label.flatten())
                            ionsource_loss = self.ionsource_eval_loss_fn(ionsource_emb, ionsource_label)

                            pred = pred[label_mask]
                            label = label[label_mask]
                            pred_loss = self.eval_loss_fn(pred, label)
                    elif self.cfg.device == 'cpu':
                        iontype_pred, ionsource_emb, pred = self.model(encoder_input, decoder_input=decoder_input)

                        iontype_loss = self.iontype_eval_loss_fn(iontype_pred.view(-1, len(label_types)),
                                                                 iontype_label.flatten())
                        ionsource_loss = self.ionsource_eval_loss_fn(ionsource_emb, ionsource_label)

                        pred = pred[label_mask]
                        label = label[label_mask]
                        pred_loss = self.eval_loss_fn(pred, label)

                    iontype_loss_cum += iontype_loss
                    ionsource_loss_cum += ionsource_loss
                    loss_cum += pred_loss
                    total_seq_len += label_mask.sum()
                    total_wseq_len += iontype_label.shape[0] * iontype_label.shape[1]
                    total_wseq2_len += iontype_label.shape[0] * iontype_label.shape[1] ** 2

            if self.distributed:
                dist.barrier()
                dist.all_reduce(iontype_loss_cum)
                dist.all_reduce(ionsource_loss_cum)
                dist.all_reduce(loss_cum)
                dist.all_reduce(total_seq_len)
                dist.all_reduce(total_wseq_len)
                dist.all_reduce(total_wseq2_len)

            iontype_loss = (iontype_loss_cum / total_wseq_len).item()
            ionsource_loss = (ionsource_loss_cum / total_wseq2_len).item()
            report_loss = (loss_cum / total_seq_len).item()

            return iontype_loss, ionsource_loss, report_loss
        else:
            raise NotImplementedError

    def inference(self):
        if self.cfg.task == 'pretrain':
            pretrain_infer(self.cfg, self.test_dl, self.model)
        elif self.cfg.task == 'optimal_path':
            opt_path = OptimalPathInference(self.cfg, self.device, self.test_spec_header, self.model, self.test_dl)
            opt_path.inference()
        elif self.cfg.task == 'sequence_generation':
            sequence_generation_inference(self.cfg, self.test_spec_header, self.test_dl, self.model, self.device)
        else:
            raise NotImplementedError
