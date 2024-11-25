'''
@Author: Zeping Mao
@Date: 2023-11-03 20:46:57
@Last Modified by:   Zeping Mao
@Last Modified time: 2023-11-03 20:46:57
'''
import gzip
import os
import pickle

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from tqdm import tqdm

from .dataset import astral_filtering

def get_total_peak_num(ms1_num, ms2_num, precursor_charge, cfg):
    ion_types = len(cfg.data.n_term_ion) + len(cfg.data.c_term_ion)

    if precursor_charge == 2:
        return ms1_num * cfg.data.precursor_max_charge + ms2_num * ion_types + 2
    else:
        return ms1_num * cfg.data.precursor_max_charge + ms2_num * cfg.data.product_max_charge * ion_types + 2

class RNovaBucketBatchSampler(Sampler):
    """Wraps another sampler to yield a mini-batch of indices.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, cfg, dataset_dir, spec_header, bin_borders, bin_batch_size,
                 shuffle=True, drop_last=True):
        super().__init__(data_source=None)

        self.cfg = cfg
        self.dataset_dir = dataset_dir
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.spec_header = spec_header.copy()
        self.process_spec_header()

        self.epoch = 0
        self.rng = np.random.default_rng()

        self.bin_borders = bin_borders
        self.bin_batch_size = bin_batch_size

    def process_spec_header(self):
        peak_num_list = []
        for _, row in tqdm(self.spec_header.iterrows(), total=len(self.spec_header), desc='Generating Peak Number'):

            if astral_filtering:
                with open(os.path.join(self.dataset_dir, row['MSGP File Name']), 'rb') as f:
                    f.seek(row['MSGP Datablock Pointer'])
                    try:
                        spec = pickle.loads(gzip.decompress(f.read(row['MSGP Datablock Length'])))
                    except Exception as e:
                        print("Cannot read data")
                        print("file_name, ", row['MSGP File Name'])
                        raise (e)

                ms1_xgrams = spec['ms1_xgrams']
                try:
                    ms1_multiscan_indices = torch.sum(ms1_xgrams.bool(), dim=1) >= 2
                    ms1_num = torch.sum(ms1_multiscan_indices).item()
                except IndexError:
                    ms1_num = 0

                ms2_xgrams = spec['ms2_xgrams']
                ms2_multiscan_indices = torch.sum(ms2_xgrams.bool(), dim=1) >= 2
                ms2_num = torch.sum(ms2_multiscan_indices).item()

                peak_number = get_total_peak_num(ms1_num, ms2_num, row['Charge'], self.cfg)
                peak_num_list.append(peak_number)
            else:
                peak_number = get_total_peak_num(row['MS1 Peak Number'], row['MS2 Peak Number'], row['Charge'], self.cfg)
                peak_num_list.append(peak_number)

        self.spec_header['Peak Number'] = peak_num_list

        self.spec_header = self.spec_header[self.spec_header['Annotated Sequence'].apply(lambda x: len(x) < self.cfg.data.peptide_max_len)]

    def get_bin_batch_size(self, data_len: int):
        for i, border in enumerate(self.bin_borders[1:]):
            if data_len <= border:
                return self.bin_batch_size[i]

        return 0

    def __iter__(self):
        self.generate_bins()
        if self.shuffle: self.bzs_sampling()
        self.epoch += 1
        return self

    def __next__(self):
        # 如果所有的桶都是空的，那么抛出 StopIteration。
        if not self.buckets: raise StopIteration

        # 如果需要随机化，随机选择一个非空桶。
        if self.shuffle:
            bucket_lengths = [len(bucket) for bucket in self.buckets]
            weights = np.array(bucket_lengths / self.t_bzs_proportion)
            chosen_bucket_idx = self.rng.choice(len(self.buckets), p=weights / np.sum(weights))
        # 否则，按照桶的顺序选择一个非空桶。
        else:
            chosen_bucket_idx = 0

        chosen_bucket = self.buckets[chosen_bucket_idx]

        sample_index = chosen_bucket[0]
        sample_spec = self.spec_header.loc[sample_index]
        sample_data_len = sample_spec['Peak Number']
        batch_size = self.get_bin_batch_size(sample_data_len)

        batch = chosen_bucket[:batch_size]

        # 从桶中删除已使用的索引。
        self.buckets[chosen_bucket_idx] = chosen_bucket[batch_size:]
        if not self.buckets[chosen_bucket_idx]:
            del self.buckets[chosen_bucket_idx]
            if self.shuffle: self.t_bzs_proportion = np.delete(self.t_bzs_proportion, chosen_bucket_idx)
        return batch

    def __len__(self):
        return len(self.spec_header)

    def generate_bins(self):
        if self.shuffle: self.spec_header = self.spec_header.sample(frac=1, random_state=self.epoch)

        peak_num_set = set()
        self.buckets = [[] for _ in peak_num_set]
        for _, row in self.spec_header.iterrows():
            peak_number = row['Peak Number']
            peak_num_set.add(peak_number)

        self.buckets = [self.spec_header[self.spec_header['Peak Number'] == i].index.tolist() for i in peak_num_set]

        tmp_buckets = []
        max_data_len = self.bin_borders[-1]
        for bucket in self.buckets:
            sample_index = bucket[0]
            sample_spec = self.spec_header.loc[sample_index]
            sample_data_len = sample_spec['Peak Number']

            if sample_data_len > max_data_len:
                continue
            else:
                tmp_buckets.append(bucket)
        self.buckets = tmp_buckets

        if dist.is_initialized():
            world_size = dist.get_world_size()  # Get the world size for distributed training

            if self.drop_last:
                tmp_buckets = []  # This will hold the processed buckets
                for bucket in self.buckets:
                    # Check if the bucket has more items than the world size
                    if len(bucket) // world_size > 0:
                        # If the length of the bucket is not divisible by world_size, truncate the bucket
                        if len(bucket) % world_size:
                            truncated_bucket = bucket[:-(len(bucket) % world_size)]
                            tmp_buckets.append(truncated_bucket)
                        else:
                            # If it is divisible, just use the bucket as is
                            tmp_buckets.append(bucket)

                self.buckets = tmp_buckets
            self.buckets = [bucket[dist.get_rank()::dist.get_world_size()] for bucket in self.buckets]
        if self.drop_last:
            tmp_buckets = []
            for bucket in self.buckets:
                sample_index = bucket[0]
                sample_spec = self.spec_header.loc[sample_index]
                sample_data_len = sample_spec['Peak Number']
                batch_size = self.get_bin_batch_size(sample_data_len)
                assert batch_size > 0

                bucket_len = len(bucket)
                new_bucket = bucket[:-(bucket_len % batch_size)] if bucket_len % batch_size > 0 else bucket
                tmp_buckets.append(new_bucket)
            self.buckets = tmp_buckets

        self.buckets = [bucket for bucket in self.buckets if len(bucket)]

    def bzs_sampling(self):
        ret = np.zeros(len(self.buckets))

        for b_index, bucket in enumerate(self.buckets):
            sample_index = bucket[0]
            sample_spec = self.spec_header.loc[sample_index]
            sample_data_len = sample_spec['Peak Number']
            batch_size = self.get_bin_batch_size(sample_data_len)

            assert batch_size > 0

            ret[b_index] = batch_size

        self.t_bzs_proportion = ret

class RNovaSequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.
    """

    def __init__(self, cfg, dataset_dir, spec_header):
        super().__init__(data_source=None)
        self.cfg = cfg
        self.dataset_dir = dataset_dir
        self.spec_header = spec_header
        self.process_spec_header()

    def process_spec_header(self):
        peak_num_list = []
        for _, row in tqdm(self.spec_header.iterrows(), total=len(self.spec_header), desc='Generating Peak Number'):

            if astral_filtering:
                with open(os.path.join(self.dataset_dir, row['MSGP File Name']), 'rb') as f:
                    f.seek(row['MSGP Datablock Pointer'])
                    try:
                        spec = pickle.loads(gzip.decompress(f.read(row['MSGP Datablock Length'])))
                    except Exception as e:
                        print("Cannot read data")
                        print("file_name, ", row['MSGP File Name'])
                        raise (e)

                ms1_xgrams = spec['ms1_xgrams']
                try:
                    ms1_multiscan_indices = torch.sum(ms1_xgrams.bool(), dim=1) >= 2
                    ms1_num = torch.sum(ms1_multiscan_indices).item()
                except IndexError:
                    ms1_num = 0

                ms2_xgrams = spec['ms2_xgrams']
                ms2_multiscan_indices = torch.sum(ms2_xgrams.bool(), dim=1) >= 2
                ms2_num = torch.sum(ms2_multiscan_indices).item()

                peak_number = get_total_peak_num(ms1_num, ms2_num, row['Charge'], self.cfg)
                peak_num_list.append(peak_number)
            else:
                peak_number = get_total_peak_num(row['MS1 Peak Number'], row['MS2 Peak Number'], row['Charge'], self.cfg)
                peak_num_list.append(peak_number)

        self.spec_header['Peak Number'] = peak_num_list

        self.spec_header = self.spec_header[self.spec_header['Annotated Sequence'].apply(lambda x: len(x) < self.cfg.data.peptide_max_len)]

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == len(self.spec_header): raise StopIteration
        index = self.spec_header.iloc[self.idx:self.idx + 1].index
        self.idx += 1
        return index

    def __len__(self) -> int:
        return len(self.spec_header)
