import csv
import gzip
import os
import pickle

import torch
from hydra.utils import get_original_cwd
from torch.cuda.amp import autocast
from tqdm import tqdm

float16_type = torch.bfloat16

@torch.no_grad()
def pretrain_infer(cfg, test_dl, model):
    graphnovo_dir = get_original_cwd()

    # save result
    print('graphnovo_dir:', graphnovo_dir)
    pretrain_path = os.path.join(graphnovo_dir, cfg.infer.output_path)
    print('pretrain_path: ', pretrain_path)
    pretrain_csvfile = open(os.path.join(pretrain_path, cfg.infer.output_filename), 'w', buffering=1)
    nodeclass_fieldnames = ['graph_idx', 'MSGP File Name', 'MSGP Datablock Pointer', 'MSGP Datablock Length']
    writer = csv.DictWriter(pretrain_csvfile, fieldnames=nodeclass_fieldnames, quoting=csv.QUOTE_NONNUMERIC)
    writer.writeheader()

    file_size = 0
    file_num = 0
    msgp_writer = open(os.path.join(pretrain_path, f'{file_num}.msgp'), 'wb')

    for encoder_input, _, idx in tqdm(test_dl, total=len(test_dl)):
        if torch.is_tensor(idx): idx = idx.tolist()
        idx = idx[0]

        if file_size >= 2 * 1024 ** 3:
            msgp_writer.close()
            file_size = 0
            file_num += 1
            msgp_writer = open(os.path.join(pretrain_path, f'{file_num}.msgp'), 'wb')

        with autocast(dtype=float16_type):
            ori_embedding = model.encoder(**encoder_input)
            peak_class = model.iontype_linear(ori_embedding)
            peak_source = model.ionsource_linear(ori_embedding)

        record = {'ori_embedding': ori_embedding.cpu(), 'peak_class': peak_class.cpu(), 'peak_source': peak_source.cpu(), 'idx': idx}

        compressed_data = gzip.compress(pickle.dumps(record))
        file_size += len(compressed_data)

        writer.writerow({'graph_idx': idx, \
                         'MSGP Datablock Pointer': msgp_writer.tell(), \
                         'MSGP File Name': f'{file_num}.msgp', \
                         'MSGP Datablock Length': len(compressed_data)})
        msgp_writer.write(compressed_data)
        msgp_writer.flush()
