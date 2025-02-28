import csv
import os

import pandas as pd
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.cuda.amp import autocast
from tqdm import tqdm

from .sequence_generation_inference_utils import GenerationInference
from utils.data.BasicClass import Residual_seq


def sequence_generation_inference(cfg: DictConfig, spec_header, test_dl, model, model_gnova, device):
    graphnovo_dir = get_original_cwd()
    optimal_path_result = pd.read_csv(os.path.join(graphnovo_dir, cfg.infer.optimal_path_file), index_col="graph_idx")
    optimal_path_result = optimal_path_result.drop(["label_path"], axis=1)

    # dictionary
    aa_dict = {aa: i for i, aa in enumerate(Residual_seq.output_aalist(), start=3)}
    aa_dict['<pad>'] = 0
    aa_dict['<bos>'] = 1
    aa_dict['<eos>'] = 2

    # save result
    print('graphnovo_dir:', graphnovo_dir)
    filename = os.path.join(graphnovo_dir, cfg.infer.output_file)
    print("output file: ", filename)
    csvfile = open(filename, 'w', buffering=1)
    fieldnames = ['graph_idx', 'pred_seq', 'pred_prob', 'pred_path', 'label_seq']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
    writer.writeheader()

    aa_matched_num_total, aa_predict_len_total, aa_label_len_total = 0, 0, 0
    peptide_matched_num = 0
    peptide_predict_num = 0
    gen_infer = GenerationInference(cfg, device, graphnovo_dir, spec_header, optimal_path_result, model, aa_dict)

    for d in tqdm(test_dl, total=len(test_dl)):
        encoder_input, decoder_input, label, label_mask, idx = d['rnova']
        gnova_encoder_input_list, _ = d['gnova']
        meta_info_list = d['meta_info']

        if torch.is_tensor(idx): idx = idx.tolist()

        ######
        # Filter large peptides
        spec_head = spec_header.loc[idx[0]]

        total_peak_num = spec_head['Peak Number']
        if total_peak_num > 80_000:
            continue
        ######

        try:
            seq_label, precursor_moverz, precursor_charge, edge_known_list, path_mass = gen_infer.read_spec_data(idx)
        except KeyError as e:
            continue
        seq_label_sep = seq_label

        with torch.no_grad():
            with autocast(dtype=torch.bfloat16):
                gnova_output_list = [model_gnova.encoder(**gnova_encoder_input_list[0])]
                encoder_input = gen_infer.input_cuda(encoder_input)
                encoder_output = model.encoder(**encoder_input, meta_info_list=meta_info_list, gnova_encoder_output_list=gnova_output_list)

        pred_seq, pred_prob = gen_infer.inference_step(precursor_moverz, precursor_charge, \
                                                       encoder_input, encoder_output, \
                                                       edge_known_list, path_mass)

        writer.writerow({'graph_idx': idx[0], 'pred_seq': pred_seq, 'pred_prob': pred_prob, \
                         'pred_path': path_mass.tolist(), 'label_seq': seq_label_sep})

        aa_matched_num, aa_predict_len, aa_label_len = \
            gen_infer.match_AA_novor(seq_label.replace(' ', ''), pred_seq)
        if aa_matched_num == aa_predict_len and aa_predict_len == aa_label_len:
            peptide_matched_num += 1
        aa_matched_num_total += aa_matched_num
        aa_predict_len_total += aa_predict_len
        aa_label_len_total += aa_label_len
        peptide_predict_num += 1

        #####
        # Limit the total number of peptide
        if peptide_predict_num >= 20_000:
            break
        #####

    print('aa_matched_num_total:', aa_matched_num_total)
    print('aa_predict_len_total: ', aa_predict_len_total)
    print('aa_label_len_total: ', aa_label_len_total)
    print('aa precision: ', aa_matched_num_total / aa_predict_len_total)
    print('aa recall: ', aa_matched_num_total / aa_label_len_total)
    print('peptide recall: ', peptide_matched_num / peptide_predict_num)
