import gzip
import os
import pickle

import numpy as np
import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F

from rnova.model.rnova import RNova
from rnova.task.knapsack import knapsack_mask_x2
from utils.data.BasicClass import Residual_seq, Ion


def generate_model_input(pred_seq, aa_id, input_cuda):
    if len(pred_seq) == 0:
        decoder_input_seq = torch.tensor([aa_id['<bos>']]).long()
    else:
        decoder_input_seq = torch.tensor([aa_id['<bos>']] + [aa_id[aa] for aa in pred_seq]).long()
    seq_len = len(decoder_input_seq)
    if seq_len == 1:
        step_mass = torch.tensor([0.0])
    else:
        step_mass = torch.tensor(np.insert(Residual_seq(pred_seq).step_mass, 0, 0.0))

    pos = torch.tensor(list(range(seq_len))).long()

    decoder_input = {}
    decoder_input['seq'] = decoder_input_seq.unsqueeze(0)
    decoder_input['step_mass'] = step_mass.unsqueeze(0)
    decoder_input['pos'] = pos
    input_cuda(decoder_input)

    return decoder_input

def generate_next_token_prob(model: RNova, decoder_input, encoder_input, encoder_output):
    seq = decoder_input['seq']
    tgt = model.tgt_embedding(seq)
    step_mass = decoder_input['step_mass']
    pos = decoder_input['pos']

    peak_mzs = encoder_input['moverz']

    with torch.no_grad():
        with autocast(dtype=torch.bfloat16):
            tgt = model.decoder(tgt, step_mass, pos, encoder_output, peak_mzs)
            tgt = model.decoder_output_linear(tgt)

    tgt_mask = torch.zeros_like(tgt)
    tgt_mask[:, :, :3] = float('-inf')
    tgt = tgt + tgt_mask
    return tgt

class GenerationInference():
    def __init__(self, cfg, device, graphnovo_dir, spec_header, optimum_path_result, model, aa_id):
        self.cfg = cfg
        self.beam_size = cfg.infer.beam_size
        self.dataset_dir = cfg.infer.dataset_dir
        self.device = device
        self.graphnovo_dir = graphnovo_dir
        self.spec_header = spec_header
        self.optimum_path_result = optimum_path_result
        self.aa_mass_dict = {aa: Residual_seq(aa).mass for aa in Residual_seq.output_aalist()}
        self.aa_id = aa_id
        self.id_aa = {aa_id[aa]:aa for aa in aa_id}
        self.knapsack_matrix = np.load(os.path.join(self.graphnovo_dir,cfg.serialized_model_path.split('/')[0],'knapsack/knapsack.npy'))
        self.model = model

    def read_spec_data(self, idx):
        idx = idx[0]
        if isinstance(idx, str):
            spec_head = self.spec_header.loc[idx]
        elif isinstance(idx, int):
            spec_head = self.spec_header.iloc[idx]
        else:
            raise NotImplementedError
        seq_label = spec_head['Annotated Sequence'].replace('L', 'I')

        precursor_moverz = spec_head['m/z']
        precursor_charge = spec_head['Charge']

        edge_known_list = []
        for p in self.optimum_path_result.loc[idx].pred_seq.strip().split(' '):
            try:
                edge_known_list.append(float(p))
            except ValueError as _: # p is not a float, it should be an amino acid
                edge_known_list.append(p)

        pred_path = [float(x) for x in self.optimum_path_result.loc[idx].pred_path.strip().split(' ')]
        path_mass = np.cumsum(pred_path)

        edge_known_list = edge_known_list[1:]
        path_mass = path_mass[1:]

        return seq_label, precursor_moverz, precursor_charge, edge_known_list, path_mass

    def input_cuda(self, input):
        if isinstance(input, torch.Tensor):
            input = input.to(self.device)
        else:
            for section_key in input:
                if isinstance(input[section_key], torch.Tensor):
                    input[section_key] = input[section_key].to(self.device)
                    continue
                for key in input[section_key]:
                    if isinstance(input[section_key][key], torch.Tensor):
                        input[section_key][key] = input[section_key][key].to(self.device)
        return input

    @staticmethod
    def match_AA_novor(target, predicted):
        num_match = 0
        target_len = len(target)
        predicted_len = len(predicted)
        target_mass = np.array([Residual_seq(aa).mass for aa in target])
        target_mass_cum = np.cumsum(target_mass)
        predicted_mass = np.array([Residual_seq(aa).mass for aa in predicted])
        predicted_mass_cum = np.cumsum(predicted_mass)

        i = 0
        j = 0
        while i < target_len and j < predicted_len:
            if abs(target_mass_cum[i] - predicted_mass_cum[j]) < 0.5:
                if abs(target_mass[i] - predicted_mass[j]) < 0.1:
                    num_match += 1
                i += 1
                j += 1
            elif target_mass_cum[i] < predicted_mass_cum[j]:
                i += 1
            else:
                j += 1
        return num_match, len(predicted), len(target)

    def extract_pred_prob(self, encoder_input, encoder_output, pred_seq, edge_known_list, path_mass):
        decoder_input = generate_model_input(pred_seq[:-1], self.aa_id, self.input_cuda)
        tgt = generate_next_token_prob(self.model, decoder_input, encoder_input, encoder_output)
        tgt = F.softmax(tgt, dim=-1) # shape = (1, seq_len, aa_type_num)
        pred_seq_id = [self.aa_id[aa] for aa in pred_seq]
        pred_prob = []
        edge_index = 0

        mass_threshold = self.cfg.data.ms2_threshold
        for t_idx, t in enumerate(tgt[0]):
            if isinstance(edge_known_list[edge_index], str):
                pred_prob.append(1.0)
                edge_index += 1
            else:
                pred_prob.append(t[pred_seq_id[t_idx]].item())
                if Residual_seq(pred_seq[:(t_idx+1)]).mass > path_mass[edge_index] - mass_threshold:
                    edge_index += 1
        return pred_prob

    def pick_best_pred_seq(self, score_final_complete, pred_seq_list_final_complete, precursor_charge, precursor_moverz):
        score_final_complete = torch.concat([s.unsqueeze(0) for s in score_final_complete])
        _, seq_index_sort = torch.sort(score_final_complete, descending=True)
        mass_equal_flag = False
        for seq_index in seq_index_sort:
            pred_seq = pred_seq_list_final_complete[seq_index]
            theo_mz = Ion.precursormass2ion(Residual_seq(pred_seq).mass, precursor_charge)
            if abs(theo_mz - precursor_moverz) < 10 * 1e-6 * theo_mz: # 10 ppm
                mass_equal_flag = True
                break
        if not mass_equal_flag:
            seq_index = torch.argmax(score_final_complete).item()
            pred_seq = pred_seq_list_final_complete[seq_index]

        return pred_seq

    def pick_best_k_pred_seq(self, score_complete, pred_seq_list_complete, target_mass):
        score_complete = torch.tensor(score_complete)
        _, seq_index_sort = torch.sort(score_complete, descending=True)

        new_score_complete = []
        new_pred_seq_list_complete = []
        for seq_index in seq_index_sort:
            pred_seq = pred_seq_list_complete[seq_index]
            theo = Residual_seq(pred_seq).mass
            if abs(theo - target_mass) < 1.0: #TODO(m) setting this to 0.1 will cause an error, with list being empty， 10.0 will be okay
                new_score_complete.append(score_complete[seq_index])
                new_pred_seq_list_complete.append(pred_seq)
                if len(new_score_complete) >= self.beam_size:
                    break

        return new_score_complete, new_pred_seq_list_complete

    def divide_complete_incomplete(self, \
                                   pred_seq_list, score_list, \
                                   pred_seq_list_complete, score_complete, \
                                   pred_seq_list_final_complete, score_final_complete, \
                                   target_mass, precursor_mass):
        pred_seq_list_incomplete = []
        score_incomplete = []
        for p_idx, pred_seq in enumerate(pred_seq_list):
            sub_total_mass = Residual_seq(pred_seq).mass
            if sub_total_mass + 10 > precursor_mass or len(pred_seq) >= self.cfg.data.peptide_max_len: #TODO(m) 1 instead of 10?
                pred_seq_list_final_complete.append(pred_seq)
                score_final_complete.append(score_list[p_idx])
            elif sub_total_mass + 10 > target_mass:
                pred_seq_list_complete.append(pred_seq)
                score_complete.append(score_list[p_idx])
            else:
                pred_seq_list_incomplete.append(pred_seq)
                score_incomplete.append(score_list[p_idx])

        return pred_seq_list_complete, score_complete, \
               pred_seq_list_incomplete, score_incomplete, \
               pred_seq_list_final_complete, score_final_complete

    def beam_best_k(self, pred_seq_list, tgt_list, score_list, pred_seq_list_complete, score_complete, path_mass, precursor_mass, seq_total_mass):
        """
        tgt_list: list of tgt[-, -1] of shape (aa_type_num,)
        """

        aa_type_num = tgt_list[0].shape[0]
        total_single_aa_candidate = 0

        error_tolerance = 200 + round(10 * 1e-6 * precursor_mass * 1e4)
        while total_single_aa_candidate == 0:
            score_list_extend = []
            score_list_extend_concat = []
            total_single_aa_candidate = 0
            for t_idx, tgt in enumerate(tgt_list):
                # knapsack #
                if len(pred_seq_list[t_idx]) == 0:
                    sub_total_mass = 0
                else:
                    sub_total_mass = Residual_seq(pred_seq_list[t_idx]).mass

                try:
                    node_idx = np.searchsorted(path_mass, sub_total_mass + 50) #TODO(m) 50 will cause IndexError, 10 will work
                    predict_mask, single_aa_candidate = knapsack_mask_x2(tgt, self.knapsack_matrix, \
                                                                         seq_total_mass, path_mass[node_idx]-sub_total_mass, \
                                                                         error_tolerance, 10000, self.aa_id)
                except IndexError as e:
                    raise(e)

                if len(single_aa_candidate) == 0:
                    score_sum = -float('inf') * torch.ones_like(tgt)
                else:
                    score_sum = score_list[t_idx] + F.log_softmax(tgt, dim=-1) + predict_mask
                score_list_extend.append(score_sum)
                score_list_extend_concat.append(score_sum)

                total_single_aa_candidate += len(single_aa_candidate)
            total_single_aa_candidate += len(score_complete)
            error_tolerance += 200  # this is for the exception that there is no candidate

        for s in score_complete:
            score_list_extend_concat.append(s.unsqueeze(0))

        score_list_extend_concat = torch.concat(score_list_extend_concat, dim=-1)
        beam_size = min(sum(score_list_extend_concat > float('-inf')), self.beam_size)
        beam_size = min(beam_size, total_single_aa_candidate)
        score_list_extend_concat = torch.nan_to_num(score_list_extend_concat, nan=-float('inf'))
        _, topk_index = torch.topk(score_list_extend_concat, k=beam_size)

        topk_pred_seq_list = []
        new_score_list = []
        new_pred_seq_list_complete = []
        new_score_complete = []

        for idx in topk_index:
            graph_index = int(int(idx) / aa_type_num)
            if graph_index < len(tgt_list):
                aa_index = idx % aa_type_num
                topk_pred_seq_list.append(pred_seq_list[graph_index] + self.id_aa[aa_index.item()])
                new_score_list.append(score_list_extend[graph_index][aa_index])
            else:
                score_index = idx - len(tgt_list) * aa_type_num
                new_score_complete.append(score_complete[score_index])
                new_pred_seq_list_complete.append(pred_seq_list_complete[score_index])
        return topk_pred_seq_list, new_score_list, new_pred_seq_list_complete, new_score_complete

    def inference_step(self, precursor_moverz, precursor_charge, encoder_input, encoder_output, edge_known_list, path_mass):
        seq_total_mass = sum([p for p in edge_known_list if isinstance(p, float)])

        with torch.no_grad():
            precursor_mass = Ion.precursorion2mass(precursor_moverz, precursor_charge)
            pred_seq = ''
            decoder_input = generate_model_input(pred_seq, self.aa_id, self.input_cuda)
            tgt = generate_next_token_prob(self.model, decoder_input, encoder_input, encoder_output)
            tgt_list = [tgt[0][-1]]
            pred_seq_list = [pred_seq]
            score_list = [0.0]
            pred_seq_list_complete, score_complete = [], []
            pred_seq_list_final_complete, score_final_complete = [], []
            edge_index = 0

            while True:
                # choose best k from seqs generated by beam search and the current pred_seq_complete
                # and update current score_complete(removing the low prob)
                if isinstance(edge_known_list[edge_index], str):
                    pred_seq_list_tmp = []
                    score_list_tmp = []
                    aa = edge_known_list[edge_index]
                    for p_idx, p in enumerate(pred_seq_list):
                        pred_seq_list_tmp.append(p + aa)
                        score_list_tmp.append(score_list[p_idx] + F.log_softmax(tgt_list[p_idx], dim=-1)[self.aa_id[aa]])

                    pred_seq_list = pred_seq_list_tmp
                    score_list = score_list_tmp

                    assert len(pred_seq_list_complete) == 0
                    assert len(score_complete) == 0
                else:
                    try:
                        pred_seq_list, score_list, \
                        pred_seq_list_complete, score_complete = self.beam_best_k(pred_seq_list, tgt_list, score_list, \
                                                                                  pred_seq_list_complete, score_complete, \
                                                                                  path_mass, precursor_mass, seq_total_mass)
                    except IndexError as e:
                        raise(e)

                # update all lists (moving completes from pred_seq_list into pred_seq_complete or final_complete)
                pred_seq_list_complete, score_complete, \
                pred_seq_list_incomplete, score_incomplete, \
                pred_seq_list_final_complete, score_final_complete = \
                    self.divide_complete_incomplete(pred_seq_list, score_list, \
                                                    pred_seq_list_complete, score_complete, \
                                                    pred_seq_list_final_complete, score_final_complete, \
                                                    path_mass[edge_index], precursor_mass)

                if self.beam_size <= len(score_final_complete) or len(score_incomplete) + len(score_complete) == 0:
                    pred_seq = self.pick_best_pred_seq(score_final_complete, pred_seq_list_final_complete, \
                                                       precursor_charge, precursor_moverz)
                    pred_prob = self.extract_pred_prob(encoder_input, encoder_output, pred_seq, edge_known_list, path_mass)
                    break
                elif self.beam_size <= len(score_complete) or len(score_incomplete) == 0:
                    score_list, pred_seq_list = self.pick_best_k_pred_seq(score_complete, pred_seq_list_complete, path_mass[edge_index])
                    edge_index += 1
                    pred_seq_list_complete, score_complete = [], []
                else:
                    pred_seq_list = pred_seq_list_incomplete
                    score_list = score_incomplete

                pred_seq_list_incomplete = []
                score_incomplete = []

                tgt_list = []
                for pred_seq in pred_seq_list:
                    decoder_input = generate_model_input(pred_seq, self.aa_id, self.input_cuda)
                    tgt = generate_next_token_prob(self.model, decoder_input, encoder_input, encoder_output)
                    tgt_list.append(tgt[0][-1])

        return pred_seq, pred_prob