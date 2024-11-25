import csv
import gzip
import os
import pickle
from itertools import combinations_with_replacement

import numpy as np
import torch
from hydra.utils import get_original_cwd
from torch.cuda.amp import autocast
from tqdm import tqdm

from rnova.task.infer_utils import aa_datablock_dict_generate, input_cuda
from rnova.task.knapsack import knapsack_search
from utils.data.BasicClass import Residual_seq, Ion

float16_type = torch.bfloat16

class OptimalPathInference:
    def __init__(self, cfg, device, spec_header, model, test_dl):
        self.cfg = cfg
        self.device = device
        self.spec_header = spec_header
        self.model = model
        self.test_dl = test_dl

        aa_dict = {aa: i for i, aa in enumerate(Residual_seq.output_aadict(), start=3)}
        aa_dict['<pad>'] = 0
        aa_dict['<bos>'] = 1
        aa_dict['<eos>'] = 2
        self.aa_dict = aa_dict

        self.graphnovo_dir = get_original_cwd()
        self.knapsack_matrix = np.load(os.path.join(self.graphnovo_dir, cfg.serialized_model_path.split('/')[0], 'knapsack/knapsack.npy'))

        all_possible_mass = []
        aalist = Residual_seq.output_aalist()
        for num in range(1, 7):
            for i in combinations_with_replacement(aalist, num):
                all_possible_mass.append(Residual_seq(i).mass)
        all_possible_mass = np.unique(np.array(all_possible_mass))
        self.all_possible_mass = all_possible_mass

        self.aa_datablock_dict, self.aa_datablock_dict_reverse = aa_datablock_dict_generate()
        self.aa_datablock = np.array(sorted(list(self.aa_datablock_dict.values())))

    def generate_decoder_input(self, graph_probability_input, encoder_input):
        # graph_probability_input has shape (batch_size, seq_len, peak_num)
        decoder_input = {}
        decoder_input['graph_probs'] = graph_probability_input

        graph_probability_input_list = [x.unsqueeze(0) for x in graph_probability_input]

        mass_tag_list = []
        for i, graph_probability_input in enumerate(graph_probability_input_list):
            current_path = self.extract_path_from_graphprob_input(graph_probability_input, encoder_input)
            mass_tag_list.append(current_path)
        mass_tag_list = torch.stack(mass_tag_list) # (batch_size, seq_len)
        step_mass = torch.cumsum(mass_tag_list, dim=1)
        decoder_input['step_mass'] = step_mass

        seq_len = graph_probability_input.shape[1]
        pos = torch.arange(seq_len)

        decoder_input['pos'] = pos
        decoder_input = input_cuda(decoder_input, self.device)

        return decoder_input

    def extract_path_from_graphprob_input(self, graph_probability_input, encoder_input):
        # graph_probability_input has shape (1, seq_len, peak_num)
        graph_probability_input = graph_probability_input[0,1:,:]
        seq_len, peak_num = graph_probability_input.shape

        moverz = encoder_input['moverz'][0]

        masses = [0.0]
        cum_masses = [0.0]
        for j in range(seq_len):
            graph_label = graph_probability_input[j, :]
            current_mass = torch.mean(moverz[graph_label == 1]).item()
            mass_tag = current_mass - cum_masses[-1]

            masses.append(mass_tag)
            cum_masses.append(current_mass)

        return torch.Tensor(masses)  # shape (seq_len,)

    def extract_totmass_from_graphprob_input(self, graph_probability_input, encoder_input):
        # graph_probability_input has shape (1, seq_len, peak_num)
        seq_len = graph_probability_input.shape[1]
        last_graph_label = graph_probability_input[0,-1,:]
        moverz = encoder_input['moverz'][0]

        current_mass = torch.mean(moverz[last_graph_label == 1]).item()

        if current_mass > 0.0 or (current_mass == 0.0 and seq_len == 1):
            return current_mass
        else:
            raise ValueError("last_graph_label should contain at least one non-zero entry")

    def generate_graph_probability(self, decoder_input, encoder_input, encoder_output):
        with autocast(dtype=float16_type):
            graph_probs = decoder_input['graph_probs']
            step_mass = decoder_input['step_mass']
            pos = decoder_input['pos']

            batch_size, seq_len, peak_num = graph_probs.shape
            query_node = graph_probs @ encoder_output
            moverz = encoder_input['moverz']

            query_node = self.model.decoder(query_node, step_mass, pos, encoder_output, moverz)
            query_node = self.model.query_node_linear(query_node)
            graph_node = self.model.graph_node_linear(encoder_output).transpose(1, 2)

            graph_probability = query_node @ graph_node # shape (batch_size, seq_len, peak_num)

            ms1_ms2_flag = encoder_input['ms1_ms2_flag'][0]
            ms1_ms2_flag = ms1_ms2_flag.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1)
            graph_probability[ms1_ms2_flag==0] = float('-inf')

            graph_probability = graph_probability.softmax(dim=-1) # shape (batch_size, seq_len, peak_num)
        return graph_probability

    def extract_pred_prob(self, graph_probability_input, encoder_input, encoder_output):
        # graph_probability_input has shape (1, seq_len, peak_num)
        graph_probability_label = graph_probability_input[:,1:,:]
        graph_probability_input = graph_probability_input[:,:-1,:]

        decoder_input = self.generate_decoder_input(graph_probability_input, encoder_input)
        graph_probability = self.generate_graph_probability(decoder_input, encoder_input, encoder_output)

        graph_probability = graph_probability * graph_probability_label
        graph_probability = graph_probability.cpu().float().numpy() # shape (1, seq_len, peak_num)

        graph_probability = graph_probability[0,:,:]
        seq_len, peak_num = graph_probability.shape
        mass_scores = [1.0]
        for j in range(seq_len):
            scores = graph_probability[j, :]
            mass_prob = self.get_mass_prob(scores)
            mass_scores.append(mass_prob)

        mass_scores = torch.Tensor(mass_scores) # seq_len
        return mass_scores

    @staticmethod
    def get_mass_prob(peak_probs):
        summation = np.sum(peak_probs)
        summation_square = np.sum([x ** 2 for x in peak_probs])
        return summation_square / (summation + 1.0)

    def is_knapsack_possible(self, suffix_mass: float):
        aa_resolution = 10000
        mass_precision_tolerance = 650 # This is equivalent to 0.065 Da
        knapsack_candidate = knapsack_search(self.knapsack_matrix, suffix_mass, mass_precision_tolerance, aa_resolution, self.aa_dict)
        return len(knapsack_candidate) > 0

    def vote_on_next_graphnode(self, graph_probability, encoder_input, current_mass, precursor_mass):
        # graph_probability has shape [1, seq_len, peak_num]
        graph_probability = graph_probability[0, -1, :] # shape [peak_num]
        ms1_ms2_flag = encoder_input['ms1_ms2_flag'][0] # shape [peak_num]
        moverz = encoder_input['moverz'][0] # shape [peak_num]

        mass_threshold = self.cfg.data.ms2_threshold * 2
        peak_num = len(moverz)

        mass_scores = {}
        mass_alts = {}
        for i in range(peak_num):
            if ms1_ms2_flag[i] == 1:
                new_mass = moverz[i].item()
                if new_mass > current_mass:
                    prob = graph_probability[i].item()

                    mass_tag = new_mass - current_mass
                    suffix_mass = precursor_mass - new_mass

                    if suffix_mass - mass_threshold <= 0.0 <= suffix_mass + mass_threshold:
                        is_possible = True
                    else:
                        is_possible = self.all_possible_mass.searchsorted(mass_tag - mass_threshold) != self.all_possible_mass.searchsorted(mass_tag + mass_threshold)
                        is_possible = is_possible or mass_tag > self.all_possible_mass[-1]
                        is_possible = is_possible and self.is_knapsack_possible(suffix_mass)

                    if is_possible:
                        exist = False
                        for mass in mass_scores.keys():
                            if abs(mass - mass_tag) < mass_threshold:
                                exist = True
                                mass_scores[mass].append(prob)
                                mass_alts[mass].append(mass_tag)
                        if not exist:
                            mass_scores[mass_tag] = [prob]
                            mass_alts[mass_tag] = [mass_tag]

        new_mass_scores = {}
        for mass, probs in mass_scores.items():
            masses = mass_alts[mass]
            avg_mass = np.mean(masses)
            new_mass_scores[avg_mass] = probs

        ret = {}
        for mass_tag in new_mass_scores.keys():
            ret[mass_tag] = self.get_mass_prob(new_mass_scores[mass_tag])

        return ret

    def generate_next_graph_probability_input(self, graph_probability_input, current_mass, precursor_mass, encoder_input):
        mass_threshold = self.cfg.data.ms2_threshold
        moverz = encoder_input['moverz'][0] # shape [peak_num]
        num_peaks = len(moverz)
        ms1_ms2_flag = encoder_input['ms1_ms2_flag'][0] # shape [peak_num]

        moverz_copy = moverz.detach().clone()
        moverz_copy[ms1_ms2_flag == 0] = float('-inf')

        ret = torch.zeros(num_peaks)
        if current_mass - 2 * mass_threshold <= precursor_mass <= current_mass + 2 * mass_threshold:
            ret[-1] = 1.0
        else:
            lower_idx = torch.searchsorted(moverz_copy, current_mass - mass_threshold)
            upper_idx = torch.searchsorted(moverz_copy, current_mass + mass_threshold)
            ret[lower_idx:upper_idx] = 1.0

        ret = ret.unsqueeze(0).unsqueeze(0)  # (1, 1, peak_num)
        ret = input_cuda(ret, self.device)
        ret = torch.cat([graph_probability_input, ret], dim=1) # (1, seq_len+1, peak_num)
        return ret


    def is_graph_probability_input_complete(self, graph_probability_input: torch.Tensor):
        # graph_probability_input has shape (1, seq_len, peak_num)
        last_labels = graph_probability_input[0, -1, :]
        if last_labels[-1].item() == 1:
            return True
        else:
            return False

    def format_seq_predict(self, pred_path):
        seq_predict = []
        for edge_mass in pred_path:
            mass_threshold = self.cfg.data.ms2_threshold

            # mass_diff-mass_threshold <= aa_l <= edge_mass+mass_threshold <= aa_r
            l = self.aa_datablock.searchsorted(edge_mass - mass_threshold, side='left')
            r = self.aa_datablock.searchsorted(edge_mass + mass_threshold, side='left')
            aa_values = [self.aa_datablock[idx] for idx in range(l, r)]

            # Note: here may have some states with no aa_block existing
            aa_block = [''.join(self.aa_datablock_dict_reverse[aa_value]) for aa_value in aa_values]
            if len(aa_block) == 0 or len(aa_block) > 1 or (len(aa_block) == 1 and len(aa_block[0]) > 1):
                seq_predict.append(str(edge_mass))
            elif len(aa_block) == 1 and len(aa_block[0]) == 1:
                seq_predict.append(aa_block[0])
                assert edge_mass > Residual_seq(aa_block[0]).mass - mass_threshold
                assert edge_mass < Residual_seq(aa_block[0]).mass + mass_threshold
                assert aa_block[0] not in ['N', 'Q']  # because mass of NQ can be combination of other multiple aas

        seq_predict = ' '.join(seq_predict)

        return seq_predict

    def path_evaluation(self, graph_probability_input, label, label_mask, encoder_input):
        # graph_probability_input has shape (batch_size, seq_len, peak_num)
        # label should have same shape
        # label_mask should have shape (batch_size, seq_len)
        """ For single sample """

        pred_path = self.extract_path_from_graphprob_input(graph_probability_input, encoder_input)
        pred_len = len(pred_path)

        label_peakprob_input = label
        peak_num = label_peakprob_input.shape[2]
        label_peakprob_input = torch.cat([input_cuda(torch.zeros(1, 1, peak_num), self.device), label_peakprob_input], dim=1) # the first row of label_peakprob_input does not matter anyway
        label_path = self.extract_path_from_graphprob_input(label_peakprob_input, encoder_input)
        label_len = len(label_path)

        target_mass_cum = np.cumsum(label_path)
        predicted_mass_cum = np.cumsum(pred_path)
        num_match = 0
        i = 0
        j = 0
        while i < label_len and j < pred_len:
            if abs(target_mass_cum[i] - predicted_mass_cum[j]) < 0.5:
                if abs(label_path[i] - pred_path[j]) < 0.1:
                    num_match += 1
                i += 1
                j += 1
            elif target_mass_cum[i] < predicted_mass_cum[j]:
                i += 1
            else:
                j += 1

        return num_match, pred_len, label_len, pred_path, label_path

    def beam_best_k(self, graph_probability_input_list, score_list, graph_probability_input_complete, score_complete,
                    graph_probability_list, encoder_input, precursor_mass, beam_size, sum_flag='multiply'):
        _, path_len, peak_num = graph_probability_input_list[0].shape

        score_list_extend = []
        current_mass_extend = []
        graph_probability_input_list_extend = []
        for i, graph_probability in enumerate(graph_probability_list):
            graph_probability_input = graph_probability_input_list[i]
            current_mass = self.extract_totmass_from_graphprob_input(graph_probability_input, encoder_input)
            current_score = score_list[i]
            vote_result = self.vote_on_next_graphnode(graph_probability, encoder_input, current_mass, precursor_mass)

            for mass_tag in vote_result.keys():
                mass_score = vote_result[mass_tag]

                if sum_flag == 'sum':
                    score_list_extend.append(current_score + mass_score)
                elif sum_flag == 'multiply':
                    score_list_extend.append(current_score * mass_score)

                # graph_probability_input_list_extend.append(self.generate_next_graph_probability_input(graph_probability_input, current_mass + mass_tag, precursor_mass, encoder_input))
                graph_probability_input_list_extend.append(graph_probability_input)
                current_mass_extend.append(current_mass + mass_tag)

        for s in score_complete:
            score_list_extend.append(s) # s should be a single float, representing the score of the complete path

        score_list_extend = torch.Tensor(score_list_extend)
        beam_size = min(beam_size, len(score_list_extend))
        _, topk_index = torch.topk(score_list_extend, k=beam_size)

        topk_graph_probability_input = []
        new_graph_probability_input_complete = []
        new_score_list = []
        new_score_complete = []
        for idx in topk_index:
            idx = idx.item()
            try:
                graph_probability_input = graph_probability_input_list_extend[idx]
                current_mass = current_mass_extend[idx]
                next_graph_probability_input = self.generate_next_graph_probability_input(graph_probability_input, current_mass, precursor_mass, encoder_input)
                topk_graph_probability_input.append(next_graph_probability_input)
                new_score_list.append(score_list_extend[idx])
            except IndexError as _:
                assert idx >= len(graph_probability_input_list_extend)
                complete_index = idx - len(graph_probability_input_list_extend)
                new_score_complete.append(score_complete[complete_index])
                new_graph_probability_input_complete.append(graph_probability_input_complete[complete_index])

        return topk_graph_probability_input, new_score_list, new_graph_probability_input_complete, new_score_complete

    def inference(self):
        print('graphnovo_dir:', self.graphnovo_dir)
        path_file_name = os.path.join(self.graphnovo_dir, self.cfg.infer.optimal_path_file)
        print('path_file_name:', path_file_name)
        path_csvfile = open(path_file_name, 'w', buffering=1)
        path_fieldnames = ['graph_idx', 'pred_path', 'pred_prob', 'label_path', 'pred_seq']
        writer_path = csv.DictWriter(path_csvfile, fieldnames=path_fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer_path.writeheader()

        # metrics initialization
        matched_num_total, predict_len_total, label_len_total, whole_seq_match, peptide_predict_num = 0, 0, 0, 0, 0
        # main
        sum_flag = 'multiply'
        for encoder_input, _, _, label, label_mask, idx in tqdm(self.test_dl, total=len(self.test_dl)):
            if torch.is_tensor(idx): idx = idx.tolist()

            spec_head = self.spec_header.loc[idx[0]]

            ######
            # Filter large peptides
            total_peak_num = spec_head['Peak Number']
            if total_peak_num > 60000:
                continue
            ######

            precursor_charge = int(spec_head['Charge'])
            precursor_mz = float(spec_head['m/z'])
            precursor_mass = Ion.precursorion2mass(precursor_mz, precursor_charge)

            with torch.no_grad():
                encoder_input = input_cuda(encoder_input, self.device)
                with autocast(dtype=float16_type):
                    encoder_output = self.model.encoder(**encoder_input)
                    # shape is (1, seq_len, hidden_size)

                moverz = encoder_input['moverz'][0]
                peak_num = len(moverz)

                graph_probability_input = torch.zeros(1, 1, peak_num)
                ms1_num = (1 - encoder_input['ms1_ms2_flag'][0]).sum().item()
                graph_probability_input[0, 0, 0 + ms1_num] = 1.0
                graph_probability_input = input_cuda(graph_probability_input, self.device)

                decoder_input = self.generate_decoder_input(graph_probability_input, encoder_input)
                graph_probability = self.generate_graph_probability(decoder_input, encoder_input, encoder_output)

                beam_size = min(self.cfg.infer.beam_size, peak_num - 1)
                graph_probability_input_list = [graph_probability_input] # elements in the list has shape (1, seq_len, peak_num)
                graph_probability_input_complete = []
                graph_probability_input_incomplete = []
                score_list = [1.0]
                score_complete = []
                score_incomplete = []
                graph_probability_list = [graph_probability] # elements in the list has shape (1, seq_len, peak_num]

                while(True):
                    graph_probability_input_list, score_list, graph_probability_input_complete, score_complete = \
                        self.beam_best_k(graph_probability_input_list, score_list, graph_probability_input_complete, score_complete, \
                                         graph_probability_list, encoder_input, precursor_mass, beam_size, sum_flag=sum_flag)

                    for path_idx, graph_probability_input in enumerate(graph_probability_input_list):
                        if self.is_graph_probability_input_complete(graph_probability_input):
                            graph_probability_input_complete.append(graph_probability_input)
                            score_complete.append(score_list[path_idx])
                        else:
                            graph_probability_input_incomplete.append(graph_probability_input)
                            score_incomplete.append(score_list[path_idx])

                    if beam_size <= len(score_complete) or len(score_incomplete) == 0:
                        score_complete = torch.tensor(score_complete)
                        path_index = score_complete.argmax().item()
                        graph_probability_input = graph_probability_input_complete[path_index]
                        encoder_output = encoder_output[0, :, :].unsqueeze(0)
                        pred_scores = self.extract_pred_prob(graph_probability_input, encoder_input, encoder_output)
                        break

                    graph_probability_input_list = graph_probability_input_incomplete
                    graph_probability_input_incomplete = []
                    score_list = score_incomplete
                    score_incomplete = []

                    # all paths not complete continue to generate the distribution to choose next node of the path
                    graph_probability_input_list = torch.stack([x.squeeze(0) for x in graph_probability_input_list])

                    decoder_input = self.generate_decoder_input(graph_probability_input_list, encoder_input)
                    encoder_output = torch.stack(len(graph_probability_input_list) * [encoder_output[0]])
                    graph_probability_list = self.generate_graph_probability(decoder_input, encoder_input, encoder_output)

                    graph_probability_input_list = [x.unsqueeze(0) for x in graph_probability_input_list]
                    graph_probability_list = [x.unsqueeze(0) for x in graph_probability_list]

                matched_num, predict_len, label_len, pred_path, label_path = self.path_evaluation(graph_probability_input, label, label_mask, encoder_input)
                pred_path = pred_path.cpu().numpy()
                label_path = label_path.cpu().numpy()
                pred_scores = pred_scores.cpu().numpy()

                path_pred_print = ' '.join([str(p) for p in pred_path])
                seq_predict = self.format_seq_predict(pred_path)

                path_label_print = ' '.join([str(p) for p in label_path])

                matched_num_total += matched_num
                predict_len_total += predict_len
                label_len_total += label_len
                if matched_num == predict_len == label_len:
                    whole_seq_match += 1
                peptide_predict_num += 1

                pred_prob = ' '.join([str(p) for p in pred_scores])
                writer_path.writerow({'graph_idx': idx[0], 'pred_path': path_pred_print, 'pred_prob': pred_prob, \
                                      'label_path': path_label_print, 'pred_seq': seq_predict})

        # Print the final evaluation
        print('matched_num_total: ', matched_num_total)
        print('predict_len_total: ', predict_len_total)
        print('label_len_total: ', label_len_total)
        print('node precision: ', matched_num_total / predict_len_total)
        print('node recall: ', matched_num_total / label_len_total)
        print('sequence recall: ', whole_seq_match / peptide_predict_num)
