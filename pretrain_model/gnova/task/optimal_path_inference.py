import csv
import gzip
import os
import pickle
from itertools import combinations_with_replacement

import numpy as np
import torch
from hydra.utils import get_original_cwd

from gnova.task.infer_utils import input_cuda, aa_datablock_dict_generate
from gnova.task.knapsack import knapsack_search
from utils.data.BasicClass import Ion, Residual_seq
from utils.data.theo_peak_information import mass_H, mass_C, mass_O, mass_N, mass_proton
from utils.graph_constructor import graph_gen

class OptimalPathInference:
    def __init__(self, cfg, device, spec_header, model, test_dl):
        self.cfg = cfg
        self.device = device
        self.spec_header = spec_header
        self.model = model
        self.test_dl = test_dl

        aa_dict = {aa: i for i, aa in enumerate(Residual_seq.output_aalist(), start=3)}
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

        self.label_dict = graph_gen.label_dict
        self.label_dict_reverse = {v: k for k, v in graph_gen.label_dict.items()}

        self.aa_datablock_dict, self.aa_datablock_dict_reverse = aa_datablock_dict_generate()
        self.aa_datablock = np.array(sorted(list(self.aa_datablock_dict.values())))

    def logits_to_class(self, graph_probabilities: torch.Tensor, ms1_peak_num: int, precursor_mz: float, mass_threshold: float, peak_mzs: torch.Tensor):
        # graph_probabilities should havec shape (batch_size, seq_len, peak_num, ion_types)
        # it is the direct output of model
        i = graph_probabilities[0, -1, :, :] # (peak_num, ion_types)
        peak_num = i.shape[0]

        ret = []
        for j in range(peak_num):
            peak_mz = peak_mzs[j].item()
            if peak_mz - mass_threshold <= precursor_mz <= peak_mz + mass_threshold:
                i_except0 = i[j, 1:]
                _, o = torch.max(i_except0)
            else:
                i_except0last = i[j, 1:-1]
                _, o = torch.max(i_except0last)
            o = o + 1
            ret.append(o.item())

            if j >= peak_num - ms1_peak_num:
                ret[-1] = self.label_dict['precursor']

        return torch.IntTensor(ret) # peak_num

    @staticmethod
    def one_hot_encode(label: torch.Tensor, num_classes: int=14):
        label = label.long()
        one_hot = torch.eye(num_classes)[label]
        return one_hot

    def generate_decoder_input(self, peak_probability_input, peak_mzs, precursor_mass):
        """ For single sample due to node_mass and dist"""
        # graph_probability_input has shape (batch_size, seq_len, peak_num) and each value is in range [0, ion_types]
        decoder_input = {}
        predict_peak_labels = self.one_hot_encode(peak_probability_input)
        decoder_input['peak_probs'] = predict_peak_labels

        peak_probability_input_list = [x.unsqueeze(0) for x in peak_probability_input]
        step_mass_list = []
        for i, peak_probability_input in enumerate(peak_probability_input_list):
            current_path = self.extract_totmass_from_peakprob_input(peak_probability_input, peak_mzs, precursor_mass)
            current_path = torch.cat([[0.0], current_path])
            step_mass = torch.cumsum(current_path, dim=0)
            step_mass_list.append(step_mass)
        decoder_input['step_mass'] = torch.stack(step_mass_list)

        seq_len = predict_peak_labels.shape[1]
        decoder_input['pos'] = torch.arange(seq_len)
        decoder_input = input_cuda(decoder_input, self.device)

        return decoder_input

    def generate_peak_probability(self, decoder_input, encoder_input, encoder_output):
        """ For the batch """
        peak_probs_input = decoder_input['peak_probs']
        step_mass = decoder_input['step_mass']
        pos = decoder_input['pos']

        batch_size, seq_len, peak_num, ion_types_test = peak_probs_input.shape
        try:
            assert ion_types_test == self.model.ion_types
        except AssertionError as e:
            print(f'ion_types_test: {ion_types_test}, model.ion_types: {self.model.ion_types}')
            raise e

        graph_node_expanded = self.model.encoder_expand_linear(encoder_output).view(batch_size, peak_num, self.model.ion_types, self.model.cfg.encoder.hidden_size)
        query_node = torch.einsum('ijkl,iklm->ijm', peak_probs_input, graph_node_expanded) # batch_size, seq_len, hidden_size
        query_node = query_node + self.model.pos_embedding(pos)

        peak_mzs = encoder_input['peak_mzs']
        peak_mzs_embed = self.model.encoder.peak_mzs_embedding(peak_mzs)
        neg_peak_mzs_embed = self.model.encoder.peak_mzs_embedding(-peak_mzs)

        query_node = self.model.decoder(query_node, step_mass, graph_node=encoder_output, \
                                        peak_mzs_embed=peak_mzs_embed, neg_peak_mzs_embed=neg_peak_mzs_embed)

        query_node = self.model.query_node_linear(query_node)  # batch_size, seq_len, hidden_size
        graph_node_expanded = self.model.graph_node_linear(graph_node_expanded).permute(0, 3, 1, 2)  # batch_size, hidden_size, peak_num, ion_types

        peak_probability = torch.einsum('ijk,iklm->ijlm', query_node, graph_node_expanded)  # batch_size, seq_len, peak_num, ion_types

        return peak_probability.softmax(dim=-1)

    @staticmethod
    def get_mass_prob(peak_probs):
        summation = np.sum(peak_probs)
        summation_square = np.sum([x ** 2 for x in peak_probs])
        return summation_square / (summation + 1.0)

    def extract_path_from_peakprob_input(self, peak_probability_input, peak_mzs, precursor_mass):
        # peak_probability_input has shape (1, seq_len, peak_num) and each value is in range [0, ion_types]
        peak_probability_input = peak_probability_input[0,1:,:]
        seq_len, peak_num = peak_probability_input.shape

        masses = [0.0]
        for j in range(seq_len):
            peak_label = peak_probability_input[j, :]
            for k, label in enumerate(peak_label):
                mz = peak_mzs[k].item()
                if label > 0:
                    ion_label = self.label_dict_reverse[label]
                    if k < seq_len - 1:
                        assert ion_label != 'precursor'
                        new_mass = Ion.peak2sequencemz(mz, ion_label)
                        if ion_label in graph_gen.c_term_ion_list:
                            new_mass = precursor_mass - new_mass
                        mass_tag = new_mass - masses[-1]
                    else:
                        assert ion_label == 'precursor'
                        new_mass = precursor_mass
                        mass_tag = new_mass - masses[-1]

                    masses.append(mass_tag)
                    break

        return torch.Tensor(masses[1:])

    def extract_totmass_from_peakprob_input(self, peak_probability_input, peak_mzs, precursor_mass):
        # peak_probability_input has shape (1, seq_len, peak_num) and each value is in range [0, ion_types]
        peak_probability_input = peak_probability_input[0,-1,:]

        last_label = peak_probability_input[-1]

        if last_label == self.label_dict['precursor']:
            return precursor_mass
        else:
            for k, label in enumerate(peak_probability_input):
                mz = peak_mzs[k].item()
                if label > 0:
                    ion_label = self.label_dict_reverse[label]
                    assert ion_label != 'precursor'
                    new_mass = Ion.peak2sequencemz(mz, ion_label)
                    if ion_label in graph_gen.c_term_ion_list:
                        new_mass = precursor_mass - new_mass
                    return new_mass

        return 0.0

    def extract_pred_prob(self, peak_probability_input, encoder_input, encoder_output, peak_mzs, precursor_mass):
        # peak_probability_input has shape (1, seq_len, peak_num) and each value is in range [0, ion_types]
        peak_probability_onehot = self.one_hot_encode(peak_probability_input[0,1:,:]) # shape [seqlen - 1, peak_num, ion_types]
        peak_probability_input = peak_probability_input[0,:-1,:]
        decoder_input = self.generate_decoder_input(peak_probability_input, peak_mzs, precursor_mass)
        pred_prob = self.generate_peak_probability(decoder_input, encoder_input, encoder_output)
        pred_prob = pred_prob[0]
        pred_prob = pred_prob * peak_probability_onehot # only keep the probability of the predicted label. shape [seqlen - 1, peak_num, ion_types]
        pred_prob = pred_prob.sum(-1) # shape [seqlen - 1, peak_num]

        seq_len, peak_num = pred_prob.shape
        mass_scores = [0.0]
        for j in range(seq_len):
            scores = pred_prob[j,:].cpu().numpy()
            mass_prob = self.get_mass_prob(scores)
            mass_scores.append(mass_prob)

        return torch.Tensor(mass_scores[1:])

    def is_knapsack_possible(self, suffix_mass: float):
        aa_resolution = 10000
        mass_precision_tolerance = 650
        knapsack_candidate = knapsack_search(self.knapsack_matrix, suffix_mass, mass_precision_tolerance, aa_resolution, self.aa_dict)
        return len(knapsack_candidate) > 0

    def vote_on_next_graphnode(self, peak_probabilities: torch.Tensor, current_mass: float, precursor_mass: float, precursor_mz: float, \
                               peak_mzs: torch.Tensor, ms1_peak_num: int):
        mass_threshold = graph_gen.mass_error_da + graph_gen.mass_error_ppm * precursor_mass * 1e-6

        # peak_probability has shape (batch_size, seq_len, peak_num, ion_types)
        peak_labels = self.logits_to_class(peak_probabilities, ms1_peak_num, precursor_mz, mass_threshold, peak_mzs) # peak_num
        peak_labels_onehot = self.one_hot_encode(peak_labels)  # peak_num, ion_types

        peak_probabilities = peak_probabilities[0, -1, :, :] # peak_num, ion_types
        peak_probabilities = peak_probabilities * peak_labels_onehot # only keep the probability of the predicted label. shape [peak_num, ion_types]
        peak_probabilities = peak_probabilities.sum(1) # shape [peak_num]

        mass_scores = {}
        for i in range(peak_probabilities.shape[0]):
            peak_label = peak_labels[i].item()
            peak_probability = peak_probabilities[i].item()
            mz = peak_mzs[i].item()
            ion_label = self.label_dict_reverse[peak_label]
            new_mass = Ion.peak2sequencemz(mz, ion_label)
            if ion_label in graph_gen.c_term_ion_list:
                new_mass = precursor_mass - new_mass
            if ion_label == 'precursor':
                new_mass = precursor_mass

            mass_tag = new_mass - current_mass
            suffix_mass = precursor_mass - new_mass

            if suffix_mass - mass_threshold <= 0 <= suffix_mass + mass_threshold:
                is_possible = True
            else:
                is_possible = self.all_possible_mass.searchsorted(mass_tag - mass_threshold) != self.all_possible_mass.searchsorted(mass_tag + mass_threshold)
                is_possible = is_possible and self.is_knapsack_possible(suffix_mass)

            if is_possible:
                for mass in mass_scores.keys():
                    if abs(mass - mass_tag) < mass_threshold:
                        mass_scores[mass].append(peak_probability)
                    else:
                        mass_scores[mass_tag] = [peak_probability]

        ret = {}
        for mass_tag in mass_scores.keys():
            ret[mass_tag] = self.get_mass_prob(mass_scores[mass_tag])

        return ret

    @staticmethod
    def get_theoretical_peaks_from_mass(prefix_mass: float, precursor_mass: float):
        suffix_mass = precursor_mass - prefix_mass + mass_H + mass_O
        # suffix_mass = sum([mass_AA[aa] for aa in mod_seq[location + 1:]]) + mass_O + mass_H

        neutral_mass_a = prefix_mass - mass_C - mass_H - mass_O + mass_H
        neutral_mass_a_nh3 = neutral_mass_a - mass_N - 3 * mass_H
        neutral_mass_a_h2o = neutral_mass_a - 2 * mass_H - mass_O
        neutral_mass_b = prefix_mass - mass_H + mass_H
        neutral_mass_b_nh3 = neutral_mass_b - mass_N - 3 * mass_H
        neutral_mass_b_h2o = neutral_mass_b - 2 * mass_H - mass_O
        neutral_mass_c = prefix_mass + mass_N + 2 * mass_H + mass_H
        neutral_mass_x = suffix_mass + mass_C + mass_O - mass_H
        neutral_mass_y = suffix_mass + mass_H
        neutral_mass_y_nh3 = neutral_mass_y - mass_N - 3 * mass_H
        neutral_mass_y_h2o = neutral_mass_y - 2 * mass_H - mass_O
        neutral_mass_z = suffix_mass - mass_N - 2 * mass_H

        def get_mz_from_neutral_mass(neutral_mass: float, charge: int):
            mass = neutral_mass + charge * mass_proton
            return mass / charge

        theoretical_mzs = {'1a': get_mz_from_neutral_mass(neutral_mass_a, 1),
                           '2a': get_mz_from_neutral_mass(neutral_mass_a, 2),
                           '1a-NH3': get_mz_from_neutral_mass(neutral_mass_a_nh3, 1),
                           '2a-NH3': get_mz_from_neutral_mass(neutral_mass_a_nh3, 2),
                           '1a-H2O': get_mz_from_neutral_mass(neutral_mass_a_h2o, 1),
                           '2a-H2O': get_mz_from_neutral_mass(neutral_mass_a_h2o, 2),
                           '1b': get_mz_from_neutral_mass(neutral_mass_b, 1),
                           '2b': get_mz_from_neutral_mass(neutral_mass_b, 2),
                           '1b-NH3': get_mz_from_neutral_mass(neutral_mass_b_nh3, 1),
                           '2b-NH3': get_mz_from_neutral_mass(neutral_mass_b_nh3, 2),
                           '1b-H2O': get_mz_from_neutral_mass(neutral_mass_b_h2o, 1),
                           '2b-H2O': get_mz_from_neutral_mass(neutral_mass_b_h2o, 2),
                           '1c': get_mz_from_neutral_mass(neutral_mass_c, 1),
                           '2c': get_mz_from_neutral_mass(neutral_mass_c, 2),
                           '1x': get_mz_from_neutral_mass(neutral_mass_x, 1),
                           '2x': get_mz_from_neutral_mass(neutral_mass_x, 2),
                           '1y': get_mz_from_neutral_mass(neutral_mass_y, 1),
                           '2y': get_mz_from_neutral_mass(neutral_mass_y, 2),
                           '1y-NH3': get_mz_from_neutral_mass(neutral_mass_y_nh3, 1),
                           '2y-NH3': get_mz_from_neutral_mass(neutral_mass_y_nh3, 2),
                           '1y-H2O': get_mz_from_neutral_mass(neutral_mass_y_h2o, 1),
                           '2y-H2O': get_mz_from_neutral_mass(neutral_mass_y_h2o, 2),
                           '1z': get_mz_from_neutral_mass(neutral_mass_z, 1),
                           '2z': get_mz_from_neutral_mass(neutral_mass_z, 2)}

        return theoretical_mzs

    def generate_next_peak_probability_input(self, peak_probability_input, current_mass, precursor_mass, precursor_mz, peak_mzs, ms1_peak_num):
        mass_threshold = graph_gen.mass_error_da + graph_gen.mass_error_ppm * precursor_mass * 1e-6
        if current_mass - mass_threshold <= precursor_mass <= current_mass + mass_threshold:
            ret = []
            for i, mz in enumerate(peak_mzs):
                if i < len(peak_mzs) - ms1_peak_num:
                    if mz - mass_threshold <= precursor_mz <= mz + mass_threshold:
                        ret.append(self.label_dict['precursor'])
                    else:
                        ret.append(self.label_dict['noise'])
                else:
                    ret.append(self.label_dict['precursor'])
        else:
            all_theo_mzs = self.get_theoretical_peaks_from_mass(current_mass, precursor_mass)
            chosen_theo_mzs = [(mz, ion) for ion, mz in all_theo_mzs.items() if ion in graph_gen.ion_types]

            ret = []
            for i, mz in enumerate(peak_mzs):
                if i < len(peak_mzs) - ms1_peak_num:
                    for theo_mz, ion in chosen_theo_mzs:
                        if mz - mass_threshold <= theo_mz <= mz + mass_threshold:
                            ret.append(self.label_dict[ion])
                            break
                else:
                    ret.append(self.label_dict['noise'])

        ret = torch.IntTensor(ret)
        ret = ret.unsqueeze(0).unsqueeze(0) # (1, 1, peak_num)
        ret = torch.cat([peak_probability_input, ret], dim=1)
        return ret


    def beam_best_k(self, peak_probability_input_list, score_list, peak_probability_input_complete, score_complete, \
                    peak_probability_list, beam_size, sum_flag, precursor_mass, precursor_mz, peak_mzs, ms1_peak_num):
        """ Return k best results and the corresponding scores.
            For single sample.
        """
        _, path_len, peak_num = peak_probability_input_list[0].shape

        score_list_extend = []
        peak_probability_input_list_extend = []
        for i, peak_probability in enumerate(peak_probability_list):
            peak_probability_input = peak_probability_input_list[i]
            current_mass = self.extract_totmass_from_peakprob_input(peak_probability_input, peak_mzs, precursor_mass)
            current_score = score_list[i]
            vote_result = self.vote_on_next_graphnode(peak_probability, current_mass, precursor_mass, precursor_mz, \
                                                      peak_mzs, ms1_peak_num)

            for mass_tag in vote_result.keys():
                mass_score = vote_result[mass_tag]

                if sum_flag == 'sum':
                    score_list_extend.append(current_score + mass_score)
                elif sum_flag == 'multiply':
                    score_list_extend.append(current_score * mass_score)
                peak_probability_input_list_extend.append(self.generate_next_peak_probability_input(peak_probability_input, current_mass + mass_tag, precursor_mass, precursor_mz, peak_mzs, ms1_peak_num))

        for s in score_complete:
            score_list_extend.append(s)  # s should be a single float, representing the score of the complete path

        score_list_extend = torch.Tensor(score_list_extend)
        _, topk_index = torch.topk(score_list_extend, k=beam_size)

        topk_peak_probability_input = []
        new_peak_probability_input_complete = []
        new_score_list = []
        new_score_complete = []
        for idx in topk_index:
            idx = idx.item()
            try:
                peak_probability_input = peak_probability_input_list_extend[idx]
                topk_peak_probability_input.append(peak_probability_input)
                new_score_list.append(score_list_extend[idx])
            except IndexError as _:
                complete_index = idx - len(peak_probability_input_list_extend)
                new_score_complete.append(score_complete[complete_index])
                new_peak_probability_input_complete.append(peak_probability_input_complete[complete_index])

        return topk_peak_probability_input, new_score_list, new_peak_probability_input_complete, new_score_complete

    def inference(self):
        print('graphnovo_dir:', self.graphnovo_dir)
        path_file_name = os.path.join(self.graphnovo_dir, self.cfg.infer.optimal_path_file)
        print('path_file_name: ', path_file_name)
        path_csvfile = open(path_file_name, 'w', buffering=1)
        path_fieldnames = ['psm_idx', 'pred_path', 'pred_prob', 'label_path', 'pred_seq']
        writer_path = csv.DictWriter(path_csvfile, fieldnames=path_fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer_path.writeheader()

        # metrics initialization
        matched_num_total, predict_len_total, label_len_total = 0, 0, 0
        # main
        sum_flag = 'multiply'
        for i, (encoder_input, _, label, label_mask, idx) in enumerate(self.test_dl):
            if i % 100 == 0 and i > 0:
                print('Num of Samples: ', i)
            if torch.is_tensor(idx): idx = idx.tolist()
            spec_head = self.spec_header.loc[idx[0]]

            with open(os.path.join(self.graphnovo_dir, self.cfg.infer.data_dir, spec_head['MSGP File Name']), 'rb') as f:
                f.seek(spec_head['MSGP Datablock Pointer'])
                spec = pickle.loads(gzip.decompress(f.read(spec_head['MSGP Datablock Length'])))

                precursor_charge = int(spec_head['Charge'])
                precursor_mz = float(spec_head['m/z'])
                precursor_mass = Ion.precursorion2mass(precursor_mz, precursor_charge)

                ms1_peak_num = spec['ms1_peak_num']

            with torch.no_grad():
                encoder_input = input_cuda(encoder_input, self.device)
                encoder_output = self.model.encoder(**encoder_input)

                peak_mzs = encoder_input['peak_mzs']
                peak_num = peak_mzs.shape[1]

                peak_probability_input = torch.zeros(1, 1, peak_num) # (batch_size, seq_len, peak_num)
                peak_probability_input = peak_probability_input.to(self.device)

                decoder_input = self.generate_decoder_input(peak_probability_input, peak_mzs, precursor_mass)
                peak_probability = self.generate_peak_probability(decoder_input, encoder_input, encoder_output)

                beam_size = min(self.cfg.infer.beam_size, peak_num - 1)
                peak_probability_input_list = [peak_probability_input] # elements in the list has shape (1, seq_len, peak_num)
                peak_probability_input_complete = []
                peak_probability_input_incomplete = []
                score_list = [0.0]
                score_complete = []
                score_incomplete = []
                peak_probability_list = [peak_probability] # elements in the list has shape (1, seq_len, peak_num, ion_types)

                while(True):
                    peak_probability_input_list, score_list, peak_probability_input_complete, score_complete = \
                        self.beam_best_k(peak_probability_input_list, score_list, peak_probability_input_complete, score_complete, \
                                         peak_probability_list, beam_size, sum_flag, precursor_mass, precursor_mz, peak_mzs, ms1_peak_num)

                    for path_idx, peak_probability_input in enumerate(peak_probability_input_list):
                        if self.is_peak_probability_input_complete(peak_probability_input):
                            peak_probability_input_complete.append(peak_probability_input)
                            score_complete.append(score_list[path_idx])
                        else:
                            peak_probability_input_incomplete.append(peak_probability_input)
                            score_incomplete.append(score_list[path_idx])

                    if beam_size == len(score_complete):
                        score_complete = torch.Tensor(score_complete)
                        path_index = score_complete.argmax().item()
                        peak_probability_input = peak_probability_input_complete[path_index]
                        pred_scores = self.extract_pred_prob(peak_probability_input, encoder_input, encoder_output, peak_mzs, precursor_mass)
                        break

                    peak_probability_input_list = peak_probability_input_incomplete
                    peak_probability_input_incomplete = []
                    score_list = score_incomplete
                    score_incomplete = []

                    # all paths not complete continue to generate the distribution to choose next node of the path
                    peak_probability_input_list = torch.stack(peak_probability_input_list, dim=0)
                    decoder_input = self.generate_decoder_input(peak_probability_input_list, peak_mzs, precursor_mass)
                    encoder_output = torch.stack(len(peak_probability_input_list) * [encoder_output[0]], dim=0)
                    peak_probability_list = self.generate_peak_probability(decoder_input, encoder_input, encoder_output)

                    peak_probability_input_list = [x.unsqueeze(0) for x in peak_probability_input_list]
                    peak_probability_list = [x.unsqueeze(0) for x in peak_probability_list]

            matched_num, predict_len, label_len, pred_path, label_path = self.path_evaluation(peak_probability_input, label, label_mask, peak_mzs, precursor_mass)
            path_pred_print = ' '.join([str(p) for p in pred_path])
            seq_predict = self.format_seq_predict(pred_path, precursor_mass)

            path_label_print_tmp = ['/'.join([str(p) for p in ps]) for ps in label_path]
            path_label_print = ' '.join(path_label_print_tmp)

            matched_num_total += matched_num
            predict_len_total += predict_len
            label_len_total += label_len

            pred_prob = ' '.join([str(p) for p in pred_scores])
            writer_path.writerow({'graph_idx': idx[0], 'pred_path': path_pred_print, 'pred_prob': pred_prob, \
                                  'label_path': path_label_print, 'pred_seq': seq_predict})

        # Print the final evaluation
        print('matched_num_total: ', matched_num_total)
        print('predict_len_total: ', predict_len_total)
        print('label_len_total: ', label_len_total)
        print('node precision: ', matched_num_total / predict_len_total)
        print('node recall: ', matched_num_total / label_len_total)

    def format_seq_predict(self, pred_path, precursor_mass):
        seq_predict = []
        for edge_mass in pred_path:
            mass_threshold = graph_gen.mass_error_da + graph_gen.mass_error_ppm * precursor_mass * 1e-6

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

    def is_peak_probability_input_complete(self, peak_probability_input: torch.Tensor):
        # peak_probability_input has shape (1, seq_len, peak_num) and each value is in range [0, ion_types]
        peak_labels = peak_probability_input[0, -1, :]
        if peak_labels[-1].item() == self.label_dict['precursor']:
            return True
        else:
            return False

    def path_evaluation(self, peak_probability_input, label, label_mask, peak_mzs, precursor_mass):
        # peak_probability_input has shape (batch_size, seq_len, peak_num) and each value is in range [0, ion_types]
        # label should have same shape
        # label_mask should have shape (batch_size, seq_len)
        """ For single sample """
        pred_path = self.extract_path_from_peakprob_input(peak_probability_input, peak_mzs, precursor_mass)
        pred_len = len(pred_path)

        label_len = torch.sum(label_mask[0]).item()
        label_peakprob_input = label[:, :label_len, :]
        peak_num = label_peakprob_input[2]
        label_peakprob_input = torch.cat([torch.zeros(1, 1, peak_num), label_peakprob_input], dim=1)
        label_path = self.extract_path_from_peakprob_input(label_peakprob_input, peak_mzs, precursor_mass)

        target_mass_cum = np.concatenate([[0.0], np.cumsum(label_path)])
        predicted_mass_cum = np.concatenate([[0.0], np.cumsum(pred_path)])
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
