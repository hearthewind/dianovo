import gzip
import os
import pickle
import traceback

from omegaconf import DictConfig
from tqdm import tqdm

import sys
sys.path.append('../')

from utils.data.peptide import read_diann_result
from utils.data.spectrum import get_mzml_list, divide_feature_by_mzml, read_one_mzml
from utils.graph_constructor import graph_gen
import random

def main(worker, total_worker, input_mzml_folder, input_feature_file, input_whole_feature_file, output_folder)->None:
    print('Reading diann result')
    raw_feature_list = read_diann_result(input_feature_file)
    whole_feature_list = read_diann_result(input_whole_feature_file)

    print('Dividing feature by mzml')
    mzml_list = get_mzml_list(input_mzml_folder)
    assert 1 <= worker <= total_worker
    mzml_per_worker = int(len(mzml_list) / total_worker)
    start_mzml, end_mzml = mzml_per_worker * (worker - 1), mzml_per_worker * worker
    if worker == total_worker:
        end_mzml = len(mzml_list)
    chosen_mzml_list = mzml_list[start_mzml:end_mzml]

    chosen_psm_list = divide_feature_by_mzml(raw_feature_list, chosen_mzml_list)

    print('Start generating graph')
    for psm_head, mzml_file in zip(chosen_psm_list, chosen_mzml_list):
        mzml_path = os.path.join(input_mzml_folder, mzml_file + '.mzML')
        ms1_df, ms2_df = read_one_mzml(mzml_path, ion_mobility=False)

        file_size = 0
        file_num = 0
        writer = open(os.path.join(output_folder, f'{mzml_file}_{file_num}.msgp'), 'wb')

        with open(os.path.join(output_folder, f'{mzml_file}.csv'), 'w', buffering=1) as index_writer:
            index_writer.write(
                'Spec Index,Annotated Sequence,Charge,m/z,MS1 Peak Number,MS2 Peak Number,MSGP File Name,MSGP Datablock Pointer,MSGP Datablock Length\n')

            for peptide in tqdm(psm_head):
                if file_size >= 2 * 1024 ** 3:
                    writer.close()
                    file_size = 0
                    file_num += 1
                    writer = open(os.path.join(output_folder, f'{mzml_file}_{file_num}.msgp'), 'wb')

                seq = peptide['mod_sequence']
                seq = seq.replace(' ', '')

                try:
                    ms1_mzs, ms1_xgrams, ms1_features, ms2_mzs, ms2_xgrams, ms2_features, \
                        cofragment_ms1s, cofragment_ms2s, target_ms1s, target_ms2s = graph_gen(peptide, ms1_df, ms2_df,
                                                                                               whole_feature_list)
                except Exception as e:
                    print('Error when generating graph for peptide: ', peptide['feature_id'])
                    traceback.print_exc()
                    continue

                record = {'ms1_mzs': ms1_mzs,
                          'ms1_xgrams': ms1_xgrams,
                          'ms1_features': ms1_features,
                          'ms2_mzs': ms2_mzs,
                          'ms2_xgrams': ms2_xgrams,
                          'ms2_features': ms2_features,
                          'cofragment_ms1_labels': cofragment_ms1s,
                          'cofragment_ms2_labels': cofragment_ms2s,
                          'target_ms1_labels': target_ms1s,
                          'target_ms2_labels': target_ms2s}
                compressed_data = gzip.compress(pickle.dumps(record))
                file_size += len(compressed_data)

                spec_index = peptide['feature_id']
                ms1_num, ms2_num = ms1_mzs.numel(), ms2_mzs.numel()
                charge = peptide['charge']

                index_writer.write(
                    '{},{},{},{},{},{},{},{},{}\n'.format(spec_index, seq, charge, peptide["precursor_mz"],
                                                          ms1_num, ms2_num, "{}_{}.msgp".format(mzml_file, file_num),
                                                          writer.tell(),
                                                          len(compressed_data)))

                writer.write(compressed_data)

                writer.flush()
                index_writer.flush()


if __name__=='__main__':
    worker, total_worker, input_mzml_folder, input_feature_file, input_whole_feature_file, output_folder = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
    main(worker, total_worker, input_mzml_folder, input_feature_file, input_whole_feature_file, output_folder)
