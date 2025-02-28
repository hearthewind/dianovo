import gzip
import os
import pickle
import traceback

import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../')

from utils.data.spectrum import read_one_mzml
from utils.graph_constructor import graph_gen

def main(worker, total_worker, input_mzml_file, input_feature_file, output_folder)->None:
    print('Reading feature detection result')
    raw_feature_df = pd.read_csv(input_feature_file, index_col='Cluster_Index')

    feature_per_worker = int(len(raw_feature_df) / total_worker)
    start_feature, end_feature = feature_per_worker * (worker - 1), feature_per_worker * worker

    chosen_feature_df = raw_feature_df.iloc[start_feature:end_feature]

    mzml_path = input_mzml_file
    mzml_filename = os.path.basename(mzml_path).split('.')[0]
    ms1_df, ms2_df = read_one_mzml(mzml_path, ion_mobility=False)

    print('Start generating graph')

    file_size = 0
    file_num = 0
    writer = open(os.path.join(output_folder, f'{mzml_filename}_{worker}_{file_num}.msgp'), 'wb')

    with open(os.path.join(output_folder, f'{mzml_filename}_{worker}.csv'), 'w', buffering=1) as index_writer:
        index_writer.write(
            'Spec Index,Charge,m/z,MS1 Peak Number,MS2 Peak Number,MSGP File Name,MSGP Datablock Pointer,MSGP Datablock Length\n')

        for _, precursor in tqdm(chosen_feature_df.iterrows(), total=len(chosen_feature_df)):
            if file_size >= 2 * 1024 ** 3:
                writer.close()
                file_size = 0
                file_num += 1
                writer = open(os.path.join(output_folder, f'{mzml_filename}_{worker}_{file_num}.msgp'), 'wb')

            try:
                ms1_mzs, ms1_xgrams, ms1_features, ms2_mzs, ms2_xgrams, ms2_features = graph_gen(precursor, ms1_df, ms2_df)
            except Exception as e:
                print('Error when generating graph for precursor: ', precursor.name)
                traceback.print_exc()
                continue

            record = {'ms1_mzs': ms1_mzs,
                      'ms1_xgrams': ms1_xgrams,
                      'ms1_features': ms1_features,
                      'ms2_mzs': ms2_mzs,
                      'ms2_xgrams': ms2_xgrams,
                      'ms2_features': ms2_features}
            compressed_data = gzip.compress(pickle.dumps(record))
            file_size += len(compressed_data)

            spec_index = precursor.name
            ms1_num, ms2_num = ms1_mzs.numel(), ms2_mzs.numel()
            charge = precursor['Charge']

            index_writer.write(
                '{},{},{},{},{},{},{},{}\n'.format(f'feature_{spec_index}', int(charge), precursor["mz1"],
                                                      ms1_num, ms2_num, "{}_{}_{}.msgp".format(mzml_filename, worker, file_num),
                                                      writer.tell(),
                                                      len(compressed_data)))

            writer.write(compressed_data)

            writer.flush()
            index_writer.flush()


if __name__=='__main__':
    worker, total_worker, input_mzml_file, input_feature_file, output_folder = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
    main(worker, total_worker, input_mzml_file, input_feature_file, output_folder)
