import os

import numpy as np
import pandas as pd
import pymzml
from tqdm import tqdm


def get_mzml_list(mzml_folder: str):
    mzml_list = [file_name for file_name in os.listdir(mzml_folder) if file_name.endswith('mzML')]
    mzml_list = [file_name.strip('.mzML') for file_name in mzml_list]
    # mzml_list = sorted(mzml_list) #TODO do this to ensure reproducibility
    return mzml_list


def read_one_mzml(mzml_path, timstof=False):
    ms1_df = {}
    ms2_df = {}

    run = pymzml.run.Reader(mzml_path)
    mzml_file = os.path.basename(mzml_path).strip('.mzML')

    for spectrum in tqdm(run, desc=f'Reading mzML {mzml_file}', total=run.get_spectrum_count()):
        if spectrum.ms_level == 1:
            ms1_df[mzml_file + ':' + str(spectrum.index)] = {
                'idx': spectrum.index,
                'run': mzml_file,
                'intensity': np.array(spectrum.i, dtype=np.float32),
                'mz': np.array(spectrum.mz,dtype=np.float32),
                'rt': spectrum.scan_time_in_minutes(),
            }

            if timstof:
                ms1_df[mzml_file + ':' + str(spectrum.index)]['ion_mobility'] = np.array(spectrum.get_tims_tof_ion_mobility(), dtype=np.float32)

        if spectrum.ms_level == 2:
            ms2_df[mzml_file + ':' + str(spectrum.index)] = {
                'idx': spectrum.index,
                'run': mzml_file,
                'intensity': np.array(spectrum.i, dtype=np.float32),
                'mz': np.array(spectrum.mz, dtype=np.float32),
                'rt': spectrum.scan_time_in_minutes(),
                'precursor_mz_target': spectrum["MS:1000827"],
                'precursor_mz_lower': spectrum["MS:1000827"] - spectrum["MS:1000828"],
                'precursor_mz_upper': spectrum["MS:1000827"] + spectrum["MS:1000829"]
            }

            if timstof:
                ms2_df[mzml_file + ':' + str(spectrum.index)]['ion_mobility'] = np.array(spectrum.get_tims_tof_ion_mobility(), dtype=np.float32)

    ms1_df = pd.DataFrame.from_dict(ms1_df, orient='index')
    ms2_df = pd.DataFrame.from_dict(ms2_df, orient='index')

    return ms1_df, ms2_df

def read_all_mzml(mzml_folder, ion_mobility=False):
    mzml_list = get_mzml_list(mzml_folder)
    ms1_df_list = []
    ms2_df_list = []
    for mzml_name in mzml_list:
        mzml_path = os.path.join(mzml_folder, mzml_name + '.mzML')
        ms1_df, ms2_df = read_one_mzml(mzml_path, ion_mobility)
        ms1_df_list.append(ms1_df)
        ms2_df_list.append(ms2_df)

    ms1_df = pd.concat(ms1_df_list)
    ms2_df = pd.concat(ms2_df_list)

    return ms1_df, ms2_df

def divide_feature_by_mzml(feature_list, mzml_list):
    divided_feature_list = [[] for _ in range(len(mzml_list))]
    for feature in feature_list:
        for i, mzml_name in enumerate(mzml_list):
            if mzml_name == feature['run']:
                divided_feature_list[i].append(feature)

    return divided_feature_list
