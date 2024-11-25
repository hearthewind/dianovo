import csv

import pandas as pd

import sys
sys.path.append('../../../')

from utils.data.BasicClass import Residual_seq, Ion


def read_diann_result(diann_result_file):
    with open(diann_result_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')

        ret = []
        for item in reader:
            mod_seq = item['Modified.Sequence']
            mod_seq = mod_seq.replace('M(UniMod:35)', 'm').replace('C(UniMod:4)', 'c').replace('L', 'I')
            stripped_seq = item['Stripped.Sequence']
            charge = int(item['Precursor.Charge'])
            diann_score = float(item['CScore'])
            rt = float(item['RT'])
            rt_start = float(item['RT.Start'])
            rt_stop = float(item['RT.Stop'])
            run = item['Run']
            precursor_id = item['Precursor.Id']
            area = float(item['Ms1.Area'])

            precursor_mass = Residual_seq(mod_seq).mass
            precursor_mz = Ion.precursormass2ion(precursor_mass, charge)
            feature_id = precursor_id + '_' + run

            new_item = {'mod_sequence': mod_seq,
                        'stripped_sequence': stripped_seq,
                        'charge': charge,
                        'diann_score': diann_score,
                        'rt': rt,
                        'rt_start': rt_start,
                        'rt_stop': rt_stop,
                        'run': run,
                        'precursor_mass': precursor_mass,
                        'precursor_mz': precursor_mz,
                        'precursor_id': precursor_id,
                        'feature_id': feature_id,
                        'area': area}
            ret.append(new_item)

        return ret


def get_ms2_from_precursor(precursor: dict, ms2_df: pd.DataFrame):

    rt_start = precursor['rt_start']
    rt_stop = precursor['rt_stop']
    run = precursor['run']
    precursor_mz = precursor['precursor_mz']

    selected_ms2 = ms2_df[(ms2_df['rt'] >= rt_start) & (ms2_df['rt'] <= rt_stop) & \
                          (ms2_df['precursor_mz_lower'] <= precursor_mz) & \
                          (ms2_df['precursor_mz_upper'] >= precursor_mz) & \
                          (ms2_df['run'] == run)]

    return selected_ms2


def get_ms1_from_precursor(precursor: dict, ms1_df: pd.DataFrame):

    rt_start = precursor['rt_start']
    rt_stop = precursor['rt_stop']
    run = precursor['run']

    selected_ms1 = ms1_df[(ms1_df['rt'] >= rt_start) & (ms1_df['rt'] <= rt_stop) & \
                          (ms1_df['run'] == run)]

    return selected_ms1

def filter_ms1_precursor_window(related_ms1, related_ms2):
    first_ms2 = related_ms2.iloc[0]
    precursor_mz_lower = first_ms2['precursor_mz_lower']
    precursor_mz_upper = first_ms2['precursor_mz_upper']

    new_related_ms1 = []

    for i, row in related_ms1.iterrows():
        mzs = row.mz
        selected_indices = (mzs > precursor_mz_lower) & (mzs < precursor_mz_upper)

        related_ms1_i = pd.DataFrame(row).transpose()
        related_ms1_i['mz'] = related_ms1_i['mz'].apply(lambda x: x[selected_indices])
        related_ms1_i['intensity'] = related_ms1_i['intensity'].apply(lambda x: x[selected_indices])

        new_related_ms1.append(related_ms1_i)

    return pd.concat(new_related_ms1)