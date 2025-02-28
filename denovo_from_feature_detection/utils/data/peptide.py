import csv

import pandas as pd

import sys
sys.path.append('../../../')



def get_ms2_from_precursor(precursor: pd.Series, ms2_df: pd.DataFrame):

    rt_start = precursor['StartRT']
    rt_stop = precursor['EndRT']
    precursor_mz = precursor['mz1']

    selected_ms2 = ms2_df[(ms2_df['rt'] >= rt_start) & (ms2_df['rt'] <= rt_stop) & \
                          (ms2_df['precursor_mz_lower'] <= precursor_mz) & \
                          (ms2_df['precursor_mz_upper'] >= precursor_mz)]

    return selected_ms2


def get_ms1_from_precursor(precursor: pd.Series, ms1_df: pd.DataFrame):

    rt_start = precursor['StartRT']
    rt_stop = precursor['EndRT']

    selected_ms1 = ms1_df[(ms1_df['rt'] >= rt_start) & (ms1_df['rt'] <= rt_stop)]

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