#!/bin/sh

sh parallel_preprocessing.sh 16 \
~/data1/denovo_dia_mzml/pain19777_train_valid/ \
~/data1/denovo_dia_reports/pain19777_train_valid/report.tsv.train \
~/data1/denovo_dia_reports/pain19777_train_valid/report.tsv.whole \
~/data2/RNova-DIA-Multi-Data/pain19777_train_valid/train/
