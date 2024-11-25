#!/bin/sh

sh parallel_preprocessing.sh 23 \
~/data1/denovo_dia_mzml/26600/narrow_mzml/ \
~/data1/denovo_dia_reports/26600_fixed/report_narrow_trypsin.tsv.unique \
~/data1/denovo_dia_reports/26600_fixed/report_narrow_trypsin.tsv \
~/data2/RNova-DIA-Multi-Data/26600_fixed/unique/
