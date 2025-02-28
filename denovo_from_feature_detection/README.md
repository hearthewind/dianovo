# De Novo from Feature Detection

# How to generate de novo data

Please refer to DIA-Umpire for the feature detection, you will need DIA-Umpire's signal extraction module. The feature detection result is stored in the PeakCluster.csv file.
For more information, refer to https://github.com/Nesvilab/DIA-Umpire.

Under the utils folder, you can find the script to generate de novo data, ``gen_test.sh``. The program takes in DIAUmpire feature detecvtion file (PeakCluster.csv file) and spectrum files (.mzML files) and produce our internal data (.csv files for the header, as well as .msgp files for the data).

The format to run the code is 

```
sh parallel_preprocessing.sh [num_worker] \ 
[mzml file path] \
[DIAUmpire PeakCluster path] \
[output dir]
```

It takes around 0.8 seconds to process a single peptide, but this process is parallelized.

# How to perform de novo sequencing