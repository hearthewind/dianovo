import numpy as np

ion_types = ['1a','1b','2a','2b','1a-NH3','1a-H2O','1b-NH3','1b-H2O'] + ['1y','1y-NH3','1y-H2O','2y'] # total 12 ions

label_types = ['noise'] + ion_types + ['ms1']
label_dict = dict(zip(label_types, np.arange(len(label_types))))

mass_threshold = 0.02





