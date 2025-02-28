from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(["utils/cofragment_peptide_processing_cy.pyx",
                           "utils/data/processing_ms2_cy.pyx",
                           "utils/data/peak_feature_generator_cy.pyx",]),
    include_dirs=[np.get_include()]
)