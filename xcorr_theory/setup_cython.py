from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "cython_xcorr",
        ["cython_xcorr.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name="cython_xcorr",
    ext_modules=cythonize(extensions),
)
