from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("LebwohlLasher_cython", ["LebwohlLasher_cython.pyx"], include_dirs=[numpy.get_include()]),
]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)