from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy
extensions = [
    Extension("LebwohlLasher_cy1", ["LebwohlLasher_cy1.pyx"],include_dirs=[numpy.get_include()])
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': 3}),
)
