
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

'''
compile with python setup_flif.py build_ext --inplace
'''

extensions = [
    Extension(
        name="flifnetwork_step",
        sources=["flifnetwork_step.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O2"]
    )
]

setup(
    name="flifnetwork_step",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
