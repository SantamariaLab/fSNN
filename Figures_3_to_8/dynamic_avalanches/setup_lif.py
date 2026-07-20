
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

'''
compile with python setup_lif.py build_ext --inplace
'''

extensions = [
    Extension(
        name="lifnetwork_step",
        sources=["lifnetwork_step.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O2"]
    )
]

setup(
    name="lifnetwork_step",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
