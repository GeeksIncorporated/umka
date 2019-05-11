from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([
        "solve_puzzle.pyx", "chess/*.pyx", "core/*.pyx"
    ]),
)
