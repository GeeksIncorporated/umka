from distutils.core import setup
from Cython.Build import cythonize

setup(name='Hello world app',
      ext_modules=cythonize([
          "core/*.pyx",
          "chess/*.pyx"],
          annotate=True))
