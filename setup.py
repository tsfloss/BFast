import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "BFast",
    version = "0.0.1",
    author = "Thomas Flöss",
    author_email = "tsfloss@gmail.com",
    description = ("A fast GPU based bispectrum estimator implemented using TensorFlow"),
    license = "MIT",
    url = "https://github.com/tsfloss/BFast",
    packages=['BFast'],
    install_requires=[
          'numpy',"tqdm","jax-tqdm",
      ],
)