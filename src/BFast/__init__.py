"""
BFast

Bispectrum estimator in jax

Author: Thomas Flöss <tsfloss@gmail.com>
License: MIT
"""

# Package version
__version__ = "2.0.0"

# Optional metadata
__author__ = "Thomas Flöss"
__email__ = "tsfloss@gmail.com"

# Expose main submodules / classes / functions for easy import
from .core.bispectrum import *

# # __all__ controls what gets imported with `from your_project import *`
# __all__ = [
#     "Bk",
#     "Model",
#     "compute_statistics",
#     "plot_field",
# ]
