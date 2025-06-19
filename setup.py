# setup.py

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

# Define the Cython extension
extensions = [
    Extension(
        "src.processing.cython_optimizations",
        ["src/processing/cython_optimizations.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],  # Optimization level
    )
]

setup(
    name="gsr_rgbt_project",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
    include_dirs=[np.get_include()],
    install_requires=[
        "numpy",
        "pandas",
        "opencv-python",
        "torch",
        "scikit-learn",
        "cython",
        "neurokit2",
        "pyshimmer",
        "pyyaml",
        "mediapipe",
        "matplotlib",
        "PyQt5",
    ],
)
