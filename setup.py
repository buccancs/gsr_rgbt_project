# setup.py

from setuptools import setup, find_packages
import numpy as np

setup(
    name="gsr_rgbt_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "opencv-python",
        "torch",
        "scikit-learn",
        "numba",
        "neurokit2",
        "pyshimmer",
        "pyyaml",
        "mediapipe",
        "matplotlib",
        "PyQt5",
        "pyserial",
    ],
)
