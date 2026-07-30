import os
from distutils.sysconfig import get_python_inc

import numpy as np
from setuptools import find_packages, setup

# Kept minimal: the README installs PyTorch and the scientific stack through
# conda, then runs `pip install -e . --no-deps`. These are the packages the
# eight methods and the evaluation pipeline import directly.
requirements = [
    'numpy',
    'scipy',
    'h5py',
    'imageio',
    'tifffile',
    'scikit-image',
    'scikit-learn',
    'opencv-python',
    'einops',
    'tqdm',
    'matplotlib',
    'yacs',
    'pyyaml',
    'tensorboard',
    'mahotas',
    'fire',
    'tabulate',
]


def getInclude():
    dirName = get_python_inc()
    return [dirName, os.path.dirname(dirName), np.get_include()]


def setup_package():
    setup(name='connectomics',
          description='Neuron segmentation methods and evaluation pipeline for BrainEM',
          version='0.1',
          url='https://github.com/kwinderic/BrainEM',
          license='MIT',
          install_requires=requirements,
          include_dirs=getInclude(),
          packages=find_packages(),
          )


if __name__ == '__main__':
    setup_package()
