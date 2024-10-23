#!/usr/bin/env python
from setuptools import setup, find_packages
import sys
# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# check that python version is 3.7 or above
python_version = sys.version_info
if python_version < (3, 10):
    sys.exit("Python < 3.10 is not supported, aborting setup")

setup(
    name="pobs",
    version="0.1.0",
    description="posterior overlap, bayesian statistics",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Hemantakumar",
    license="MIT",
    author_email="hemantaphurailatpam@gmail.com",
    url="https://github.com/hemantaph/pobs",
    packages=find_packages(),
    python_requires='>=3.10',
    package_data={
        'pobs': ['data/*.json', 'data/*.pkl', 'data/*.h5'],
      },
    install_requires=[
        #"ler>=0.4.1",
        # sklearn
        # tensorflow
    ],
)