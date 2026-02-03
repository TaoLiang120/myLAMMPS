#!/usr/bin/env python

import os

from setuptools import setup, find_packages

module_dir = os.path.dirname(os.path.abspath(__file__))

with open('LICENSE.rst') as f:
    license = f.read()

setup(name='mylammps',
      version='1.0.0',
      description='My machine learning program for high entropy alloys',
      long_description=open(os.path.join(module_dir, 'README.rst')).read(),
      url='https://github.com/TaoLiang/myPolaron',
      author='Tao Liang',
      author_email='tliang7@utk.edu',
      license='MIT',
      packages=find_packages(),
      package_data={"mylammps.submission": ["*.json", "*.csv", "*.JOB", "*.ini"],
                    "mylammps.inputs": ["*.tmp"],
                    },
      install_requires=["numpy>=1.10.3",
                        "scipy>=0.17.1",
                        "matplotlib>=1.5.1",
                        "pymatgen>=4.4.0",
                        "scikit-learn>=1.3.2",
                        "pandas>=0.20.3"],
      )
