"""
    Homonim: Correction of aerial and satellite imagery to surface reflectance
    Copyright (C) 2021 Dugal Harris
    Email: dugalh@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import sys
from pathlib import Path

from setuptools import setup, find_packages

"""
 Build and upload to testpypi:
     conda install -c conda-forge build twine
     python -m build
     python -m twine upload --repository testpypi dist/*

 Install from testpypi:
    python -m pip install --extra-index-url https://test.pypi.org/simple/ homonim

 Install local development version:
    pip install -e .
"""

this_directory = Path(__file__).parent
long_description = (this_directory / 'README.rst').read_text()
sys.path[0:0] = ['homonim']
from version import __version__

setup(
    name='homonim',
    version=__version__,
    description='Correct aerial and satellite imagery to surface reflectance.',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    author='Dugal Harris',
    author_email='dugalh@gmail.com',
    url='https://github.com/dugalh/homonim',
    license='AGPLv3',
    packages=find_packages(include=['homonim']),
    install_requires=[
        'numpy>=1.19',
        'rasterio>=1.1',
        'opencv-python-headless>=4.5',
        'click>=8',
        'tqdm>=4.6',
        'pyyaml>=5.4',
        'cloup>=0.15',
        'tabulate>=0.8',
    ],
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Operating System :: OS Independent',
    ],
    keywords=[
        'drone imagery', 'aerial imagery', 'satellite imagery', 'surface reflectance', 'correction', 'harmonization',
        'anisotropy', 'brdf', 'atmospheric correction',
    ],
    entry_points={'console_scripts': ['homonim=homonim.cli:cli']},
    project_urls={
        'Documentation': 'https://homonim.readthedocs.io',
        'Source': 'https://github.com/dugalh/homonim',
    },
)  # yapf: disable
