"""
    Homonim: Radiometric homogenisation of aerial and satellite imagery
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
from pathlib import Path

from setuptools import setup, find_packages

"""
 Build and upload to testpypi:
     conda install -c conda-forge build twine
     python -m build
     python -m twine upload --repository testpypi dist/*

 Install from testpypi:
    python -m pip install --extra-index-url https://test.pypi.org/simple/ geedim

 Install local development version:
    pip install -e .
"""

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
version = {}
with open("homonim/version.py") as fp:
    exec(fp.read(), version)

setup(
    name="homonim",
    version=version['__version__'],
    description="Radiometric homogenisation of aerial imagery",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Dugal Harris",
    author_email="dugalh@gmail.com",
    url="https://github.com/dugalh/homonim",
    license='AGPLv3',
    packages=find_packages(exclude=['tests', 'data'], include=['homonim']),
    install_requires=["numpy>=1.2, <2", "rasterio>=1.1, <2", "click>=7.1, <8"],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License",
        "Operating System :: OS Independent",
    ],
    entry_points={"console_scripts": ["homonim=homonim.cli:cli"]},
)
