import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

dependencies = [
    "tqdm",
    "ipykernel",
    "seaborn",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "scipy",
    "torch",
    "torch-geometric",
    "lightning", 
    "xxhash",
    "uproot"
]

setup(
    name="epic_clustering",
    version="0.0.1",
    packages=find_packages(),
    author="Patrick McCormack, Daniel Murnane, Liv Helen Vage & Hannah Bossi",
    description=("Clustering of ePIC data"),
    license="MIT",
    keywords="ePIC clustering",
    url="https://github.com/wpmccormack/ePIC_Clustering_2023/tree/master",
    long_description=read('README.md'),
    install_requires=dependencies,
)
