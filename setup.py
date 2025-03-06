from setuptools import setup, find_packages

__version__ = "0.2.0"

setup(
    name="gradflow",
    version=__version__,
    description="A simple tensor library powered by numpy",
    packages=find_packages(),
    url="https://github.com/agnarbjoernstad/gradflow",
    author="Agnar Martin Bjørnstad",
    author_email="agnar.bjornstad@gmail.com",
    license="MIT",
    install_requires=[
        "numpy>=2.0",
        "matplotlib>=3.0",
        "networkx>=2.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
