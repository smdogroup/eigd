from setuptools import setup
import re
import os

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open("eigd/__init__.py").read(),
)[0]

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="eigd",
    version=__version__,
    description="Tools for eigenvector derivatives",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="",
    author="",
    author_email="graeme.kennedy@ae.gatech.edu",
    url="https://github.com/smdogroup/eigd",
    license="Apache License, Version 2.0",
    packages=[
        "eigd",
    ],
    install_requires=[
        "numpy>=1.16",
        "scipy>=1.7",
    ],
)
