import os
import re
import shutil
from distutils.core import Command
from pathlib import Path

from setuptools import find_packages, setup


# IMPORTANT:
# 1. all dependencies should be listed here with their version requirements if any
_deps = [
    "isort>=5.5.4",
    "numpy>=1.17",
    "pyyaml>=5.1",
    "pytest",
    "pytest-timeout",
    "pytest-xdist",
    "python>=3.8.0",
    "timm",
    "torch>=1.0",
    "tqdm>=4.27",
]


# this is a lookup table with items like:
#
# tokenizers: "tokenizers==0.9.4"
# packaging: "packaging"
#
# some of the values are versioned whereas others aren't.
deps = {b: a for a, b in (re.findall(r"^(([^!=<>~]+)(?:[!=<>~].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]

extras = {}

extras["torch"] = deps_list("torch")

extras["timm"] = deps_list("timm")

extras["testing"] = (
    deps_list(
        "pytest",
        "pytest-xdist",
        "pytest-timeout",
    )
)


extras["quality"] = deps_list("isort")

extras["all"] = (
    extras["torch"]
    + extras["timm"]
)

extras["dev-torch"] = (
    extras['testing']
    + extras['torch']
    + extras["timm"]
    + extras["quality"]
)
extras["dev"] = (
    extras["all"]
    + extras["testing"]
    + extras["quality"]
)

extras["torchhub"] = deps_list(
    "numpy",
    "torch",
    "tqdm",
)


install_requires = [
    deps["numpy"],
    deps["pyyaml"],  # used for the model cards metadata
    deps["tqdm"],  # progress bars in model download and training scripts
]

setup(
    name="pytorch-models",
    version="0.0.1.dev0",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    author="Christian Saravia",
    author_email="chrissaher@gmail.com",
    description="",
    long_description="",
    long_description_content_type="",
    keywords="",
    license="Apache",
    url="https://github.com/chrissaher/pytorch-models",
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=False,
    extras_require=extras,
    python_requires=">=3.8.0",
    install_requires=install_requires,
)
