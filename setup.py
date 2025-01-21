# -*- coding: utf-8 -*-
from __future__ import annotations

from setuptools import find_packages
from setuptools import setup


setup(
    name="pioreactor-calibration-shrinkage",
    version="1.0.0",
    license="MIT",
    description="Fuse multiple calibrations into better calibrations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author_email="hello@pioreactor.com",
    author="Pioreactor",
    url="https://github.com/Pioreactor/pioreactor-calibration-shrinkage",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "pioreactor.plugins": "pioreactor_calibration_shrinkage = pioreactor_calibration_shrinkage"
    },
)
