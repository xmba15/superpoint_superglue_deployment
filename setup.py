import os
from io import open
from typing import Any, Dict

from setuptools import find_packages, setup

_PARENT_DIRECTORY = os.path.abspath(os.path.dirname(__file__))
_PACKAGE_NAME = "superpoint_superglue_deployment"

_PACKAGE_VARS: Dict[str, Any] = {}
exec(open(os.path.join(_PARENT_DIRECTORY, _PACKAGE_NAME, "version.py")).read(), _PACKAGE_VARS)

_LONG_DESCRIPTION = open(os.path.join(_PARENT_DIRECTORY, "README.md"), encoding="utf-8").read()
_INSTALL_REQUIRES = open(os.path.join(_PARENT_DIRECTORY, "requirements.txt")).read().splitlines()
_INSTALL_REQUIRES = [line for line in _INSTALL_REQUIRES if line and not line.startswith("#")]


def main():
    setup(
        name=_PACKAGE_NAME,
        version=_PACKAGE_VARS["__version__"],
        description="",
        long_description=_LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        author="Ba Tran",
        url="https://github.com/xmba15/superpoint_superglue_deployment",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
        ],
        packages=find_packages(exclude=["tests"]),
        install_requires=_INSTALL_REQUIRES,
    )


if __name__ == "__main__":
    main()
