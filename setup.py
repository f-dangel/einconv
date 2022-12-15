"""Setup file for einconv.

Use setup.cfg to configure the project.
"""

import sys

from pkg_resources import VersionConflict, require
from setuptools import find_packages, setup

try:
    require("setuptools>=38.3")
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)


if __name__ == "__main__":
    setup(use_scm_version=True, packages=find_packages(exclude=["third_party"]))
