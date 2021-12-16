import os
import sys
from pathlib import Path

from setuptools import find_packages, setup

sys.path.insert(1, os.path.join(sys.path[0], ".."))

with open("requirements.txt") as f:
    required = f.read().splitlines()

os.chdir(Path(__file__).parent.absolute())
setup(
    name="cgtasknet",
    version="0.0.1",
    packages=find_packages(
        include=[
            "cgtasknet",
            "cgtasknet.*",
            "net",
            "net.*",
            "instrumetns",
            "instruments.*",
            "tasks",
            "tasks.*",
        ]
    ),
    install_requires=required,
    tests_require=["pytest"],
    setup_requires=["flake8", "pytest-runner"],
)
