#!/usr/bin/env python3
from pathlib import Path

import setuptools
from setuptools import setup

this_dir = Path(__file__).parent
module_dir = this_dir / "wyoming_securewakeword"

requirements = []
requirements_path = this_dir / "requirements.txt"
if requirements_path.is_file():
    with open(requirements_path, "r", encoding="utf-8") as requirements_file:
        requirements = requirements_file.read().splitlines()

models_dir = module_dir / "models"
data_files = [str(f.relative_to(module_dir)) for f in models_dir.glob("*.tflite")]

version_path = module_dir / "VERSION"
version = version_path.read_text(encoding="utf-8").strip()
data_files.append(str(version_path))

# -----------------------------------------------------------------------------

setup(
    name="wyoming_securewakeword",
    version=version,
    description="Wyoming server for a voice-authenticable wakeword",
    url="http://github.com/gws8820/wyoming-securewakeword",
    author="Michael Hansen",
    author_email="mike@rhasspy.org",
    packages=setuptools.find_packages(),
    package_data={"wyoming_securewakeword": data_files},
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="rhasspy wyoming securewakeword",
    entry_points={
        "console_scripts": ["wyoming-securewakeword = wyoming_securewakeword.__main__:run"]
    },
)
