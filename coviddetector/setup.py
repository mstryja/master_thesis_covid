from setuptools import setup, find_packages
import os

here = os.path.dirname(os.path.abspath(__file__))

name = 'coviddet'


def requirements():
    with open(os.path.join(here, name, "requirements.txt")) as f:
        return f.read().splitlines()


def version(*parts):
    version_file = os.path.join(*parts)
    version_ns = {}

    with open(version_file) as f:
        exec(f.read(), {}, version_ns)

    return version_ns["__version__"]


setup(
    name=name,
    version=version(here, name, "_version.py"),
    install_requires=requirements(),
    packages=find_packages(),
)