from pathlib import Path
from setuptools import find_namespace_packages, setup

setup(
    name="mmd-glm",
    version="0.1",
    url="https://github.com/diegoarri91/mmd-glm",
    author="Diego M. Arribas",
    author_email="diegoarri91@gmail.com",
    license="MIT",
    install_requires=[
        "matplotlib",
        "numpy",
        "scipy",
        "torch",
    ],
    packages=find_namespace_packages(str(Path(__file__).parent), include=["mmdglm*"]),
)
