from setuptools import find_packages, setup
import rrmpg


with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="rrmpg",
    version=rrmpg.__reversion__,
    author=rrmpg.__author__,
    author_email="f.kratzert[at]gmail.com",
    description=("Rainfall-Runoff-Model-PlayGround: a Python library for"
                 "hydrological modeling."),
    url="https//www.github.com/kratzert/RRMPG",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    license="MIT-License",
    keywords="hydrology rainfall-runoff modeling",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License"],
    python_requires=">=3.11",
    )
