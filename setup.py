from distutils.core import setup

import rrmpg

setup(
    name="rrmpg",
    version=rrmpg.__reversion__,
    author=rrmpg.__author__,
    author_email="f.kratzert[at]gmail.com",
    description=("Rainfall-Runoff-Model-PlayGround: a Python library for"
                 "hydrological modeling."),
    url="https//www.github.com/kratzert/RRMPG",
    packages=["rrmpg", "rrmpg.models", "rrmpg.tools", "rrmpg.utils"],
    license="MIT-License",
    keywords="hydrology rainfall-runoff modeling",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License"]
    )
