from distutils.core import setup

import rrmpg

setup(
    name="rrmpg",
    description="Rainfall-Runoff-Model-PlayGround",
    author=rrmpg.__author__,
    url="https//www.github.com/kratzert/RRMPG",
    version=rrmpg.__reversion__,
    packages=["rrmpg", "rrmpg.models", "rrmpg.tools", "rrmpg.utils"],
    license="MIT-License"
    )
