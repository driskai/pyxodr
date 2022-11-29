from setuptools import find_packages, setup

requires = [
    "lxml>=4.9.1",
    "matplotlib>=3.5.0",
    "numpy>=1.21.0",
    "rich>=12.6.0",
    "scipy>=1.7.0",
    "Shapely>=1.8.4",
]

extras = {"dev": ["pytest>=7.1.3"]}

setup(
    author="Hugh Blayney",
    author_email="hugh@drisk.ai",
    description="Read OpenDRIVE files.",
    install_requires=requires,
    extras_require=extras,
    name="pyxodr",
    version="0.1",
    packages=find_packages(
        where=".",
        include=["pyxodr", "pyxodr.*"],
    ),
)
