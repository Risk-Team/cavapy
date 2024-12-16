from setuptools import setup

setup(
    name="cava_data",
    version="0.1.0",
    py_modules=["climate"],
    install_requires=[
        "xclim==0.53.2",
        "netCDF4=1.7.2",
        "geopandas==0.14.4"
    ],
    description="A python library to facilitate access to CAVA data",
    author="Riccardo Soldan, Oleh Lokshyn",
    author_email="riccardosoldan@hotmail.it",
    url="https://github.com/Risk-Team/cava_data/tree/main",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
