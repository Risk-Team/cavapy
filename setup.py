from setuptools import setup, find_packages

setup(
    name='cavapy',
    version='0.1.0',
    packages=find_packages(),  # Automatically discover all packages in the current directory
    install_requires=[
        'xclim>=0.53.2',
        'netcdf4==1.7.2',
        'geopandas==0.14.4',
    ],
    entry_points={
        'console_scripts': [
            'climate = climate:get_climate_data',  # If you have a `main` function in climate.py, this will make it callable
        ],
    },
    package_data={
        '': ['climate.py'],  # Include climate.py in the package
    }
)
