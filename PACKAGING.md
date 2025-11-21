# Build and publish a PyPi package

First, create an account and a Token at pypi.org.

Configure PyPi Credentials:
```
poetry config pypi-token.pypi your-pypi-token
```
Replace your-pypi-token with the token you obtained from PyPI.

Publish to PyPI: With your credentials configured, publish your package:
```
poetry publish --build
```

Verify the package in a clean python environment:
```
conda create -n test python=3.11
conda activate test
pip install cavapy
python -c 'import cavapy; cavapy.get_climate_data(country="Zambia", cordex_domain="AFR-22", rcp="rcp26", gcm="MPI", rcm="REMO", years_up_to=2030, obs=False, bias_correction=True, historical=False)'
```
