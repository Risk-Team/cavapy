"""Configuration constants and logging setup for cavapy."""

import os
import logging
import warnings

# Suppress cartopy download warnings for Natural Earth data
try:
    from cartopy.io import DownloadWarning
    warnings.filterwarnings("ignore", category=DownloadWarning)
except ImportError:
    # Fallback to suppressing all UserWarnings from cartopy.io
    warnings.filterwarnings("ignore", category=UserWarning, module="cartopy.io")

logger = logging.getLogger("climate")
logger.handlers = []  # Remove any existing handlers
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
handler.setFormatter(formatter)
for hdlr in logger.handlers[:]:  # remove all old handlers
    logger.removeHandler(hdlr)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

VARIABLES_MAP = {
    "pr": "tp",
    "tasmax": "t2mx",
    "tasmin": "t2mn",
    "hurs": "hurs",
    "sfcWind": "sfcwind",
    "rsds": "ssrd",
}
VALID_VARIABLES = list(VARIABLES_MAP)
VALID_DOMAINS = [
    "NAM-22",
    "EUR-22",
    "AFR-22",
    "EAS-22",
    "SEA-22",
    "WAS-22",
    "AUS-22",
    "SAM-22",
    "CAM-22",
]
VALID_RCPS = ["rcp26", "rcp85"]
VALID_GCM = ["MOHC", "MPI", "NCC"]
VALID_RCM = ["REMO", "Reg"]
VALID_DATASETS = ["CORDEX-CORE", "CORDEX-CORE-BC"]

INVENTORY_DATA_REMOTE_URL = (
    "https://hub.ipcc.ifca.es/thredds/fileServer/inventories/cava.csv"
)
INVENTORY_DATA_LOCAL_PATH = os.path.join(
    os.path.expanduser("~"), "shared/inventories/cava/inventory.csv"
)
ERA5_DATA_REMOTE_URL = (
    "https://hub.ipcc.ifca.es/thredds/dodsC/fao/observations/ERA5/0.25/ERA5_025.ncml"
)
ERA5_DATA_LOCAL_PATH = os.path.join(
    os.path.expanduser("~"), "shared/data/observations/ERA5/0.25/ERA5_025.ncml"
)
DEFAULT_YEARS_OBS = range(1980, 2006)
