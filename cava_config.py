"""Configuration constants and logging setup for cavapy."""

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
formatter = logging.Formatter(
    "%(asctime)s | %(process)d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger.propagate = False

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
    "CAS-22",
]
VALID_RCPS = ["rcp26", "rcp85"]
VALID_GCM = ["MOHC", "MPI", "NCC"]
VALID_RCM = ["REMO", "Reg"]
VALID_DATASETS = ["CORDEX-CORE", "CORDEX-CORE-BC"]

INVENTORY_DATA_REMOTE_URL = (
    "https://hub.ipcc.ifca.es/thredds/fileServer/inventories/cava.csv"
)
ERA5_DATA_REMOTE_URL = (
    "https://hub.ipcc.ifca.es/thredds/dodsC/fao/observations/ERA5/0.25/ERA5_025.ncml"
)
DEFAULT_YEARS_OBS = range(1980, 2006)

# THREDDS connection throttling configuration
MAX_CONCURRENT_CONNECTIONS = 8  # Limit concurrent OpenDAP connections to avoid server overload
RETRY_BASE_DELAY_S = 2.0        # Initial retry delay in seconds
RETRY_BACKOFF_FACTOR = 2.0      # Exponential backoff multiplier (2s -> 4s -> 8s)
RETRY_MAX_ATTEMPTS = 3          # Maximum retry attempts for transient failures
