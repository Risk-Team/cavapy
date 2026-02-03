"""CAVA Python package for retrieving and visualizing climate data."""

__version__ = "2.0.0"

from .cavapy import get_climate_data
from .cava_plot import plot_spatial_map, plot_time_series
from .cava_config import (
    VALID_DOMAINS,
    VALID_GCM,
    VALID_RCM,
    VALID_RCPS,
    VALID_VARIABLES,
)

__all__ = [
    "get_climate_data",
    "plot_spatial_map",
    "plot_time_series",
    "VALID_DOMAINS",
    "VALID_GCM",
    "VALID_RCM",
    "VALID_RCPS",
    "VALID_VARIABLES",
]
