import os
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
import warnings

import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
import numpy as np  # noqa: E402
import xsdba as sdba  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import seaborn as sns  # noqa: E402
from datetime import datetime  # noqa: E402
from typing import Union, List, Tuple, Optional  # noqa: E402

import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import cartopy.io.shapereader as shpreader  # noqa: E402

# Suppress cartopy download warnings for Natural Earth data
try:
    from cartopy.io import DownloadWarning
    warnings.filterwarnings('ignore', category=DownloadWarning)
except ImportError:
    # Fallback to suppressing all UserWarnings from cartopy.io
    warnings.filterwarnings('ignore', category=UserWarning, module='cartopy.io')

logger = logging.getLogger("climate")
logger.handlers = []  # Remove any existing handlers
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(name)s | %(process)d:%(thread)d [%(levelname)s]: %(message)s"
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
# TODO: Throw an error if the selected country is not in the selected domain
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


def get_climate_data(
    *,
    country: str | None,
    years_obs: range | None = None,
    obs: bool = False,
    cordex_domain: str | None = None,
    rcp: str | None = None,
    gcm: str | None = None,
    rcm: str | None = None,
    years_up_to: int | None = None,
    bias_correction: bool = False,
    historical: bool = False,
    buffer: int = 0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    remote: bool = True,
    variables: list[str] | None = None,
    num_processes: int = len(VALID_VARIABLES),
    max_threads_per_process: int = 8,
) -> dict[str, xr.DataArray]:
    f"""
    Process climate data required by pyAEZ climate module.
    The function automatically access CORDEX-CORE models at 0.25° and the ERA5 datasets.

    Args:
    country (str): Name of the country for which data is to be processed.
        Use None if specifying a region using xlim and ylim.
    years_obs (range): Range of years for observational data (ERA5 only). Required when obs is True. (default: None).
    obs (bool): Flag to indicate if processing observational data (default: False).
        When True, only years_obs is required. CORDEX parameters are optional.
    cordex_domain (str): CORDEX domain of the climate data. One of {VALID_DOMAINS}. 
        Required when obs is False. (default: None).
    rcp (str): Representative Concentration Pathway. One of {VALID_RCPS}. 
        Required when obs is False. (default: None).
    gcm (str): GCM name. One of {VALID_GCM}. 
        Required when obs is False. (default: None).
    rcm (str): RCM name. One of {VALID_RCM}. 
        Required when obs is False. (default: None).
    years_up_to (int): The ending year for the projected data. Projections start in 2006 and ends in 2100.
        Hence, if years_up_to is set to 2030, data will be downloaded for the 2006-2030 period.
        Required when obs is False. (default: None).
    bias_correction (bool): Whether to apply bias correction (default: False).
    historical (bool): Flag to indicate if processing historical data (default: False).
        If True, historical data is provided together with projections.
        Historical simulation runs for CORDEX-CORE initiative are provided for the 1980-2005 time period.
    buffer (int): Buffer distance to expand the region of interest (default: 0).
    xlim (tuple or None): Longitudinal bounds of the region of interest. Use only when country is None (default: None).
    ylim (tuple or None): Latitudinal bounds of the region of interest. Use only when country is None (default: None).
    remote (bool): Flag to work with remote data or not (default: True).
    variables (list[str] or None): List of variables to process. Must be a subset of {VALID_VARIABLES}. If None, all variables are processed. (default: None).
    num_processes (int): Number of processes to use, one per variable.
        By default equals to the number of all possible variables. (default: {len(VALID_VARIABLES)}).
    max_threads_per_process (int): Max number of threads within each process. (default: 8).

    Returns:
    dict: A dictionary containing processed climate data for each variable as an xarray object.
    
    Examples:
    # For observations only:
    data = get_climate_data(country="Togo", obs=True, years_obs=range(1990, 2011))
    
    # For CORDEX projections:
    data = get_climate_data(country="Togo", cordex_domain="AFR-22", rcp="rcp26", 
                           gcm="MPI", rcm="Reg", years_up_to=2030)
    """

    # Validation for basic parameters
    if xlim is None and ylim is not None or xlim is not None and ylim is None:
        raise ValueError(
            "xlim and ylim mismatch: they must be both specified or both unspecified"
        )
    if country is None and xlim is None:
        raise ValueError("You must specify a country or (xlim, ylim)")
    if country is not None and xlim is not None:
        raise ValueError("You must specify either country or (xlim, ylim), not both")

    # Conditional validation based on obs flag
    if obs:
        # When obs=True, only years_obs is required
        if years_obs is None:
            raise ValueError("years_obs must be provided when obs is True")
        if not (1980 <= min(years_obs) <= max(years_obs) <= 2020):
            raise ValueError("Years in years_obs must be within the range 1980 to 2020")
        
        # Set default values for CORDEX parameters (not used but needed for function calls)
        cordex_domain = cordex_domain or "AFR-22"  # dummy value
        rcp = rcp or "rcp26"  # dummy value
        gcm = gcm or "MPI"  # dummy value
        rcm = rcm or "Reg"  # dummy value
        years_up_to = years_up_to or 2030  # dummy value
    else:
        # When obs=False, CORDEX parameters are required
        required_params = {
            "cordex_domain": VALID_DOMAINS,
            "rcp": VALID_RCPS,
            "gcm": VALID_GCM,
            "rcm": VALID_RCM,
        }
        for param_name, valid_values in required_params.items():
            param_value = locals()[param_name]
            if param_value is None:
                raise ValueError(f"{param_name} is required when obs is False")
            if param_value not in valid_values:
                raise ValueError(
                    f"Invalid {param_name}={param_value}. Must be one of {valid_values}"
                )
        
        if years_up_to is None:
            raise ValueError("years_up_to is required when obs is False")
        if years_up_to <= 2006:
            raise ValueError("years_up_to must be greater than 2006")
        
        # Set default years_obs when not processing observations
        if years_obs is None:
            years_obs = DEFAULT_YEARS_OBS

    # Validate variables if provided
    if variables is not None:
        invalid_vars = [var for var in variables if var not in VALID_VARIABLES]
        if invalid_vars:
            raise ValueError(
                f"Invalid variables: {invalid_vars}. Must be a subset of {VALID_VARIABLES}"
            )
    else:
        variables = VALID_VARIABLES

    # Validate GCM-RCM combinations for specific domains (only for non-observational data)
    if not obs:
        _validate_gcm_rcm_combinations(cordex_domain, gcm, rcm)

    _validate_urls(gcm, rcm, rcp, remote, cordex_domain, obs, historical, bias_correction)

    bbox = _geo_localize(country, xlim, ylim, buffer, cordex_domain, obs)

    with mp.Pool(processes=min(num_processes, len(variables))) as pool:
        futures = []
        for variable in variables:
            futures.append(
                pool.apply_async(
                    process_worker,
                    args=(max_threads_per_process,),
                    kwds={
                        "variable": variable,
                        "bbox": bbox,
                        "cordex_domain": cordex_domain,
                        "rcp": rcp,
                        "gcm": gcm,
                        "rcm": rcm,
                        "years_up_to": years_up_to,
                        "years_obs": years_obs,
                        "obs": obs,
                        "bias_correction": bias_correction,
                        "historical": historical,
                        "remote": remote,
                    },
                )
            )

        results = {
            variable: futures[i].get() for i, variable in enumerate(variables)
        }

        pool.close()  # Prevent any more tasks from being submitted to the pool
        pool.join()  # Wait for all worker processes to finish

    return results


def _validate_urls(
    gcm: str = None,
    rcm: str = None,
    rcp: str = None,
    remote: bool = True,
    cordex_domain: str = None,
    obs: bool = False,
    historical: bool = False,
    bias_correction: bool = False,
):
    # Load the data
    log = logger.getChild("URL-validation")

    if obs is False:
        inventory_csv_url = (
            INVENTORY_DATA_REMOTE_URL if remote else INVENTORY_DATA_LOCAL_PATH
        )
        data = pd.read_csv(inventory_csv_url)

        # Set the column to use based on whether the data is remote or local
        column_to_use = "location" if remote else "hub"

        # Define which experiments we need
        experiments = [rcp]
        if historical or bias_correction:
            experiments.append("historical")

        # Filter the data based on the conditions
        filtered_data = data[
            lambda x: (
                x["activity"].str.contains("FAO", na=False)
                & (x["domain"] == cordex_domain)
                & (x["model"].str.contains(gcm, na=False))
                & (x["rcm"].str.contains(rcm, na=False))
                & (x["experiment"].isin(experiments))
            )
        ][["experiment", column_to_use]]

        # Extract the column values as a list
        for _, row in filtered_data.iterrows():
            if row["experiment"] == "historical":
                log_hist = logger.getChild("URL-validation-historical")
                log_hist.info(f"{row[column_to_use]}")
            else:
                log_proj = logger.getChild("URL-validation-projections")
                log_proj.info(f"{row[column_to_use]}")

    else:  # when obs is True
        log_obs = logger.getChild("URL-validation-observations")
        log_obs.info(f"{ERA5_DATA_REMOTE_URL}")


def _get_country_bounds(country_name: str) -> tuple[float, float, float, float]:
    """
    Get country bounding box using cartopy's Natural Earth data.
    
    Args:
        country_name: Name of the country
        
    Returns:
        tuple: (minx, miny, maxx, maxy) bounding box
        
    Raises:
        ValueError: If country not found
    """
    # Use Natural Earth countries dataset via cartopy
    countries_feature = cfeature.NaturalEarthFeature(
        'cultural', 'admin_0_countries', '50m'
    )
    
    # Get the actual shapefile path from the feature
    shapefile_path = countries_feature.with_scale('50m').geometries()
    
    # Search for the country using Natural Earth records
    for country_record in shpreader.Reader(shpreader.natural_earth(resolution='50m', category='cultural', name='admin_0_countries')).records():
        # Try multiple name fields for better matching
        country_names = [
            country_record.attributes.get('NAME', ''),
            country_record.attributes.get('NAME_LONG', ''),
            country_record.attributes.get('ADMIN', ''),
            country_record.attributes.get('NAME_EN', '')
        ]
        
        if any(name.lower() == country_name.lower() for name in country_names if name):
            return country_record.geometry.bounds
    
    # If not found, check for capitalization issue
    if country_name and country_name[0].islower():
        capitalized = country_name.capitalize()
        raise ValueError(f"Country '{country_name}' not found. Try capitalizing the first letter: '{capitalized}'")
    else:
        raise ValueError(f"Country '{country_name}' is unknown.")


def _geo_localize(
    country: str = None,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    buffer: int = 0,
    cordex_domain: str = None,
    obs: bool = False,
) -> dict[str, tuple[float, float]]:
    if country:
        if xlim or ylim:
            raise ValueError(
                "Specify either a country or bounding box limits (xlim, ylim), but not both."
            )
        
        bounds = _get_country_bounds(country)
        xlim, ylim = (bounds[0], bounds[2]), (bounds[1], bounds[3])
    elif not (xlim and ylim):
        raise ValueError(
            "Either a country or bounding box limits (xlim, ylim) must be specified."
        )

    # Apply buffer
    xlim = (xlim[0] - buffer, xlim[1] + buffer)
    ylim = (ylim[0] - buffer, ylim[1] + buffer)

    # Only validate CORDEX domain when processing non-observational data
    # Skip validation for observations or when using dummy values
    if not obs and cordex_domain:
        _validate_cordex_domain(xlim, ylim, cordex_domain)

    return {"xlim": xlim, "ylim": ylim}


def _validate_gcm_rcm_combinations(cordex_domain: str, gcm: str, rcm: str):
    """
    Validate that the GCM-RCM combination is available for the specified CORDEX domain.
    
    Args:
        cordex_domain: CORDEX domain name
        gcm: Global Climate Model name
        rcm: Regional Climate Model name
    
    Raises:
        ValueError: If the combination is not available for the domain
    """
    # Define invalid combinations per domain
    invalid_combinations = {
        "WAS-22": [
            ("MOHC", "Reg")  # MOHC-Reg is not available for WAS-22
        ],
        "CAS-22": [
            ("MOHC", "Reg"),  # Reg is not available for any GCM in CAS-22
            ("MPI", "Reg"),
            ("NCC", "Reg")
        ]
    }
    
    if cordex_domain in invalid_combinations:
        invalid_combos = invalid_combinations[cordex_domain]
        current_combo = (gcm, rcm)
        
        if current_combo in invalid_combos:
            # Get available combinations for this domain
            all_gcm = VALID_GCM
            all_rcm = VALID_RCM
            available_combos = []
            
            for g in all_gcm:
                for r in all_rcm:
                    if (g, r) not in invalid_combos:
                        available_combos.append(f"{g}-{r}")
            
            raise ValueError(
                f"The combination {gcm}-{rcm} is not available for domain {cordex_domain}. "
                f"Available combinations for {cordex_domain}: {', '.join(available_combos)}"
            )


def _validate_cordex_domain(xlim, ylim, cordex_domain):

    # CORDEX domains data
    cordex_domains_df = pd.DataFrame(
        {
            "min_lon": [
                -33,
                -28.3,
                89.25,
                86.75,
                19.25,
                44.0,
                -106.25,
                -115.0,
                -24.25,
                10.75,
            ],
            "min_lat": [
                -28,
                -23,
                -15.25,
                -54.25,
                -15.75,
                -4.0,
                -58.25,
                -14.5,
                -46.25,
                17.75,
            ],
            "max_lon": [
                20,
                18,
                147.0,
                -152.75,
                116.25,
                -172.0,
                -16.25,
                -30.5,
                59.75,
                140.25,
            ],
            "max_lat": [28, 21.7, 26.5, 13.75, 45.75, 65.0, 18.75, 28.5, 42.75, 69.75],
            "cordex_domain": [
                "NAM-22",
                "EUR-22",
                "SEA-22",
                "AUS-22",
                "WAS-22",
                "EAS-22",
                "SAM-22",
                "CAM-22",
                "AFR-22",
                "CAS-22",
            ],
        }
    )

    def is_bbox_contained(bbox, domain):
        """Check if bbox is contained within the domain bounding box."""
        return (
            bbox[0] >= domain["min_lon"]
            and bbox[1] >= domain["min_lat"]
            and bbox[2] <= domain["max_lon"]
            and bbox[3] <= domain["max_lat"]
        )

    user_bbox = [xlim[0], ylim[0], xlim[1], ylim[1]]
    domain_row = cordex_domains_df[cordex_domains_df["cordex_domain"] == cordex_domain]

    if domain_row.empty:
        raise ValueError(f"CORDEX domain '{cordex_domain}' is not recognized.")

    domain_bbox = domain_row.iloc[0]

    if not is_bbox_contained(user_bbox, domain_bbox):
        suggested_domains = cordex_domains_df[
            cordex_domains_df.apply(
                lambda row: is_bbox_contained(user_bbox, row), axis=1
            )
        ]

        if suggested_domains.empty:
            raise ValueError(
                f"The bounding box {user_bbox} is outside of all available CORDEX domains."
            )

        suggested_domain = suggested_domains.iloc[0]["cordex_domain"]

        raise ValueError(
            f"Bounding box {user_bbox} is not within '{cordex_domain}'. Suggested domain: '{suggested_domain}'."
        )


def _leave_one_out_bias_correction(ref, hist, variable, log):
    """
    Perform leave-one-out cross-validation for bias correction to avoid overfitting.
    
    Args:
        ref: Reference (observational) data
        hist: Historical model data
        variable: Variable name for determining correction method
        log: Logger instance
    
    Returns:
        xr.DataArray: Bias-corrected historical data
    """
    log.info("Starting leave-one-out cross-validation for bias correction")
    
    # Get unique years from historical data
    hist_years = hist.time.dt.year.values
    unique_years = np.unique(hist_years)
    
    # Initialize list to store corrected data for each year
    corrected_years = []
    
    for leave_out_year in unique_years:
        log.info(f"Processing leave-out year: {leave_out_year}")
        
        # Create masks for training (all years except leave_out_year) and testing (only leave_out_year)
        train_mask = hist.time.dt.year != leave_out_year
        test_mask = hist.time.dt.year == leave_out_year
        
        # Get training data (all years except the current one)
        hist_train = hist.sel(time=train_mask)
        hist_test = hist.sel(time=test_mask)
        
        # Get corresponding reference data for training period
        ref_train_mask = ref.time.dt.year != leave_out_year
        ref_train = ref.sel(time=ref_train_mask)
        
        # Train the bias correction model on the training data
        QM_leave_out = sdba.EmpiricalQuantileMapping.train(
            ref_train,
            hist_train,
            group="time.month",
            kind="*" if variable in ["pr", "rsds", "sfcWind"] else "+",
        )
        
        # Apply bias correction to the left-out year
        hist_corrected_year = QM_leave_out.adjust(
            hist_test, extrapolation="constant", interp="linear"
        )
        
        # Apply variable-specific constraints
        if variable == "hurs":
            hist_corrected_year = hist_corrected_year.where(hist_corrected_year <= 100, 100)
            hist_corrected_year = hist_corrected_year.where(hist_corrected_year >= 0, 0)
        
        corrected_years.append(hist_corrected_year)
    
    # Concatenate all corrected years and sort by time
    hist_bs = xr.concat(corrected_years, dim="time").sortby("time")
    
    log.info("Leave-one-out cross-validation bias correction completed")
    return hist_bs


def process_worker(num_threads, **kwargs) -> xr.DataArray:
    variable = kwargs["variable"]
    log = logger.getChild(variable)
    try:
        with ThreadPoolExecutor(
            max_workers=num_threads, thread_name_prefix="climate"
        ) as executor:
            return _climate_data_for_variable(executor, **kwargs)
    except Exception as e:
        log.exception(f"Process worker failed: {e}")
        raise


def _climate_data_for_variable(
    executor: ThreadPoolExecutor,
    *,
    variable: str,
    bbox: dict[str, tuple[float, float]],
    cordex_domain: str,
    rcp: str,
    gcm: str,
    rcm: str,
    years_up_to: int,
    years_obs: range,
    obs: bool,
    bias_correction: bool,
    historical: bool,
    remote: bool,
) -> xr.DataArray:
    log = logger.getChild(variable)

    pd.options.mode.chained_assignment = None
    inventory_csv_url = (
        INVENTORY_DATA_REMOTE_URL if remote else INVENTORY_DATA_LOCAL_PATH
    )
    data = pd.read_csv(inventory_csv_url)
    column_to_use = "location" if remote else "hub"
    
    # Filter data based on whether we need historical data
    experiments = [rcp]
    if historical or bias_correction:
        experiments.append("historical")
        
    filtered_data = data[
        lambda x: (x["activity"].str.contains("FAO", na=False))
        & (x["domain"] == cordex_domain)
        & (x["model"].str.contains(gcm, na=False))
        & (x["rcm"].str.contains(rcm, na=False))
        & (x["experiment"].isin(experiments))
    ][["experiment", column_to_use]]

    future_obs = None
    if obs or bias_correction:
        future_obs = executor.submit(
            _thread_download_data,
            url=None,
            bbox=bbox,
            variable=variable,
            obs=True,
            years_up_to=years_up_to,
            years_obs=years_obs,
            remote=remote,
        )

    if not obs:
        download_fn = partial(
            _thread_download_data,
            bbox=bbox,
            variable=variable,
            obs=False,
            years_obs=years_obs,
            years_up_to=years_up_to,
            remote=remote,
        )
        downloaded_models = list(
            executor.map(download_fn, filtered_data[column_to_use])
        )

        # Add the downloaded models to the DataFrame
        filtered_data["models"] = downloaded_models
        
        if historical or bias_correction:
            hist = filtered_data[filtered_data["experiment"] == "historical"]["models"].iloc[0]
            proj = filtered_data[filtered_data["experiment"] == rcp]["models"].iloc[0]
            
            hist = hist.interpolate_na(dim="time", method="linear")
            proj = proj.interpolate_na(dim="time", method="linear")
        else:
            proj = filtered_data["models"].iloc[0]
            proj = proj.interpolate_na(dim="time", method="linear")
            
        if bias_correction and historical:
            # Load observations for bias correction
            ref = future_obs.result()
            log.info("Training eqm with leave-one-out cross-validation")
            
            # Use leave-one-out cross-validation for historical bias correction
            hist_bs = _leave_one_out_bias_correction(ref, hist, variable, log)
            
            # For projections, train on all historical data
            QM_mo = sdba.EmpiricalQuantileMapping.train(
                ref,
                hist,
                group="time.month",
                kind="*" if variable in ["pr", "rsds", "sfcWind"] else "+",
            )
            log.info("Performing bias correction on projections with full historical training")
            proj_bs = QM_mo.adjust(proj, extrapolation="constant", interp="linear")
            log.info("Done!")
            if variable == "hurs":
                proj_bs = proj_bs.where(proj_bs <= 100, 100)
                proj_bs = proj_bs.where(proj_bs >= 0, 0)
            combined = xr.concat([hist_bs, proj_bs], dim="time")
            return combined

        elif not bias_correction and historical:
            combined = xr.concat([hist, proj], dim="time")
            return combined

        elif bias_correction and not historical:
            ref = future_obs.result()
            log.info("Training eqm with historical data")
            QM_mo = sdba.EmpiricalQuantileMapping.train(
                ref,
                hist,
                group="time.month",
                kind="*" if variable in ["pr", "rsds", "sfcWind"] else "+",
            )  # multiplicative approach for pr, rsds and wind speed
            log.info("Performing bias correction with eqm")
            proj_bs = QM_mo.adjust(proj, extrapolation="constant", interp="linear")
            log.info("Done!")
            if variable == "hurs":
                proj_bs = proj_bs.where(proj_bs <= 100, 100)
                proj_bs = proj_bs.where(proj_bs >= 0, 0)
            return proj_bs

        return proj

    else:  # when observations are True
        downloaded_obs = future_obs.result()
        log.info("Done!")
        return downloaded_obs


def _thread_download_data(url: str | None, **kwargs):
    variable = kwargs["variable"]
    temporal = "observations" if kwargs["obs"] else ("historical" if "historical" in str(url) else "projections")
    log = logger.getChild(f"{variable}-{temporal}")
    try:
        return _download_data(url=url, **kwargs)
    except Exception as e:
        log.exception(f"Failed to process data from {url}: {e}")
        raise


def _download_data(
    url: str | None,
    bbox: dict[str, tuple[float, float]],
    variable: str,
    obs: bool,
    years_obs: range,
    years_up_to: int,
    remote: bool,
) -> xr.DataArray:
    temporal = "observations" if obs else ("historical" if url and "historical" in url else "projections")
    log = logger.getChild(f"{variable}-{temporal}")
    
    if obs:
        var = VARIABLES_MAP[variable]
        log.info(f"Establishing connection to ERA5 data for {variable}({var})")
        if remote:
            ds_var = xr.open_dataset(ERA5_DATA_REMOTE_URL)[var]
        else:
            ds_var = xr.open_dataset(ERA5_DATA_LOCAL_PATH)[var]
        log.info(f"Connection to ERA5 data for {variable}({var}) has been established")

        # Coordinate normalization and renaming for 'hurs'
        if var == "hurs":
            ds_var = ds_var.rename({"lat": "latitude", "lon": "longitude"})
            ds_cropped = ds_var.sel(
                longitude=slice(bbox["xlim"][0], bbox["xlim"][1]),
                latitude=slice(bbox["ylim"][0], bbox["ylim"][1]),
            )
        else:
            ds_var.coords["longitude"] = (ds_var.coords["longitude"] + 180) % 360 - 180
            ds_var = ds_var.sortby(ds_var.longitude)
            ds_cropped = ds_var.sel(
                longitude=slice(bbox["xlim"][0], bbox["xlim"][1]),
                latitude=slice(bbox["ylim"][1], bbox["ylim"][0]),
            )

        # Unit conversion
        if var in ["t2mx", "t2mn", "t2m"]:
            ds_cropped -= 273.15  # Convert from Kelvin to Celsius
            ds_cropped.attrs["units"] = "°C"
        elif var == "tp":
            ds_cropped *= 1000  # Convert precipitation
            ds_cropped.attrs["units"] = "mm"
        elif var == "ssrd":
            ds_cropped /= 86400  # Convert from J/m^2 to W/m^2
            ds_cropped.attrs["units"] = "W m-2"
        elif var == "sfcwind":
            ds_cropped = ds_cropped * (
                4.87 / np.log((67.8 * 10) - 5.42)
            )  # Convert wind speed from 10 m to 2 m
            ds_cropped.attrs["units"] = "m s-1"

        # Select years
        years = [x for x in years_obs]
        time_mask = (ds_cropped["time"].dt.year >= years[0]) & (
            ds_cropped["time"].dt.year <= years[-1]
        )

    else:
        log.info(f"Establishing connection to CORDEX data for {variable}")
        ds_var = xr.open_dataset(url)[variable]
        
        # Check if time dimension has a prefix, indicating variable is not available
        time_dims = [dim for dim in ds_var.dims if dim.startswith('time_')]
        if time_dims:
            msg = f"Variable {variable} is not available for this model: {url}"
            log.exception(msg)
            raise ValueError(msg)
            
        log.info(f"Connection to CORDEX data for {variable} has been established")
        ds_cropped = ds_var.sel(
            longitude=slice(bbox["xlim"][0], bbox["xlim"][1]),
            latitude=slice(bbox["ylim"][1], bbox["ylim"][0]),
        )

        # Unit conversion
        if variable in ["tas", "tasmax", "tasmin"]:
            ds_cropped -= 273.15  # Convert from Kelvin to Celsius
            ds_cropped.attrs["units"] = "°C"
        elif variable == "pr":
            ds_cropped *= 86400  # Convert from kg m^-2 s^-1 to mm/day
            ds_cropped.attrs["units"] = "mm"
        elif variable == "rsds":
            ds_cropped.attrs["units"] = "W m-2"
        elif variable == "sfcWind":
            ds_cropped = ds_cropped * (
                4.87 / np.log((67.8 * 10) - 5.42)
            )  # Convert wind speed from 10 m to 2 m
            ds_cropped.attrs["units"] = "m s-1"

        # Select years based on rcp
        if "rcp" in url:
            years = [x for x in range(2006, years_up_to + 1)]
        else:
            years = [x for x in DEFAULT_YEARS_OBS]

        # Add missing dates
        ds_cropped = ds_cropped.convert_calendar(
            calendar="gregorian", missing=np.nan, align_on="date"
        )
    
        time_mask = (ds_cropped["time"].dt.year >= years[0]) & (
            ds_cropped["time"].dt.year <= years[-1]
        )

    # subset years
    ds_cropped = ds_cropped.sel(time=time_mask)

    assert isinstance(ds_cropped, xr.DataArray)

    if obs:
        log.info(
            f"ERA5 data for {variable} has been processed: unit conversion ({ds_cropped.attrs.get('units', 'unknown units')}), time selection ({years[0]}-{years[-1]})"
        )
    else:
        log.info(
            f"CORDEX data for {variable} has been processed: unit conversion ({ds_cropped.attrs.get('units', 'unknown units')}), calendar transformation (360-day to Gregorian), time selection ({years[0]}-{years[-1]})"
        )

    return ds_cropped


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_spatial_map(
    data: xr.DataArray,
    time_period: Optional[Tuple[int, int]] = None,
    aggregation: str = "mean",
    title: Optional[str] = None,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (12, 8),
    show_countries: bool = True,
    save_path: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Create a spatial map visualization of climate data.
    
    Args:
        data (xr.DataArray): Climate data array with latitude/longitude coordinates
        time_period (tuple, optional): (start_year, end_year) to subset data. If None, uses all data
        aggregation (str): Temporal aggregation method ('mean', 'sum', 'min', 'max', 'std')
        title (str, optional): Plot title. If None, auto-generated
        cmap (str): Colormap name
        figsize (tuple): Figure size (width, height)
        show_countries (bool): Whether to show country boundaries
        save_path (str, optional): Path to save the plot
        **kwargs: Additional arguments passed to the plot function
        
    Returns:
        matplotlib.figure.Figure: The created figure
        
    Examples:
        # Plot mean temperature for 2020-2030
        fig = plot_spatial_map(data['tasmax'], time_period=(2020, 2030), 
                              title="Mean Max Temperature 2020-2030")
        
        # Plot precipitation sum with custom colormap
        fig = plot_spatial_map(data['pr'], aggregation='sum', cmap='Blues')
    """
    
    # Subset data by time period if specified
    plot_data = data.copy()
    if time_period is not None:
        start_year, end_year = time_period
        plot_data = plot_data.sel(
            time=slice(f"{start_year}-01-01", f"{end_year}-12-31")
        )
    
    # Apply temporal aggregation
    if aggregation == "mean":
        plot_data = plot_data.mean(dim="time")
    elif aggregation == "sum":
        plot_data = plot_data.sum(dim="time")
    elif aggregation == "min":
        plot_data = plot_data.min(dim="time")
    elif aggregation == "max":
        plot_data = plot_data.max(dim="time")
    elif aggregation == "std":
        plot_data = plot_data.std(dim="time")
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation}")
    
    # Create figure with cartopy
    fig, ax = plt.subplots(
        figsize=figsize, 
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    
    # Plot data
    im = plot_data.plot(
        ax=ax,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        **kwargs
    )
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    if show_countries:
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.7)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
    
    # Set extent to data bounds with small buffer
    lon_min, lon_max = plot_data.longitude.min().item(), plot_data.longitude.max().item()
    lat_min, lat_max = plot_data.latitude.min().item(), plot_data.latitude.max().item()
    buffer = 0.5
    ax.set_extent([lon_min - buffer, lon_max + buffer, 
                  lat_min - buffer, lat_max + buffer], ccrs.PlateCarree())
    
    # Add gridlines with labels only on left and bottom
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False     # No labels on top
    gl.right_labels = False   # No labels on right
    gl.left_labels = True     # Labels on left (latitude)
    gl.bottom_labels = True   # Labels on bottom (longitude)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    if hasattr(plot_data, 'units'):
        cbar.set_label(f"{plot_data.name} ({plot_data.units})", rotation=270, labelpad=20)
    else:
        cbar.set_label(f"{plot_data.name}", rotation=270, labelpad=20)
    
    # Set title
    if title is None:
        var_name = plot_data.name or "Climate Variable"
        if time_period:
            title = f"{aggregation.title()} {var_name} ({time_period[0]}-{time_period[1]})"
        else:
            title = f"{aggregation.title()} {var_name}"
    
    ax.set_title(title, fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_time_series(
    data: Union[xr.DataArray, List[xr.DataArray]],
    aggregation: str = "mean",
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    trend_line: bool = False,
    save_path: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Create time series plots of climate data.
    
    Args:
        data (xr.DataArray or list): Single DataArray or list of DataArrays to plot
        aggregation (str): Spatial aggregation method ('mean', 'sum', 'min', 'max', 'std')
        labels (list, optional): Labels for multiple datasets
        title (str, optional): Plot title
        ylabel (str, optional): Y-axis label
        figsize (tuple): Figure size
        trend_line (bool): Whether to add trend line
        save_path (str, optional): Path to save the plot
        **kwargs: Additional arguments passed to plot function
        
    Returns:
        matplotlib.figure.Figure: The created figure
        
    Examples:
        # Plot single time series
        fig = plot_time_series(data['tasmax'], title="Temperature Trends")
        
        # Compare multiple scenarios
        fig = plot_time_series([data1['pr'], data2['pr']], 
                              labels=['RCP2.6', 'RCP8.5'], 
                              title="Precipitation Comparison")
    """
    
    # Ensure data is a list
    if isinstance(data, xr.DataArray):
        data_list = [data]
        labels = labels or [data.name or "Data"]
    else:
        data_list = data
        labels = labels or [f"Dataset {i+1}" for i in range(len(data_list))]
    
    if len(data_list) != len(labels):
        raise ValueError("Number of labels must match number of datasets")
    
    # Set up the plot
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Process and plot each dataset
    for i, (dataset, label) in enumerate(zip(data_list, labels)):
        # Apply spatial aggregation
        if aggregation == "mean":
            ts_data = dataset.mean(dim=["latitude", "longitude"])
        elif aggregation == "sum":
            ts_data = dataset.sum(dim=["latitude", "longitude"])
        elif aggregation == "min":
            ts_data = dataset.min(dim=["latitude", "longitude"])
        elif aggregation == "max":
            ts_data = dataset.max(dim=["latitude", "longitude"])
        elif aggregation == "std":
            ts_data = dataset.std(dim=["latitude", "longitude"])
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")
        
        # Convert to annual means for cleaner plotting
        annual_data = ts_data.groupby("time.year").mean()
        
        # Plot the time series
        ax1.plot(annual_data.year, annual_data.values, label=label, linewidth=2, **kwargs)
        
        # Add trend line if requested
        if trend_line:
            z = np.polyfit(annual_data.year, annual_data.values, 1)
            p = np.poly1d(z)
            ax1.plot(annual_data.year, p(annual_data.year), 
                    linestyle='--', alpha=0.7, 
                    color=ax1.lines[-1].get_color())
    
    # Format main plot
    ax1.set_xlabel("Year", fontsize=12)
    if ylabel is None:
        if hasattr(data_list[0], 'units'):
            ylabel = f"{data_list[0].name} ({data_list[0].units})"
        else:
            ylabel = data_list[0].name or "Value"
    ax1.set_ylabel(ylabel, fontsize=12)
    
    if len(data_list) > 1:
        ax1.legend()
    
    ax1.grid(True, alpha=0.3)
    
    # Set main title
    if title is None:
        var_name = data_list[0].name or "Climate Variable"
        title = f"{aggregation.title()} {var_name} Time Series"
    
    ax1.set_title(title, fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig



if __name__ == "__main__":
    # Example 1: Get observational data
    print("Getting observational data...")
    obs_data = get_climate_data(
        country="Togo",
        obs=True,
        years_obs=range(1990, 2011),
        variables=["pr", "tasmax"]
    )
    print("Observational data keys:", list(obs_data.keys()))
    
    # Example 2: Get CORDEX bc projection data and bc historical data
    print("\nGetting CORDEX projection data...")
    proj_data = get_climate_data(
        country="Togo",
        variables=["tasmax", "tasmin"],
        cordex_domain="AFR-22",
        rcp="rcp26",
        gcm="MPI",
        rcm="Reg",
        years_up_to=2010,
        historical=True,
        bias_correction=True
    )
    print("Projection data keys:", list(proj_data.keys()))
    
    # Example 3: Test new country lookup functionality
    print("\nTesting country lookup functionality...")
    try:
        # Test cartopy-based country lookup
        bounds = _get_country_bounds("Togo")
        print(f"Country lookup successful - Togo bounds: {bounds}")
    except Exception as e:
        print(f"Country lookup failed: {e}")
    
    # Example 4: Plotting demonstrations (commented out to avoid blocking)
    print("\nPlotting functionality is available!")
    print("Use plot_spatial_map() and plot_time_series() functions")
    
    # Uncomment these lines to see actual plots:
    # fig1 = plot_spatial_map(obs_data['tasmax'], time_period=(2000, 2010), 
    #                        title="Mean Max Temperature 2000-2010 (ERA5)", cmap="Reds")
    # fig2 = plot_time_series(obs_data['pr'], title="Precipitation Time Series (Togo)", 
    #                        trend_line=True)
    
    print("Example completed successfully!")
