import os
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*geopandas.dataset module is deprecated.*",
)
import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
import numpy as np  # noqa: E402
from xclim import sdba  # noqa: E402


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


class VariableNotAvailableError(Exception):
    """Raised when a requested climate variable is not available in the model."""
    pass


def get_climate_data(
    *,
    country: str | None,
    cordex_domain: str,
    rcp: str,
    gcm: str,
    rcm: str,
    years_up_to: int,
    years_obs: range | None = None,
    bias_correction: bool = False,
    historical: bool = False,
    obs: bool = False,
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
    cordex_domain (str): CORDEX domain of the climate data. One of {VALID_DOMAINS}.
    rcp (str): Representative Concentration Pathway. One of {VALID_RCPS}.
    gcm (str): GCM name. One of {VALID_GCM}.
    rcm (str): RCM name. One of {VALID_RCM}.
    years_up_to (int): The ending year for the projected data. Projections start in 2006 and ends in 2100.
        Hence, if years_up_to is set to 2030, data will be downloaded for the 2006-2030 period.
    years_obs (range): Range of years for observational data (ERA5 only). Only used when obs is True. (default: None).
    bias_correction (bool): Whether to apply bias correction (default: False).
    historical (bool): Flag to indicate if processing historical data (default: False).
        If True, historical data is provided together with projections.
        Historical simulation runs for CORDEX-CORE initiative are provided for the 1980-2005 time period.
    obs (bool): Flag to indicate if processing observational data (default: False).
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
    """

    if xlim is None and ylim is not None or xlim is not None and ylim is None:
        raise ValueError(
            "xlim and ylim mismatch: they must be both specified or both unspecified"
        )
    if country is None and xlim is None:
        raise ValueError("You must specify a country or (xlim, ylim)")
    if country is not None and xlim is not None:
        raise ValueError("You must specify either country or (xlim, ylim), not both")
    verify_variables = {
        "cordex_domain": VALID_DOMAINS,
        "rcp": VALID_RCPS,
        "gcm": VALID_GCM,
        "rcm": VALID_RCM,
    }
    for var_name, valid_values in verify_variables.items():
        var_value = locals()[var_name]
        if var_value not in valid_values:
            raise ValueError(
                f"Invalid {var_name}={var_value}. Must be one of {valid_values}"
            )
    if years_up_to <= 2006:
        raise ValueError("years_up_to must be greater than 2006")
    if years_obs is not None and not (1980 <= min(years_obs) <= max(years_obs) <= 2020):
        raise ValueError("Years in years_obs must be within the range 1980 to 2020")
    if obs and years_obs is None:
        raise ValueError("years_obs must be provided when obs is True")
    if not obs or years_obs is None:
        # Make sure years_obs is set to default when obs=False
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

    _validate_urls(gcm, rcm, rcp, remote, cordex_domain, obs, historical, bias_correction)

    bbox = _geo_localize(country, xlim, ylim, buffer, cordex_domain)

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

        # Try to get the first result - if it fails, terminate all processes
        try:
            first_result = futures[0].get()
            # If first result succeeded, try to get the rest
            results = {variables[0]: first_result}
            for i, future in enumerate(futures[1:], 1):
                try:
                    results[variables[i]] = future.get()
                except Exception as e:
                    pool.terminate()
                    raise e
        except Exception as e:
            pool.terminate()
            raise e
        finally:
            pool.close()
            pool.join()

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


def _geo_localize(
    country: str = None,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    buffer: int = 0,
    cordex_domain: str = None,
) -> dict[str, tuple[float, float]]:
    if country:
        if xlim or ylim:
            raise ValueError(
                "Specify either a country or bounding box limits (xlim, ylim), but not both."
            )
        # Load country shapefile and extract bounds
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        country_shp = world[world.name == country]
        if country_shp.empty:
            raise ValueError(f"Country '{country}' is unknown.")
        bounds = country_shp.total_bounds  # [minx, miny, maxx, maxy]
        xlim, ylim = (bounds[0], bounds[2]), (bounds[1], bounds[3])
    elif not (xlim and ylim):
        raise ValueError(
            "Either a country or bounding box limits (xlim, ylim) must be specified."
        )

    # Apply buffer
    xlim = (xlim[0] - buffer, xlim[1] + buffer)
    ylim = (ylim[0] - buffer, ylim[1] + buffer)

    # Always validate CORDEX domain
    if cordex_domain:
        _validate_cordex_domain(xlim, ylim, cordex_domain)

    return {"xlim": xlim, "ylim": ylim}


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
        if "is not available in model configuration" in str(e):
            raise VariableNotAvailableError(str(e)) from e
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
            log.info("Training eqm with historical data")
            QM_mo = sdba.EmpiricalQuantileMapping.train(
                ref,
                hist,
                group="time.month",
                kind="*" if variable in ["pr", "rsds", "sfcWind"] else "+",
            )
            log.info("Performing bias correction with eqm")
            hist_bs = QM_mo.adjust(hist, extrapolation="constant", interp="linear")
            proj_bs = QM_mo.adjust(proj, extrapolation="constant", interp="linear")
            log.info("Done!")
            if variable == "hurs":
                hist_bs = hist_bs.where(hist_bs <= 100, 100)
                hist_bs = hist_bs.where(hist_bs >= 0, 0)
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
        
        # Check if time dimension has a prefix, indicating variable is not available. This is a fix implemented by Ezi
        time_dims = [dim for dim in ds_var.dims if dim.startswith('time_')]
        if time_dims:
            # Extract GCM and RCM from URL
            model_info = url.split('/')[-1]  # Get filename from URL
            msg = f"Variable {variable} is not available in model configuration: {model_info}. You can rerun the function excluding {variable} from the variables list."
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


if __name__ == "__main__":
    data = get_climate_data(
        country="Togo",
        variables=["tasmax","hurs"],
        cordex_domain="AFR-22",
        rcp="rcp26",
        gcm="MPI",
        rcm="Reg",
        years_up_to=2030,
        obs=False,
        bias_correction=False,
        historical=False,
    )
    print(data)

