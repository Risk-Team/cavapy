"""Download and post-process ERA5/CORDEX data for the CAVA pipeline."""

import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pandas as pd
import numpy as np
import xarray as xr
import xsdba as sdba

from cava_bias import _leave_one_out_bias_correction
from cava_config import (
    DEFAULT_YEARS_OBS,
    ERA5_DATA_LOCAL_PATH,
    ERA5_DATA_REMOTE_URL,
    INVENTORY_DATA_LOCAL_PATH,
    INVENTORY_DATA_REMOTE_URL,
    VARIABLES_MAP,
    logger,
)
from cava_validation import _ensure_inventory_not_empty


def process_worker(num_threads, **kwargs) -> xr.DataArray:
    """Run per-variable processing inside a thread pool and return the result."""
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
    dataset: str = "CORDEX-CORE",
) -> xr.DataArray:
    """Fetch and process one variable, optionally bias-correcting and merging runs."""
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

    # Determine activity filter based on dataset
    activity_filter = "FAO" if dataset == "CORDEX-CORE" else "CRDX-ISIMIP-025"

    filtered_data = data[
        lambda x: (x["activity"].str.contains(activity_filter, na=False))
        & (x["domain"] == cordex_domain)
        & (x["model"].str.contains(gcm, na=False))
        & (x["rcm"].str.contains(rcm, na=False))
        & (x["experiment"].isin(experiments))
    ][["experiment", column_to_use]]

    # Fail early if nothing is found
    _ensure_inventory_not_empty(
        filtered_data,
        dataset=dataset,
        cordex_domain=cordex_domain,
        gcm=gcm,
        rcm=rcm,
        experiments=experiments,
        activity_filter=activity_filter,
        log=log,
    )

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
            hist = filtered_data[filtered_data["experiment"] == "historical"][
                "models"
            ].iloc[0]
            proj = filtered_data[filtered_data["experiment"] == rcp]["models"].iloc[
                0
            ]

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

            # Apply variable-specific constraints
            if variable == "hurs":
                hist_bs = hist_bs.where(hist_bs <= 100, 100)
                hist_bs = hist_bs.where(hist_bs >= 0, 0)
                proj_bs = proj_bs.where(proj_bs <= 100, 100)
                proj_bs = proj_bs.where(proj_bs >= 0, 0)

            return xr.concat([hist_bs, proj_bs], dim="time")

        elif not bias_correction and historical:
            return xr.concat([hist, proj], dim="time")

        elif bias_correction and not historical:
            # Load observations for bias correction
            ref = future_obs.result()
            log.info("Performing bias correction with eqm")
            QM_mo = sdba.EmpiricalQuantileMapping.train(
                ref,
                proj,
                group="time.month",
                kind="*" if variable in ["pr", "rsds", "sfcWind"] else "+",
            )
            proj_bs = QM_mo.adjust(proj, extrapolation="constant", interp="linear")

            # Apply variable-specific constraints
            if variable == "hurs":
                proj_bs = proj_bs.where(proj_bs <= 100, 100)
                proj_bs = proj_bs.where(proj_bs >= 0, 0)

            return proj_bs

        else:
            return proj

    return future_obs.result()


def _thread_download_data(url: str | None, **kwargs):
    """Thread entrypoint to download a single dataset with logging and error handling."""
    variable = kwargs["variable"]
    temporal = (
        "observations"
        if kwargs["obs"]
        else ("historical" if url and "historical" in url else "projections")
    )
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
    """Download a dataset, subset it to the bbox, and perform unit/calendar handling."""
    temporal = (
        "observations"
        if obs
        else ("historical" if url and "historical" in url else "projections")
    )
    log = logger.getChild(f"{variable}-{temporal}")

    def _reindex_daily(data: xr.DataArray) -> xr.DataArray:
        """Reindex data to a daily time axis, filling missing dates with NaN."""
        time_coord = data["time"]
        if time_coord.size == 0:
            return data
        time_index = time_coord.to_index()
        if not time_index.is_monotonic_increasing:
            data = data.sortby("time")
            time_coord = data["time"]
            time_index = time_coord.to_index()
        start = time_coord.values[0]
        end = time_coord.values[-1]
        if np.issubdtype(time_coord.dtype, np.datetime64):
            full_time = pd.date_range(start=start, end=end, freq="D")
        else:
            calendar = (
                time_coord.encoding.get("calendar")
                or time_coord.attrs.get("calendar")
                or "standard"
            )
            full_time = xr.cftime_range(
                start=start, end=end, freq="D", calendar=calendar
            )
        if len(full_time) != time_coord.size:
            log.warning(
                "Reindexing time to daily range (%d steps); filling %d missing dates with NaN.",
                len(full_time),
                len(full_time) - time_coord.size,
            )
        return data.reindex(time=full_time)

    def _open_dataset_with_retry(
        url_or_path: str, *, retries: int = 3, delay_s: float = 2.0
    ) -> xr.Dataset:
        """Open a dataset with retries to mitigate transient network failures."""
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                ds = xr.open_dataset(url_or_path)
                if not ds.data_vars:
                    raise ValueError("Dataset opened with no data variables")
                return ds
            except Exception as exc:
                last_exc = exc
                if attempt == retries:
                    break
                log.warning(
                    f"open_dataset failed (attempt {attempt}/{retries}) for {url_or_path}: {exc}. "
                    f"Retrying in {delay_s:.1f}s."
                )
                time.sleep(delay_s)
        assert last_exc is not None
        raise last_exc

    if obs:
        var = VARIABLES_MAP[variable]
        log.info(f"Establishing connection to ERA5 data for {variable}({var})")
        if remote:
            ds_var = _open_dataset_with_retry(ERA5_DATA_REMOTE_URL)[var]
        else:
            ds_var = _open_dataset_with_retry(ERA5_DATA_LOCAL_PATH)[var]
        log.info(f"Connection to ERA5 data for {variable}({var}) has been established")

        # Coordinate normalization and renaming for 'hurs'
        if var == "hurs":
            ds_var = ds_var.rename({"lat": "latitude", "lon": "longitude"})
            # Normalize latitude order to match other ERA5 variables (descending)
            ds_var = ds_var.sortby("latitude", ascending=False)
            ds_cropped = ds_var.sel(
                longitude=slice(bbox["xlim"][0], bbox["xlim"][1]),
                latitude=slice(bbox["ylim"][1], bbox["ylim"][0]),
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
        ds = _open_dataset_with_retry(url)
        if variable == "sfcWind":
            dataset_var = "sfcwind" if "sfcwind" in ds.variables else "sfcWind"
        else:
            dataset_var = variable
        ds_var = ds[dataset_var]

        # Check if time dimension has a prefix, indicating variable is not available
        time_dims = [dim for dim in ds_var.dims if dim.startswith("time_")]
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
        try:
            ds_cropped = ds_cropped.convert_calendar(
                calendar="gregorian", missing=np.nan, align_on="date"
            )
        except ValueError as exc:
            msg = str(exc)
            if "date_range_like" in msg and "frequency was not inferable" in msg:
                log.warning(
                    "Time frequency not inferable; filling missing dates before calendar conversion."
                )
                ds_cropped = _reindex_daily(ds_cropped)
                ds_cropped = ds_cropped.convert_calendar(
                    calendar="gregorian", missing=np.nan, align_on="date"
                )
                ds_cropped = _reindex_daily(ds_cropped)
            else:
                raise

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
