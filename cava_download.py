"""
Download and post-process ERA5/CORDEX data for the CAVA pipeline.

Optimized for THREDDS/OpenDAP:
1. Batch variable extraction - single OpenDAP request per dataset
2. CPU-bound post-processing parallelized with ThreadPoolExecutor
3. Reduced HTTP round-trips to THREDDS server
"""

import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import xarray as xr
import xsdba as sdba

from cava_bias import _leave_one_out_bias_correction
from cava_config import (
    DEFAULT_YEARS_OBS,
    ERA5_DATA_REMOTE_URL,
    INVENTORY_DATA_REMOTE_URL,
    MAX_CONCURRENT_CONNECTIONS,
    RETRY_BACKOFF_FACTOR,
    RETRY_BASE_DELAY_S,
    RETRY_MAX_ATTEMPTS,
    VARIABLES_MAP,
    logger,
)
from cava_validation import _ensure_inventory_not_empty

# Module-level semaphore to limit concurrent THREDDS/OpenDAP connections.
# This prevents overwhelming the server when running multiple parallel processes.
_THREDDS_CONNECTION_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_CONNECTIONS)


def process_worker(
    num_threads: int,
    **kwargs,
) -> dict[str, xr.DataArray]:
    """
    Optimized worker that:
    1. Opens datasets with threading (I/O bound)
    2. Batch-extracts all variables in one OpenDAP request per dataset
    3. Parallelizes post-processing (CPU bound)

    Args:
        num_threads: Number of threads for the ThreadPoolExecutor.
    """
    log = logger.getChild("variables")

    try:
        with ThreadPoolExecutor(
            max_workers=num_threads, thread_name_prefix="climate"
        ) as executor:
            return _climate_data_batch_optimized(executor, **kwargs)
    except Exception as e:
        log.exception(f"Process worker failed: {e}")
        raise


def _climate_data_batch_optimized(
    executor: ThreadPoolExecutor,
    *,
    variables: list[str],
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
    dataset: str = "CORDEX-CORE",
) -> dict[str, xr.DataArray]:
    """
    Optimized fetch: batch extraction + parallel post-processing.

    Flow:
    1. Open datasets concurrently (I/O bound - use threads)
    2. BATCH extract all variables + spatial subset in ONE OpenDAP call per dataset
    3. Post-process each variable in parallel (CPU bound - use threads)
    """
    log = logger.getChild("inventory")
    pd.options.mode.chained_assignment = None
    data = pd.read_csv(INVENTORY_DATA_REMOTE_URL)
    column_to_use = "location"

    experiments = [rcp]
    if historical or bias_correction:
        experiments.append("historical")

    activity_filter = "FAO" if dataset == "CORDEX-CORE" else "CRDX-ISIMIP-025"

    filtered_data = data[
        lambda x: (x["activity"].str.contains(activity_filter, na=False))
        & (x["domain"] == cordex_domain)
        & (x["model"].str.contains(gcm, na=False))
        & (x["rcm"].str.contains(rcm, na=False))
        & (x["experiment"].isin(experiments))
    ][["experiment", column_to_use]]

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

    # =========================================================================
    # PHASE 1: Open datasets concurrently (I/O bound)
    # =========================================================================
    obs_future = None
    hist_future = None
    proj_future = None

    if obs or bias_correction:
        obs_future = executor.submit(_open_dataset_with_retry, ERA5_DATA_REMOTE_URL)

    if not obs:
        if historical or bias_correction:
            hist_url = filtered_data[filtered_data["experiment"] == "historical"][
                column_to_use
            ].iloc[0]
            proj_url = filtered_data[filtered_data["experiment"] == rcp][
                column_to_use
            ].iloc[0]
            hist_future = executor.submit(_open_dataset_with_retry, hist_url)
            proj_future = executor.submit(_open_dataset_with_retry, proj_url)
        else:
            proj_url = filtered_data[column_to_use].iloc[0]
            proj_future = executor.submit(_open_dataset_with_retry, proj_url)

    # =========================================================================
    # PHASE 2: Batch extract ALL variables in ONE OpenDAP request per dataset
    # This is the key optimization - reduces HTTP round-trips significantly
    # =========================================================================
    obs_ds = obs_future.result() if obs_future else None
    hist_ds = hist_future.result() if hist_future else None
    proj_ds = proj_future.result() if proj_future else None

    model_label = f"{gcm}-{rcm} {rcp}"

    # Build list of extraction tasks
    extraction_tasks = []
    if obs_ds and (obs or bias_correction):
        extraction_tasks.append(("ERA5", obs_ds, True, None))
    if hist_ds:
        extraction_tasks.append(("historical", hist_ds, False, False))
    if proj_ds:
        extraction_tasks.append(("projection", proj_ds, False, True))

    # Batch extract with spatial subsetting - ONE request per dataset
    obs_batch, hist_batch, proj_batch = {}, {}, {}

    for name, ds, is_obs_flag, is_proj in extraction_tasks:
        batch = _batch_extract_variables(
            ds,
            variables,
            bbox,
            is_obs=is_obs_flag,
            years_obs=years_obs,
            years_up_to=years_up_to,
            is_projection=is_proj,
            label=f"{model_label} {name}",
        )
        if name == "ERA5":
            obs_batch = batch
        elif name == "historical":
            hist_batch = batch
        else:
            proj_batch = batch

    # =========================================================================
    # PHASE 3: Post-process each variable in parallel (CPU bound)
    # Unit conversion, calendar conversion, bias correction
    # =========================================================================
    results: dict[str, xr.DataArray] = {}

    # Submit all post-processing tasks
    futures = {}
    for variable in variables:
        future = executor.submit(
            _postprocess_variable,
            variable=variable,
            obs_data=obs_batch.get(variable),
            hist_data=hist_batch.get(variable),
            proj_data=proj_batch.get(variable),
            obs_only=obs,
            bias_correction=bias_correction,
            historical=historical,
            years_obs=years_obs,
        )
        futures[future] = variable

    # Collect results as they complete
    for future in as_completed(futures):
        variable = futures[future]
        try:
            results[variable] = future.result()
        except Exception as exc:
            raise RuntimeError(
                f"Variable '{variable}' post-processing failed for {gcm}-{rcm} {rcp}"
            ) from exc

    return results


def _batch_extract_variables(
    ds: xr.Dataset,
    variables: list[str],
    bbox: dict[str, tuple[float, float]],
    *,
    is_obs: bool,
    years_obs: range,
    years_up_to: int,
    is_projection: bool | None = None,
    label: str = "",
) -> dict[str, xr.DataArray]:
    """
    Extract ALL variables from a dataset in a single operation.

    This is the key optimization: instead of ds[var].sel(...) for each variable,
    we do ds[vars].sel(...) once, which translates to a single OpenDAP request.
    """
    log = logger.getChild("batch_extract")
    log_prefix = f"[{label}] " if label else ""

    if is_obs:
        # Map variable names for ERA5
        # ERA5 has variable-specific coordinate handling, so we extract individually
        # but still benefit from the dataset being already open
        result = {}
        for variable in variables:
            var_name = VARIABLES_MAP[variable]
            if var_name not in ds.data_vars:
                continue

            ds_var = ds[var_name]

            # Handle coordinate differences
            if var_name == "hurs":
                ds_var = ds_var.rename({"lat": "latitude", "lon": "longitude"})
                ds_var = ds_var.sortby("latitude", ascending=False)
            else:
                ds_var.coords["longitude"] = (ds_var.coords["longitude"] + 180) % 360 - 180
                ds_var = ds_var.sortby(ds_var.longitude)

            # Spatial subset
            cropped = ds_var.sel(
                longitude=slice(bbox["xlim"][0], bbox["xlim"][1]),
                latitude=slice(bbox["ylim"][1], bbox["ylim"][0]),
            )

            # Time filter for obs
            years = list(years_obs)
            time_mask = (cropped["time"].dt.year >= years[0]) & (
                cropped["time"].dt.year <= years[-1]
            )
            # .load() fetches data into memory - critical for thread safety
            result[variable] = cropped.sel(time=time_mask).load()

        return result

    else:
        # CORDEX data - can do true batch extraction
        # Map variable names
        var_map = {}
        for v in variables:
            if v == "sfcWind":
                dataset_var = "sfcwind" if "sfcwind" in ds.variables else "sfcWind"
            else:
                dataset_var = v
            if dataset_var in ds.data_vars:
                var_map[v] = dataset_var

        if not var_map:
            log.warning("No requested variables found in CORDEX dataset")
            return {}

        # Check for problematic time dimensions
        for v, dv in list(var_map.items()):
            time_dims = [dim for dim in ds[dv].dims if dim.startswith("time_")]
            if time_dims:
                log.warning(f"Variable {v} has problematic time dimension, skipping")
                del var_map[v]

        # BATCH spatial subsetting - this is the key optimization
        # Select all variables at once with spatial bounds
        dataset_vars = list(var_map.values())

        # Single OpenDAP request for all variables + spatial subset
        # .load() fetches data into memory - critical for thread safety
        log.info(f"{log_prefix}Batch extracting {len(dataset_vars)} variables with spatial subset...")
        subset = ds[dataset_vars].sel(
            longitude=slice(bbox["xlim"][0], bbox["xlim"][1]),
            latitude=slice(bbox["ylim"][1], bbox["ylim"][0]),
        ).load()

        # Determine time range
        if is_projection:
            years = list(range(2006, years_up_to + 1))
        else:
            years = list(DEFAULT_YEARS_OBS)

        # Extract individual DataArrays with time filtering
        result = {}
        for orig_var, dataset_var in var_map.items():
            da = subset[dataset_var]

            # Calendar conversion
            try:
                da = da.convert_calendar(
                    calendar="gregorian", missing=np.nan, align_on="date"
                )
            except ValueError as exc:
                msg = str(exc)
                if "date_range_like" in msg and "frequency was not inferable" in msg:
                    log.warning("Time frequency not inferable; reindexing before calendar conversion.")
                    da = _reindex_daily(da, log=log)
                    da = da.convert_calendar(
                        calendar="gregorian", missing=np.nan, align_on="date"
                    )
                    da = _reindex_daily(da, log=log)
                else:
                    raise

            # Time filter
            time_mask = (da["time"].dt.year >= years[0]) & (
                da["time"].dt.year <= years[-1]
            )
            result[orig_var] = da.sel(time=time_mask)

        log.info(
            "%sBatch extraction completed (%d variables).",
            log_prefix,
            len(result),
        )
        return result


def _postprocess_variable(
    *,
    variable: str,
    obs_data: xr.DataArray | None,
    hist_data: xr.DataArray | None,
    proj_data: xr.DataArray | None,
    obs_only: bool,
    bias_correction: bool,
    historical: bool,
    years_obs: range,
) -> xr.DataArray:
    """
    Post-process a single variable: unit conversion + optional bias correction.

    This is CPU-bound work that benefits from parallelization.
    """
    log = logger.getChild(variable)

    # Unit conversion for observations
    ref = None
    if obs_data is not None:
        ref = _apply_unit_conversion_obs(obs_data, variable)

    if obs_only:
        return ref

    # Unit conversion for CORDEX data
    hist = None
    proj = None

    if hist_data is not None:
        hist = _apply_unit_conversion_cordex(hist_data, variable)
        hist = hist.interpolate_na(dim="time", method="linear")

    if proj_data is not None:
        proj = _apply_unit_conversion_cordex(proj_data, variable)
        proj = proj.interpolate_na(dim="time", method="linear")

    # Bias correction and merging
    if bias_correction and historical and hist is not None and proj is not None:
        log.info("Training EQM with leave-one-out cross-validation")
        hist_bs = _leave_one_out_bias_correction(ref, hist, variable, log)

        QM_mo = sdba.EmpiricalQuantileMapping.train(
            ref,
            hist,
            group="time.month",
            kind="*" if variable in ["pr", "rsds", "sfcWind"] else "+",
        )
        log.info("Performing bias correction on projections")
        proj_bs = QM_mo.adjust(proj, extrapolation="constant", interp="linear")

        if variable == "hurs":
            hist_bs = hist_bs.clip(0, 100)
            proj_bs = proj_bs.clip(0, 100)

        return xr.concat([hist_bs, proj_bs], dim="time")

    elif not bias_correction and historical and hist is not None and proj is not None:
        return xr.concat([hist, proj], dim="time")

    elif bias_correction and not historical and proj is not None:
        log.info("Performing bias correction with EQM")
        QM_mo = sdba.EmpiricalQuantileMapping.train(
            ref,
            proj,
            group="time.month",
            kind="*" if variable in ["pr", "rsds", "sfcWind"] else "+",
        )
        proj_bs = QM_mo.adjust(proj, extrapolation="constant", interp="linear")

        if variable == "hurs":
            proj_bs = proj_bs.clip(0, 100)

        return proj_bs

    else:
        return proj


def _apply_unit_conversion_obs(data: xr.DataArray, variable: str) -> xr.DataArray:
    """Apply unit conversion for ERA5 observations."""
    var = VARIABLES_MAP[variable]

    if var in ["t2mx", "t2mn", "t2m"]:
        data = data - 273.15
        data.attrs["units"] = "°C"
    elif var == "tp":
        data = data * 1000
        data.attrs["units"] = "mm"
    elif var == "ssrd":
        data = data / 86400
        data.attrs["units"] = "W m-2"
    elif var == "sfcwind":
        data = data * (4.87 / np.log((67.8 * 10) - 5.42))
        data.attrs["units"] = "m s-1"

    return data


def _apply_unit_conversion_cordex(data: xr.DataArray, variable: str) -> xr.DataArray:
    """Apply unit conversion for CORDEX data."""
    if variable in ["tas", "tasmax", "tasmin"]:
        data = data - 273.15
        data.attrs["units"] = "°C"
    elif variable == "pr":
        data = data * 86400
        data.attrs["units"] = "mm"
    elif variable == "rsds":
        data.attrs["units"] = "W m-2"
    elif variable == "sfcWind":
        data = data * (4.87 / np.log((67.8 * 10) - 5.42))
        data.attrs["units"] = "m s-1"

    return data


def _reindex_daily(data: xr.DataArray, *, log: logging.Logger) -> xr.DataArray:
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
        full_time = xr.cftime_range(start=start, end=end, freq="D", calendar=calendar)
    if len(full_time) != time_coord.size:
        log.warning(
            "Reindexing time to daily range (%d steps); filling %d missing dates with NaN.",
            len(full_time),
            len(full_time) - time_coord.size,
        )
    return data.reindex(time=full_time)


def _open_dataset_with_retry(
    url_or_path: str,
    *,
    retries: int = RETRY_MAX_ATTEMPTS,
    base_delay_s: float = RETRY_BASE_DELAY_S,
    backoff_factor: float = RETRY_BACKOFF_FACTOR,
) -> xr.Dataset:
    """
    Open a dataset with connection throttling and exponential backoff retry.

    Uses a module-level semaphore to limit concurrent THREDDS connections,
    preventing server overload when running multiple parallel processes.
    """
    last_exc = None
    delay = base_delay_s

    # Acquire semaphore to limit concurrent connections
    with _THREDDS_CONNECTION_SEMAPHORE:
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
                logger.warning(
                    "open_dataset failed (attempt %d/%d) for %s: %s. Retrying in %.1fs.",
                    attempt,
                    retries,
                    url_or_path,
                    exc,
                    delay,
                )
                time.sleep(delay)
                delay *= backoff_factor  # Exponential backoff: 2s -> 4s -> 8s

    assert last_exc is not None
    raise last_exc
