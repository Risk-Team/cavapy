"""Public API for retrieving and visualizing CAVA climate data."""

import logging
import multiprocessing as mp
import xarray as xr
from tqdm import tqdm

from cava_config import (
    DEFAULT_YEARS_OBS,
    VALID_DATASETS,
    VALID_DOMAINS,
    VALID_GCM,
    VALID_RCM,
    VALID_RCPS,
    VALID_VARIABLES,
    logger,
)
from cava_download import process_worker
from cava_plot import plot_spatial_map, plot_time_series
from cava_validation import (
    _geo_localize,
    _get_country_bounds,
    _validate_gcm_rcm_combinations,
    _validate_urls,
)


def _get_climate_data_single(
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
    max_threads_per_process: int = 3,
    dataset: str = "CORDEX-CORE",
) -> dict[str, xr.DataArray]:
    """Internal single-combination fetch (one rcp/gcm/rcm), preserves legacy behavior."""

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

    # Validate dataset parameter
    if dataset not in VALID_DATASETS:
        raise ValueError(
            f"Invalid dataset='{dataset}'. Must be one of {VALID_DATASETS}"
        )
    
    # Check for incompatible dataset and bias_correction combination
    if dataset == "CORDEX-CORE-BC" and bias_correction:
        raise ValueError(
            "Cannot apply bias_correction=True when using dataset='CORDEX-CORE-BC'. "
            "The CORDEX-CORE-BC dataset is already bias-corrected using ISIMIP methodology."
        )
    
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

    _validate_urls(
        gcm,
        rcm,
        rcp,
        remote,
        cordex_domain,
        obs,
        historical,
        bias_correction,
        dataset,
        variables,
    )

    bbox = _geo_localize(country, xlim, ylim, buffer, cordex_domain, obs)

    if num_processes <= 1 or len(variables) <= 1:
        results = {}
        for variable in variables:
            try:
                results[variable] = process_worker(
                    max_threads_per_process,
                    variable=variable,
                    bbox=bbox,
                    cordex_domain=cordex_domain,
                    rcp=rcp,
                    gcm=gcm,
                    rcm=rcm,
                    years_up_to=years_up_to,
                    years_obs=years_obs,
                    obs=obs,
                    bias_correction=bias_correction,
                    historical=historical,
                    remote=remote,
                    dataset=dataset,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Variable '{variable}' failed for {gcm}-{rcm} {rcp}"
                ) from exc
        return results

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
                        "dataset": dataset,
                    },
                )
            )

        try:
            results = {
                variable: futures[i].get() for i, variable in enumerate(variables)
            }
        except Exception as exc:
            pool.terminate()
            pool.join()
            raise RuntimeError(
                f"Variable processing failed for {gcm}-{rcm} {rcp}"
            ) from exc

        pool.close()  # Prevent any more tasks from being submitted to the pool
        pool.join()  # Wait for all worker processes to finish

    return results


def _normalize_selection(
    value: str | list[str] | None,
    valid_values: list[str],
    name: str,
) -> tuple[list[str], bool]:
    if value is None:
        return list(valid_values), True
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)
    if not values:
        raise ValueError(f"{name} list cannot be empty")
    invalid = [v for v in values if v not in valid_values]
    if invalid:
        raise ValueError(f"Invalid {name} values: {invalid}. Must be within {valid_values}")
    return values, False


def _run_combo_variable_task(
    rcp_val: str,
    gcm_val: str,
    rcm_val: str,
    variable: str,
    common_kwargs: dict,
    max_threads_per_process: int,
    bbox: dict,
):
    data = process_worker(
        max_threads_per_process,
        variable=variable,
        bbox=bbox,
        rcp=rcp_val,
        gcm=gcm_val,
        rcm=rcm_val,
        **common_kwargs,
    )
    return rcp_val, gcm_val, rcm_val, variable, data


def _combo_task_wrapper(args):
    """Wrapper for imap_unordered which requires single-argument callable."""
    return _run_combo_variable_task(*args)


def _init_pool_worker():
    """Suppress logging in worker processes to show only progress bar."""
    logging.getLogger("climate").setLevel(logging.CRITICAL)


def get_climate_data(
    *,
    country: str | None,
    years_obs: range | None = None,
    obs: bool = False,
    cordex_domain: str | None = None,
    rcp: str | list[str] | None = None,
    gcm: str | list[str] | None = None,
    rcm: str | list[str] | None = None,
    years_up_to: int | None = None,
    bias_correction: bool = False,
    historical: bool = False,
    buffer: int = 0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    remote: bool = True,
    variables: list[str] | None = None,
    num_processes: int = len(VALID_VARIABLES),
    max_threads_per_process: int = 3,
    dataset: str = "CORDEX-CORE",
    max_total_processes: int = 6,
) -> dict:
    """
    Retrieve CORDEX-CORE projections and/or ERA5 observations for a region.

    The function orchestrates validation, spatial subsetting, unit conversion,
    optional bias correction, and parallel download/processing.
    Parallelization uses processes across variables or model/variable combinations,
    with a thread pool inside each process for per-variable downloads.

    Args:
    country (str): Name of the country for which data is to be processed.
        Use None if specifying a region using xlim and ylim.
    years_obs (range): Range of years for observational data (ERA5 only). Required when obs is True. (default: None).
    obs (bool): Flag to indicate if processing observational data (default: False).
        When True, only years_obs is required. CORDEX parameters are optional.
    cordex_domain (str): CORDEX domain of the climate data. One of {VALID_DOMAINS}.
        Required when obs is False. (default: None).
    rcp (str | list[str] | None): Representative Concentration Pathway(s). One of {VALID_RCPS}.
        If None, all RCPs are used. Required when obs is False. (default: None).
    gcm (str | list[str] | None): GCM name(s). One of {VALID_GCM}.
        If None, all GCMs are used. Required when obs is False. (default: None).
    rcm (str | list[str] | None): RCM name(s). One of {VALID_RCM}.
        If None, all RCMs are used. Required when obs is False. (default: None).
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
    num_processes (int): Number of processes to use, one per variable for a single combo.
        If num_processes <= 1 or only one variable is requested, variables run sequentially.
        By default equals to the number of all possible variables. (default: {len(VALID_VARIABLES)}).
    max_threads_per_process (int): Max number of threads within each process. (default: 3).
    dataset (str): Dataset source to use. Options are "CORDEX-CORE" (original data) or "CORDEX-CORE-BC" (ISIMIP bias-corrected data). (default: "CORDEX-CORE").
    max_total_processes (int): Max number of processes when multiple models/RCPs are requested.
        Defaults to 6 (cap applies to total combo-variable tasks).

    Returns:
    dict: If a single (gcm, rcm, rcp) is requested, returns {variable: DataArray}.
          If multiple are requested, returns {rcp: {"{gcm}-{rcm}": {variable: DataArray}}}.
    """

    if obs and any(isinstance(v, list) for v in (rcp, gcm, rcm) if v is not None):
        raise ValueError("rcp/gcm/rcm lists are not supported when obs=True")

    if obs:
        return _get_climate_data_single(
            country=country,
            years_obs=years_obs,
            obs=obs,
            cordex_domain=cordex_domain,
            rcp=rcp if isinstance(rcp, str) or rcp is None else rcp[0],
            gcm=gcm if isinstance(gcm, str) or gcm is None else gcm[0],
            rcm=rcm if isinstance(rcm, str) or rcm is None else rcm[0],
            years_up_to=years_up_to,
            bias_correction=bias_correction,
            historical=historical,
            buffer=buffer,
            xlim=xlim,
            ylim=ylim,
            remote=remote,
            variables=variables,
            num_processes=num_processes,
            max_threads_per_process=max_threads_per_process,
            dataset=dataset,
        )

    if cordex_domain is None:
        raise ValueError("cordex_domain is required when obs is False")

    if xlim is None and ylim is not None or xlim is not None and ylim is None:
        raise ValueError(
            "xlim and ylim mismatch: they must be both specified or both unspecified"
        )
    if country is None and xlim is None:
        raise ValueError("You must specify a country or (xlim, ylim)")
    if country is not None and xlim is not None:
        raise ValueError("You must specify either country or (xlim, ylim), not both")

    if dataset not in VALID_DATASETS:
        raise ValueError(
            f"Invalid dataset='{dataset}'. Must be one of {VALID_DATASETS}"
        )
    if dataset == "CORDEX-CORE-BC" and bias_correction:
        raise ValueError(
            "Cannot apply bias_correction=True when using dataset='CORDEX-CORE-BC'. "
            "The CORDEX-CORE-BC dataset is already bias-corrected using ISIMIP methodology."
        )

    if years_up_to is None:
        raise ValueError("years_up_to is required when obs is False")
    if years_up_to <= 2006:
        raise ValueError("years_up_to must be greater than 2006")

    if years_obs is None:
        years_obs = DEFAULT_YEARS_OBS

    rcps, _all_rcps = _normalize_selection(rcp, VALID_RCPS, "rcp")
    gcms, all_gcms = _normalize_selection(gcm, VALID_GCM, "gcm")
    rcms, all_rcms = _normalize_selection(rcm, VALID_RCM, "rcm")

    combos = [(r, g, m) for r in rcps for g in gcms for m in rcms]
    if len(combos) == 1:
        rcp_single, gcm_single, rcm_single = combos[0]
        return _get_climate_data_single(
            country=country,
            years_obs=years_obs,
            obs=obs,
            cordex_domain=cordex_domain,
            rcp=rcp_single,
            gcm=gcm_single,
            rcm=rcm_single,
            years_up_to=years_up_to,
            bias_correction=bias_correction,
            historical=historical,
            buffer=buffer,
            xlim=xlim,
            ylim=ylim,
            remote=remote,
            variables=variables,
            num_processes=num_processes,
            max_threads_per_process=max_threads_per_process,
            dataset=dataset,
        )

    valid_combos: list[tuple[str, str, str]] = []
    invalid_combos: list[tuple[str, str, str]] = []
    for rcp_val, gcm_val, rcm_val in combos:
        try:
            _validate_gcm_rcm_combinations(cordex_domain, gcm_val, rcm_val)
            valid_combos.append((rcp_val, gcm_val, rcm_val))
        except ValueError:
            invalid_combos.append((rcp_val, gcm_val, rcm_val))

    if invalid_combos and not (all_gcms or all_rcms):
        raise ValueError(
            "Some requested GCM/RCM combinations are invalid for this domain: "
            + ", ".join(f"{g}-{m} ({r})" for r, g, m in invalid_combos)
        )
    if invalid_combos and (all_gcms or all_rcms):
        logger.warning(
            "Skipping invalid GCM/RCM combinations for %s: %s",
            cordex_domain,
            ", ".join(f"{g}-{m} ({r})" for r, g, m in invalid_combos),
        )

    if variables is not None:
        invalid_vars = [var for var in variables if var not in VALID_VARIABLES]
        if invalid_vars:
            raise ValueError(
                f"Invalid variables: {invalid_vars}. Must be a subset of {VALID_VARIABLES}"
            )
        variables_list = list(variables)
    else:
        variables_list = list(VALID_VARIABLES)

    results: dict[str, dict[str, dict[str, xr.DataArray]]] = {}

    max_workers = max_total_processes
    max_workers = max(1, min(max_workers, len(valid_combos) * len(variables_list)))

    retry_log_level = logging.DEBUG if len(valid_combos) > 1 else logging.WARNING

    common_kwargs = {
        "years_obs": years_obs,
        "obs": obs,
        "cordex_domain": cordex_domain,
        "years_up_to": years_up_to,
        "bias_correction": bias_correction,
        "historical": historical,
        "remote": remote,
        "dataset": dataset,
        "retry_log_level": retry_log_level,
    }

    bbox = _geo_localize(country, xlim, ylim, buffer, cordex_domain, obs)

    for rcp_val, gcm_val, rcm_val in valid_combos:
        _validate_urls(
            gcm_val,
            rcm_val,
            rcp_val,
            remote,
            cordex_domain,
            obs,
            historical,
            bias_correction,
            dataset,
            variables_list,
        )

    tasks = [
        (rcp_val, gcm_val, rcm_val, variable, common_kwargs, max_threads_per_process, bbox)
        for rcp_val, gcm_val, rcm_val in valid_combos
        for variable in variables_list
    ]

    with mp.Pool(processes=max_workers, initializer=_init_pool_worker) as pool:
        try:
            with tqdm(total=len(tasks), desc="Processing climate data", unit="task") as pbar:
                for rcp_val, gcm_val, rcm_val, variable, data in pool.imap_unordered(
                    _combo_task_wrapper, tasks
                ):
                    results.setdefault(rcp_val, {}).setdefault(f"{gcm_val}-{rcm_val}", {})[
                        variable
                    ] = data
                    pbar.set_postfix({"last": f"{gcm_val}-{rcm_val}/{variable}"})
                    pbar.update(1)
        except Exception as exc:
            pool.terminate()
            pool.join()
            raise RuntimeError(
                "Model/RCP processing failed. Enable DEBUG logs for details."
            ) from exc

    return results


if __name__ == "__main__":
    # Examples: show how get_climate_data parallelizes.
    cordex_domain = "AFR-22"
    years_up_to = 2015

    print("\nExample 1: multiple models (combo-variable tasks parallelized)...")
    multi = get_climate_data(
        country="Togo",
        cordex_domain=cordex_domain,
        rcp="rcp26",
        gcm=VALID_GCM,
        rcm=VALID_RCM,
        years_up_to=years_up_to,
        historical=True,
        bias_correction=False,
        dataset="CORDEX-CORE-BC",
        max_total_processes=6,
    )
    # Show a compact summary of the structure returned
    for rcp_val, model_map in multi.items():
        print(rcp_val, "models:", list(model_map.keys()))

    print("\nExample 2: single model/RCP (variables parallelized)...")
    single = get_climate_data(
        country="Togo",
        cordex_domain=cordex_domain,
        rcp="rcp26",
        gcm="MPI",
        rcm="REMO",
        years_up_to=years_up_to,
        historical=True,
        bias_correction=False,
        dataset="CORDEX-CORE",
    )
    print("Single model variables:", list(single.keys()))

    print("Examples completed successfully!")
