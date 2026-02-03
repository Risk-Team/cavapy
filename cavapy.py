"""Public API for retrieving and visualizing CAVA climate data."""

import multiprocessing as mp
import xarray as xr

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


def _auto_max_threads_per_process(
    *, obs: bool, historical: bool, bias_correction: bool
) -> int:
    if obs or bias_correction:
        return 3
    if historical:
        return 2
    return 1


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
    variables: list[str] | None = None,
    max_threads_per_process: int | None = None,
    dataset: str = "CORDEX-CORE",
) -> dict[str, xr.DataArray]:
    """Fetch a single (rcp, gcm, rcm) combination with validation and processing."""

    # Validation for basic parameters
    if (xlim is None and ylim is not None) or (xlim is not None and ylim is None):
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
        
        # Set defaults for CORDEX parameters (not used but needed for function calls)
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

    _validate_urls(gcm, rcm, rcp, cordex_domain, obs, historical, bias_correction, dataset)

    bbox = _geo_localize(country, xlim, ylim, buffer, cordex_domain, obs)

    if max_threads_per_process is None:
        max_threads_per_process = _auto_max_threads_per_process(
            obs=obs, historical=historical, bias_correction=bias_correction
        )

    try:
        return process_worker(
            max_threads_per_process,
            variables=variables,
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
            dataset=dataset,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Variable processing failed for {gcm}-{rcm} {rcp}"
        ) from exc


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


def _run_combo_task(args: tuple):
    """Wrapper for imap_unordered - takes single tuple argument."""
    rcp_val, gcm_val, rcm_val, common_kwargs, max_threads_per_process = args
    data = _get_climate_data_single(
        rcp=rcp_val,
        gcm=gcm_val,
        rcm=rcm_val,
        max_threads_per_process=max_threads_per_process,
        **common_kwargs,
    )
    return rcp_val, gcm_val, rcm_val, data


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
    variables: list[str] | None = None,
    max_threads_per_process: int | None = None,
    max_model_processes: int = 6,
    dataset: str = "CORDEX-CORE",
) -> dict:
    """
    Retrieve CORDEX-CORE projections and/or ERA5 observations for a region.

    The function orchestrates validation, spatial subsetting, unit conversion,
    optional bias correction, and parallel download/processing.

    Parallelization:
        Uses a two-level strategy optimized for THREDDS/OpenDAP:
        - Level 1 (Processes): Model/RCP combinations run in parallel processes
        - Level 2 (Threads): Within each process, datasets are opened and
          variables post-processed in parallel threads
        - All variables are batch-extracted in a single OpenDAP request per dataset

    Args:
        country (str): Country name for spatial subsetting. Use None with xlim/ylim.
        years_obs (range | None): Observation years (ERA5). Required when obs=True.
        obs (bool): When True, uses ERA5 observations; CORDEX params are ignored.
        cordex_domain (str | None): CORDEX domain. Required when obs=False.
        rcp (str | list[str] | None): RCP(s) to request. None means all.
        gcm (str | list[str] | None): GCM(s) to request. None means all.
        rcm (str | list[str] | None): RCM(s) to request. None means all.
        years_up_to (int | None): Projection end year (>=2007). Required when obs=False.
        bias_correction (bool): Apply bias correction (not allowed with CORDEX-CORE-BC).
        historical (bool): Include 1980-2005 historical runs with projections.
        buffer (int): Degrees to expand bounding box.
        xlim (tuple[float, float] | None): Longitude bounds (min, max).
        ylim (tuple[float, float] | None): Latitude bounds (min, max).
        variables (list[str] | None): Variable subset. None means all.
        max_threads_per_process (int | None): Threads per process for opening datasets
            and post-processing variables. Auto-tuned if None: 3 threads when
            bias_correction or obs, 2 when historical, else 1.
        max_model_processes (int): Max parallel processes for model/RCP combinations.
            Default 6. Increase for faster multi-model requests (more server load).
        dataset (str): "CORDEX-CORE" or "CORDEX-CORE-BC".

    Returns:
        dict: If a single (gcm, rcm, rcp) is requested, returns {variable: DataArray}.
            If multiple are requested, returns {rcp: {"{gcm}-{rcm}": {variable: DataArray}}}.
    """

    if obs and any(isinstance(v, list) for v in (rcp, gcm, rcm) if v is not None):
        raise ValueError("rcp/gcm/rcm lists are not supported when obs=True")

    if max_threads_per_process is None:
        max_threads_per_process = _auto_max_threads_per_process(
            obs=obs, historical=historical, bias_correction=bias_correction
        )

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
            variables=variables,
            max_threads_per_process=max_threads_per_process,
            dataset=dataset,
        )

    if cordex_domain is None:
        raise ValueError("cordex_domain is required when obs is False")

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
            variables=variables,
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

    results: dict[str, dict[str, dict[str, xr.DataArray]]] = {}
    max_workers = min(max_model_processes, len(valid_combos))

    common_kwargs = {
        "country": country,
        "years_obs": years_obs,
        "obs": obs,
        "cordex_domain": cordex_domain,
        "years_up_to": years_up_to,
        "bias_correction": bias_correction,
        "historical": historical,
        "buffer": buffer,
        "xlim": xlim,
        "ylim": ylim,
        "variables": variables,
        "dataset": dataset,
    }

    tasks = [
        (rcp_val, gcm_val, rcm_val, common_kwargs, max_threads_per_process)
        for rcp_val, gcm_val, rcm_val in valid_combos
    ]

    with mp.Pool(processes=max_workers) as pool:
        try:
            for rcp_val, gcm_val, rcm_val, data in pool.imap_unordered(
                _run_combo_task, tasks
            ):
                results.setdefault(rcp_val, {})[f"{gcm_val}-{rcm_val}"] = data
        except Exception as exc:
            pool.terminate()
            pool.join()
            raise RuntimeError(
                "Model/RCP processing failed. Enable DEBUG logs for details."
            ) from exc

    logger.info("All %d model combinations completed successfully.", len(valid_combos))
    return results


if __name__ == "__main__":
    # Example: Retrieve Togo data for all variables and all model combinations.
    # This loops over all GCM/RCM combinations and both RCPs.
    cordex_domain = "AFR-22"
    years_up_to = 2015
    print("\nGetting CORDEX-CORE BC data for multiple models in parallel...")
    data_parallel = get_climate_data(
        country="Togo",
        cordex_domain=cordex_domain,
        rcp="rcp26",
        gcm=VALID_GCM,
        rcm=VALID_RCM,
        years_up_to=years_up_to,
        historical=True,
        dataset="CORDEX-CORE-BC",
    )
    print("Parallel example keys:", list(data_parallel.keys()))

    print("\nGetting CORDEX-CORE data for one model with all variables...")
    data_single_all = get_climate_data(
        country="Togo",
        cordex_domain=cordex_domain,
        rcp="rcp26",
        gcm=VALID_GCM[0],
        rcm=VALID_RCM[0],
        years_up_to=years_up_to,
        historical=True,
        dataset="CORDEX-CORE",
        variables=None,
    )
    print("Single-model variables:", list(data_single_all.keys()))

    print("Example completed successfully!")
