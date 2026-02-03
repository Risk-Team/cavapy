"""Validation helpers for input parameters and spatial domain checks."""

import logging

import pandas as pd
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

from cava_config import (
    ERA5_DATA_REMOTE_URL,
    INVENTORY_DATA_LOCAL_PATH,
    INVENTORY_DATA_REMOTE_URL,
    VALID_GCM,
    VALID_RCM,
    logger,
)


def _ensure_inventory_not_empty(
    filtered_data: pd.DataFrame,
    *,
    dataset: str,
    cordex_domain: str,
    gcm: str,
    rcm: str,
    experiments: list[str],
    activity_filter: str,
    log: logging.Logger | None = None,
) -> None:
    """
    Ensure that the inventory filter returned at least one URL.
    If not, raise a clear, informative error instead of failing later with iloc[0].
    """
    if not filtered_data.empty:
        return

    msg = (
        "No CORDEX entries found in the inventory for the requested configuration.\n"
        f"  dataset        : {dataset}\n"
        f"  domain         : {cordex_domain}\n"
        f"  gcm            : {gcm}\n"
        f"  rcm            : {rcm}\n"
        f"  experiments    : {experiments}\n"
        f"  activity_filter: {activity_filter}\n\n"
        "This usually means that this GCM/RCM/experiment combination does not exist "
        "or that ther is an issue with the inventory data.\n"
        "Please check the inventory CSV at https://hub.ipcc.ifca.es/thredds/fileServer/inventories/cava.csv"
    )

    if log is not None:
        log.error(msg)

    raise ValueError(msg)


def _validate_urls(
    gcm: str = None,
    rcm: str = None,
    rcp: str = None,
    remote: bool = True,
    cordex_domain: str = None,
    obs: bool = False,
    historical: bool = False,
    bias_correction: bool = False,
    dataset: str = "CORDEX-CORE",
    variables: list[str] | None = None,
):
    """Validate inventory availability and log resolved dataset URLs."""
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

        # Determine activity filter based on dataset
        activity_filter = "FAO" if dataset == "CORDEX-CORE" else "CRDX-ISIMIP-025"

        # Filter the data based on the conditions
        filtered_data = data[
            lambda x: (
                x["activity"].str.contains(activity_filter, na=False)
                & (x["domain"] == cordex_domain)
                & (x["model"].str.contains(gcm, na=False))
                & (x["rcm"].str.contains(rcm, na=False))
                & (x["experiment"].isin(experiments))
            )
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

        # Extract the column values as a list
        for _, row in filtered_data.iterrows():
            if row["experiment"] == "historical":
                log_hist = logger.getChild(f"URL-validation-{gcm}-{rcm}-historical")
                log_hist.info(f"{row[column_to_use]}")
            else:
                log_proj = logger.getChild(f"URL-validation-{gcm}-{rcm}-{rcp}")
                log_proj.info(f"{row[column_to_use]}")

    else:  # when obs is True
        if variables:
            for variable in variables:
                log_obs = logger.getChild(f"URL-validation-ERA5-{variable}")
                log_obs.info(f"{ERA5_DATA_REMOTE_URL}")
        else:
            log_obs = logger.getChild("URL-validation-ERA5")
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
        "cultural", "admin_0_countries", "50m"
    )

    # Get the actual shapefile path from the feature
    _ = countries_feature.with_scale("50m").geometries()

    # Search for the country using Natural Earth records
    for country_record in shpreader.Reader(
        shpreader.natural_earth(
            resolution="50m", category="cultural", name="admin_0_countries"
        )
    ).records():
        # Try multiple name fields for better matching
        country_names = [
            country_record.attributes.get("NAME", ""),
            country_record.attributes.get("NAME_LONG", ""),
            country_record.attributes.get("ADMIN", ""),
            country_record.attributes.get("NAME_EN", ""),
        ]

        if any(name.lower() == country_name.lower() for name in country_names if name):
            return country_record.geometry.bounds

    # If not found, check for capitalization issue
    if country_name and country_name[0].islower():
        capitalized = country_name.capitalize()
        raise ValueError(
            f"Country '{country_name}' not found. Try capitalizing the first letter: '{capitalized}'"
        )
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
    """Resolve a country name or bbox into a validated bounding box."""
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
            ("NCC", "Reg"),
        ],
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
    """Ensure the bbox is fully contained inside the selected CORDEX domain."""
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
            "max_lat": [
                28,
                21.7,
                26.5,
                13.75,
                45.75,
                65.0,
                18.75,
                28.5,
                42.75,
                69.75,
            ],
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
