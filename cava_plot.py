"""Plotting helpers for spatial maps and time series of climate data."""

from typing import List, Optional, Tuple, Union

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def plot_spatial_map(
    data: xr.DataArray,
    time_period: Optional[Tuple[int, int]] = None,
    aggregation: str = "mean",
    title: Optional[str] = None,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (12, 8),
    show_countries: bool = True,
    save_path: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Create a spatial map visualization of climate data.

    The function subsets by time period, aggregates across time, and renders
    the result on a PlateCarree map with optional country borders.
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
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()})

    # Plot data
    im = plot_data.plot(
        ax=ax,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
        **kwargs,
    )

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    if show_countries:
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.7)
    ax.add_feature(cfeature.OCEAN, color="lightblue", alpha=0.3)
    ax.add_feature(cfeature.LAND, color="lightgray", alpha=0.3)

    # Set extent to data bounds with small buffer
    lon_min, lon_max = plot_data.longitude.min().item(), plot_data.longitude.max().item()
    lat_min, lat_max = plot_data.latitude.min().item(), plot_data.latitude.max().item()
    buffer = 0.5
    ax.set_extent(
        [lon_min - buffer, lon_max + buffer, lat_min - buffer, lat_max + buffer],
        ccrs.PlateCarree(),
    )

    # Add gridlines with labels only on left and bottom
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    if hasattr(plot_data, "units"):
        cbar.set_label(
            f"{plot_data.name} ({plot_data.units})", rotation=270, labelpad=20
        )
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
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

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
    **kwargs,
) -> plt.Figure:
    """
    Create time series plots of climate data.

    The function aggregates spatially, converts to annual means, and can
    optionally add a linear trend line.
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
    for dataset, label in zip(data_list, labels):
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
            ax1.plot(
                annual_data.year,
                p(annual_data.year),
                linestyle="--",
                alpha=0.7,
                color=ax1.lines[-1].get_color(),
            )

    # Format main plot
    ax1.set_xlabel("Year", fontsize=12)
    if ylabel is None:
        if hasattr(data_list[0], "units"):
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
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
