"""Bias-correction utilities for CORDEX data using xsdba."""

import numpy as np
import xsdba as sdba
import xarray as xr


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
            hist_corrected_year = hist_corrected_year.where(
                hist_corrected_year <= 100, 100
            )
            hist_corrected_year = hist_corrected_year.where(
                hist_corrected_year >= 0, 0
            )

        corrected_years.append(hist_corrected_year)

    # Concatenate all corrected years and sort by time
    hist_bs = xr.concat(corrected_years, dim="time").sortby("time")

    log.info("Leave-one-out cross-validation bias correction completed")
    return hist_bs
