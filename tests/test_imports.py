def test_imports():
    import cavapy

    assert hasattr(cavapy, "get_climate_data")
    assert hasattr(cavapy, "plot_spatial_map")
    assert hasattr(cavapy, "plot_time_series")
