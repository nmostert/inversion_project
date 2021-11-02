import sys
sys.path.append("..")
import project.inversion as inv
import project.io_utils as io
import pandas as pd
import numpy as np
# dev imports. remove
from tabulate import tabulate


def test_internal_consistency():
    # Read in simple set of test parameters
    config = io.read_tephra2_config("test_data/test_config.txt")

    globs = {
        "LITHIC_DIAMETER_THRESHOLD": 7.,
        "PUMICE_DIAMETER_THRESHOLD": -1.,
        "AIR_VISCOSITY": 0.000018325,
        "AIR_DENSITY":  1.293,
        "GRAVITY": 9.81,
    }
    config["WIND_SPEED"] = 10
    config["THEO_MAX_COL"] = config["PLUME_HEIGHT"]

    io.print_table(pd.DataFrame(config, index=["Value"]).T)

    # Read in example grid file
    grid = io.read_grid("test_data/colima_grid.csv")

    grid = grid[["Easting", "Northing"]]

    io.print_table(grid)

    # Construct Phi steps
    theo_phi_steps = inv.get_phi_steps(config["MIN_GRAINSIZE"],
                                       config["MAX_GRAINSIZE"],
                                       config["PART_STEPS"],
                                       config["MEDIAN_GRAINSIZE"],
                                       config["STD_GRAINSIZE"],
                                       globs["LITHIC_DIAMETER_THRESHOLD"],
                                       globs["PUMICE_DIAMETER_THRESHOLD"],
                                       config["LITHIC_DENSITY"],
                                       config["PUMICE_DENSITY"])

    # Get wind vectors
    wind_angle = np.radians(55)

    u = config["WIND_SPEED"]*np.cos(wind_angle)
    v = config["WIND_SPEED"]*np.sin(wind_angle)

    # Forward model
    forward_df = inv.gaussian_stack_forward(
        grid, int(config["COL_STEPS"]), config["VENT_ELEVATION"],
        config["PLUME_HEIGHT"], 2500, theo_phi_steps, (
            config["ALPHA"], config["BETA"]),
        config["ERUPTION_MASS"],
        (u, v), config["DIFFUSION_COEFFICIENT"], config["EDDY_CONST"],
        config["FALL_TIME_THRESHOLD"]
    )

    forward_df["radius"] = np.sqrt(
        forward_df["Easting"]**2 + forward_df["Northing"]**2)
    forward_df = forward_df.sort_values(by=['radius'])

    io.print_table(forward_df.head())

    # Invert same data
    priors_vals = {
        "a": config["ALPHA"],
        "b": config["BETA"],
        "h1": config["PLUME_HEIGHT"],
        "u": u,
        "v": v,
        "D": config["DIFFUSION_COEFFICIENT"],
        "ftt": config["FALL_TIME_THRESHOLD"],
        "eta": 0,
        "zeta": 0
    }

    invert_params = {
        "a": True,
        "b": True,
        "h1": True,
        "u": True,
        "v": True,
        "D": True,
        "ftt": True,
        "eta": False,
        "zeta": False
    }

    out = inv.gaussian_stack_inversion(
        forward_df, len(forward_df),
        config["COL_STEPS"], config["VENT_ELEVATION"],
        config["THEO_MAX_COL"], 2500, theo_phi_steps,
        config["ERUPTION_MASS"],
        invert_params=invert_params,
        priors=priors_vals,
        max_iter=40, tol=0.006,
        adjust_TGSD=False, adjust_mass=False,
        adjustment_factor=None,
        column_cap=config["THEO_MAX_COL"],
        logging=None
    )
    inversion_table, params, misfit, status, param_trace, misfit_trace, \
        tgsd_trace, mass_trace = out

    io.print_table(inversion_table)
    io.print_table(pd.DataFrame(params, index=["Values"]))

    # Test if parameters are recovered.
    assert params["a"] == config["ALPHA"], "Parameter ALPHA is not recovered"
    assert params["b"] == config["BETA"], "Parameter BETA is not recovered"
    assert params["h1"] == config["PLUME_HEIGHT"], \
        "Parameter PLUME_HEIGHT is not recovered"
    assert params["u"] == u, "Parameter u (wind-x) is not recovered"
    assert params["v"] == v, "Parameter v (wind-y) is not recovered"
    assert np.abs(params["D"] - config["DIFFUSION_COEFFICIENT"]) <= 1e-5, \
        "Parameter DIFFUSION_COEFFICIENT is not recovered"
    assert np.abs(params["ftt"] - config["FALL_TIME_THRESHOLD"]) <= 1e-5, \
        "Parameter FALL_TIME_THRESHOLD is not recovered"


if __name__ == "__main__":
    test_internal_consistency()
