import sys
sys.path.append("..")
import project.inversion as inv
import project.io_utils as io
import project.vis_utils as vis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# dev imports. remove
from tabulate import tabulate
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)


def test_internal_consistency():
    # Read in simple set of test parameters
    config = io.read_config_file("test_data/cerronegro_config.txt")

    for key in config.keys():
        config[key] = float(config[key])

    config["MIN_GRAINSIZE"] = int(config["MIN_GRAINSIZE"])
    config["MAX_GRAINSIZE"] = int(config["MAX_GRAINSIZE"])
    config["PART_STEPS"] = int(config["PART_STEPS"])
    globs = {
        "LITHIC_DIAMETER_THRESHOLD": 7.,
        "PUMICE_DIAMETER_THRESHOLD": -1.,
        "AIR_VISCOSITY": 0.000018325,
        "AIR_DENSITY":  1.293,
        "GRAVITY": 9.81,
    }
    config["WIND_SPEED"] = 10
    config["THEO_MAX_COL"] = config["PLUME_HEIGHT"]

    logging.info(io.log_table(pd.DataFrame(config, index=["Value"]).T,
                              title="Input Configuration"))

    # Read in example grid file
    grid = io.read_grid("test_data/cerronegro_grid.csv",
                        columns=["Easting", "Northing", "Elev."])

    grid = grid[["Easting", "Northing"]]

    logging.info(io.log_table(grid, title="Grid points"))

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
    wind_angle = np.radians(190)

    u = config["WIND_SPEED"]*np.cos(wind_angle)
    v = config["WIND_SPEED"]*np.sin(wind_angle)

    # Forward model
    forward_df = inv.gaussian_stack_forward(
        grid, int(config["COL_STEPS"]), config["VENT_ELEVATION"],
        config["PLUME_HEIGHT"], 1, theo_phi_steps, (
            config["ALPHA"], config["BETA"]),
        config["ERUPTION_MASS"],
        (u, v), config["DIFFUSION_COEFFICIENT"], config["EDDY_CONST"],
        config["FALL_TIME_THRESHOLD"]
    )

    forward_df["radius"] = np.sqrt(
        forward_df["Easting"]**2 + forward_df["Northing"]**2)
    forward_df = forward_df.sort_values(by=['radius'])

    logging.info(io.log_table(forward_df.head(),
                 title="Forward Run Dataframe"))

    # Invert same data
    priors_vals = {
        "a": config["ALPHA"],
        "b": config["BETA"],
        "h1": config["PLUME_HEIGHT"],
        "u": u,
        "v": v,
        "D": config["DIFFUSION_COEFFICIENT"],
        "ftt": config["FALL_TIME_THRESHOLD"],
    }

    invert_params = {
        "a": True,
        "b": True,
        "h1": True,
        "u": True,
        "v": True,
        "D": True,
        "ftt": True,
    }

    out = inv.gaussian_stack_inversion(
        forward_df, len(forward_df),
        config["COL_STEPS"], config["VENT_ELEVATION"],
        config["THEO_MAX_COL"], 1, theo_phi_steps,
        config["ERUPTION_MASS"],
        invert_params=invert_params,
        priors=priors_vals,
        max_iter=40, tol=0.006, termination="norm_diff",
        adjust_TGSD=False, adjust_mass=False,
        adjustment_factor=None,
        column_cap=config["THEO_MAX_COL"],
    )
    inversion_table, params, misfit, status, message, param_trace, \
        misfit_trace, tgsd_trace, mass_trace = out

    logging.info(io.log_table(inversion_table, title="Column Result"))
    logging.info(io.log_table(pd.DataFrame(params, index=["Values"]),
                              title="Parameters"))

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


def test_tephra2_simple():
    # Read in simple set of test parameters
    config = io.read_config_file("test_data/simple_config.txt")

    for key in config.keys():
        config[key] = float(config[key])

    config["MIN_GRAINSIZE"] = int(config["MIN_GRAINSIZE"])
    config["MAX_GRAINSIZE"] = int(config["MAX_GRAINSIZE"])
    config["PART_STEPS"] = int(config["PART_STEPS"])

    globs = {
        "LITHIC_DIAMETER_THRESHOLD": 7.,
        "PUMICE_DIAMETER_THRESHOLD": -1.,
        "AIR_VISCOSITY": 0.000018325,
        "AIR_DENSITY":  1.293,
        "GRAVITY": 9.81,
    }
    config["WIND_SPEED"] = 10
    # config["THEO_MAX_COL"] = config["PLUME_HEIGHT"]

    # io.print_table(pd.DataFrame(config, index=["Value"]).T)
    logging.info(io.log_table(config, title="Config:"))

    # Read in example grid file
    grid = io.read_grid("test_data/simple_grid.csv",
                        columns=["Northing", "Easting", "Elev."])

    # grid = grid[["Easting", "Northing"]]

    # io.print_table(grid)
    logging.info(io.log_table(grid, title="Grid points:"))

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
    print(sum([phi["probability"] for phi in theo_phi_steps]))

    # Get wind vectors
    wind_angle = np.radians(0)

    u = config["WIND_SPEED"]*np.cos(wind_angle)
    v = config["WIND_SPEED"]*np.sin(wind_angle)

    # Forward model
    forward_df = inv.gaussian_stack_forward(
        grid, int(config["COL_STEPS"]), config["VENT_ELEVATION"],
        config["PLUME_HEIGHT"], 1, theo_phi_steps, (
            config["ALPHA"], config["BETA"]),
        config["ERUPTION_MASS"],
        (u, v), config["DIFFUSION_COEFFICIENT"], config["EDDY_CONST"],
        config["FALL_TIME_THRESHOLD"]
    )

    forward_df = forward_df.sort_index()
    forward_df["radius"] = np.sqrt(
        forward_df["Easting"]**2 + forward_df["Northing"]**2)

    # io.print_table(forward_df)
    logging.info(io.log_table(forward_df,
                 title="Forward Run Dataframe"))

    t2_df, _, _, _ = io.read_tephra2(
            "./test_data/simple_output.txt")

    t2_df = t2_df.sort_index()
    t2_df["Residual"] = t2_df["MassArea"].values/forward_df["MassArea"].values
    vis.plot_residuals(t2_df, vent=(0, 0),
                       title="Residual plot of Tephra2/Our model output")
    plt.show()

    logging.info(io.log_table(t2_df,
                 title="Tephra2 Run Dataframe"))

    fig, axs = plt.subplots(1, 2, figsize=(
        18, 6), facecolor='w', edgecolor='k')
    axs = axs.ravel()

    vis.plot_sample(forward_df, vent=(0, 0), log=True,
                    show_cbar=True,
                    title="Forward Simulation (Const. Wind)",
                    cbar_label="Mass/Area", ax=axs[0])
    vis.plot_sample(t2_df, vent=(0, 0), log=True,
                    show_cbar=True,
                    title="Tephra2 Simulation (Const. Wind)",
                    cbar_label="Mass/Area", ax=axs[1])
    plt.show()
    # io.print_table(t2_df)
    # Test if parameters are recovered.
    # assert params["a"] == config["ALPHA"], "Parameter ALPHA is not recovered"
    # assert params["b"] == config["BETA"], "Parameter BETA is not recovered"
    # assert params["h1"] == config["PLUME_HEIGHT"], \
    #     "Parameter PLUME_HEIGHT is not recovered"
    # assert params["u"] == u, "Parameter u (wind-x) is not recovered"
    # assert params["v"] == v, "Parameter v (wind-y) is not recovered"
    # assert np.abs(params["D"] - config["DIFFUSION_COEFFICIENT"]) <= 1e-5, \
    #     "Parameter DIFFUSION_COEFFICIENT is not recovered"
    # assert np.abs(params["ftt"] - config["FALL_TIME_THRESHOLD"]) <= 1e-5, \
    #     "Parameter FALL_TIME_THRESHOLD is not recovered"


def test_tephra2_cerronegro():
    # Read in simple set of test parameters
    config = io.read_config_file("test_data/cerronegro_config.txt")

    for key in config.keys():
        config[key] = float(config[key])

    config["MIN_GRAINSIZE"] = int(config["MIN_GRAINSIZE"])
    config["MAX_GRAINSIZE"] = int(config["MAX_GRAINSIZE"])
    config["PART_STEPS"] = int(config["PART_STEPS"])

    vent_x = config["VENT_EASTING"]
    vent_y = config["VENT_NORTHING"]
    config["VENT_EASTING"] = 0
    config["VENT_NORTHING"] = 0
    globs = {
        "LITHIC_DIAMETER_THRESHOLD": 7.,
        "PUMICE_DIAMETER_THRESHOLD": -1.,
        "AIR_VISCOSITY": 0.000018325,
        "AIR_DENSITY":  1.293,
        "GRAVITY": 9.81,
    }
    config["WIND_SPEED"] = 10
    # config["THEO_MAX_COL"] = config["PLUME_HEIGHT"]

    # io.print_table(pd.DataFrame(config, index=["Value"]).T)
    logging.info(io.log_table(config, title="Config:"))

    # Read in example grid file
    grid = io.read_grid("test_data/cerronegro_grid.csv",
                        columns=["Northing", "Easting", "Elev."])

    print(len(grid))
    # grid.drop_duplicates(inplace=True)

    # grid.to_csv("test_data/cerronegro_grid.csv",
    #             sep=" ",
    #             header=False,
    #             index=False)
    # print(len(grid))

    grid = grid[["Easting", "Northing"]]

    logging.info(io.log_table(grid, title="Grid points:"))

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
    print(theo_phi_steps)

    # Get wind vectors
    wind_angle = np.radians(190)

    u = config["WIND_SPEED"]*np.cos(wind_angle)
    v = config["WIND_SPEED"]*np.sin(wind_angle)

    # Forward model
    forward_df = inv.gaussian_stack_forward(
        grid, int(config["COL_STEPS"]), config["VENT_ELEVATION"],
        config["PLUME_HEIGHT"], 1, theo_phi_steps, (
            config["ALPHA"], config["BETA"]),
        config["ERUPTION_MASS"],
        (u, v), config["DIFFUSION_COEFFICIENT"], config["EDDY_CONST"],
        config["FALL_TIME_THRESHOLD"]
    )

    forward_df["radius"] = np.sqrt(
        forward_df["Easting"]**2 + forward_df["Northing"]**2)
    forward_df = forward_df.sort_values(by=['radius'])

    # io.print_table(forward_df)
    logging.info(io.log_table(forward_df,
                 title="Forward Run Dataframe"))

    t2_df, _, _, _ = io.read_tephra2(
            "test_data/cerronegro_output.txt")

    t2_df["Residual"] = forward_df["MassArea"].values/t2_df["MassArea"].values
    vis.plot_residuals(t2_df, vent=(0, 0),
                       title="Residual plot of Tephra2/Our model output")
    plt.show()

    logging.info(io.log_table(t2_df,
                 title="Tephra2 Run Dataframe"))

    fig, axs = plt.subplots(1, 2, figsize=(
        18, 6), facecolor='w', edgecolor='k')
    axs = axs.ravel()

    vis.plot_sample(forward_df, vent=(0, 0), log=True,
                    show_cbar=True,
                    title="Forward Simulation (Const. Wind)",
                    cbar_label="Mass/Area", ax=axs[0])
    vis.plot_sample(t2_df, vent=(0, 0), log=True,
                    show_cbar=True,
                    title="Tephra2 Simulation (Const. Wind)",
                    cbar_label="Mass/Area", ax=axs[1])
    plt.show()
    io.print_table(t2_df)
    # Test if parameters are recovered.
    # assert params["a"] == config["ALPHA"], "Parameter ALPHA is not recovered"
    # assert params["b"] == config["BETA"], "Parameter BETA is not recovered"
    # assert params["h1"] == config["PLUME_HEIGHT"], \
    #     "Parameter PLUME_HEIGHT is not recovered"
    # assert params["u"] == u, "Parameter u (wind-x) is not recovered"
    # assert params["v"] == v, "Parameter v (wind-y) is not recovered"
    # assert np.abs(params["D"] - config["DIFFUSION_COEFFICIENT"]) <= 1e-5, \
    #     "Parameter DIFFUSION_COEFFICIENT is not recovered"
    # assert np.abs(params["ftt"] - config["FALL_TIME_THRESHOLD"]) <= 1e-5, \
    #     "Parameter FALL_TIME_THRESHOLD is not recovered"


def test_tephra2_colima():
    # Read in simple set of test parameters
    config = io.read_config_file("../data/colima/colima_config.txt")

    for key in config.keys():
        config[key] = float(config[key])

    config["MIN_GRAINSIZE"] = int(config["MIN_GRAINSIZE"])
    config["MAX_GRAINSIZE"] = int(config["MAX_GRAINSIZE"])
    config["PART_STEPS"] = int(config["PART_STEPS"])

    vent_x = config["VENT_EASTING"]
    vent_y = config["VENT_NORTHING"]
    config["VENT_EASTING"] = 0
    config["VENT_NORTHING"] = 0
    globs = {
        "LITHIC_DIAMETER_THRESHOLD": 7.,
        "PUMICE_DIAMETER_THRESHOLD": -1.,
        "AIR_VISCOSITY": 0.000018325,
        "AIR_DENSITY":  1.293,
        "GRAVITY": 9.81,
    }
    config["WIND_SPEED"] = 10.345736331007556
    # config["THEO_MAX_COL"] = config["PLUME_HEIGHT"]

    # io.print_table(pd.DataFrame(config, index=["Value"]).T)
    logging.info(io.log_table(config, title="Config:"))

    # Read in example grid file
    grid = io.read_grid("test_data/colima_grid.csv")

    # grid = grid[["Easting", "Northing"]]

    logging.info(io.log_table(grid, title="Grid points:"))

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
    print(sum([phi["probability"] for phi in theo_phi_steps]))

    # Get wind vectors
    wind_angle = np.radians(40.7454850322874)

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
    forward_df = forward_df.sort_index()

    forward_df["radius"] = np.sqrt(
        forward_df["Easting"]**2 + forward_df["Northing"]**2)
    forward_df = forward_df.sort_values(by=['radius'])

    # io.print_table(forward_df)
    logging.info(io.log_table(forward_df,
                 title="Forward Run Dataframe"))

    t2_df, _, _, _ = io.read_tephra2(
            "../data/colima/connor_marks_wind_out.txt")

    t2_df = t2_df.sort_index()
    t2_df["Residual"] = t2_df["MassArea"].values/forward_df["MassArea"].values
    vis.plot_residuals(t2_df, vent=(0, 0),
                       title="Residual plot of Tephra2/Our model output")
    plt.show()

    logging.info(io.log_table(t2_df,
                 title="Tephra2 Run Dataframe"))

    fig, axs = plt.subplots(1, 2, figsize=(
        18, 6), facecolor='w', edgecolor='k')
    axs = axs.ravel()

    vis.plot_sample(forward_df, vent=(0, 0), log=True,
                    show_cbar=True,
                    title="Forward Simulation (Const. Wind)",
                    cbar_label="Mass/Area", ax=axs[0])
    vis.plot_sample(t2_df, vent=(0, 0), log=True,
                    show_cbar=True,
                    title="Tephra2 Simulation (Const. Wind)",
                    cbar_label="Mass/Area", ax=axs[1])
    plt.show()
    io.print_table(t2_df)
    # Test if parameters are recovered.
    # assert params["a"] == config["ALPHA"], "Parameter ALPHA is not recovered"
    # assert params["b"] == config["BETA"], "Parameter BETA is not recovered"
    # assert params["h1"] == config["PLUME_HEIGHT"], \
    #     "Parameter PLUME_HEIGHT is not recovered"
    # assert params["u"] == u, "Parameter u (wind-x) is not recovered"
    # assert params["v"] == v, "Parameter v (wind-y) is not recovered"
    # assert np.abs(params["D"] - config["DIFFUSION_COEFFICIENT"]) <= 1e-5, \
    #     "Parameter DIFFUSION_COEFFICIENT is not recovered"
    # assert np.abs(params["ftt"] - config["FALL_TIME_THRESHOLD"]) <= 1e-5, \
    #     "Parameter FALL_TIME_THRESHOLD is not recovered"


if __name__ == "__main__":
    # test_internal_consistency()
    # test_tephra2_colima()
    test_tephra2_simple()
    # test_tephra2_cerronegro()
