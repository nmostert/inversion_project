import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import project.io_utils as io
import project.inversion as inv

globs = {
    "LITHIC_DIAMETER_THRESHOLD": 7.,
    "PUMICE_DIAMETER_THRESHOLD": -1.,
    "AIR_VISCOSITY": 0.000018325,
    "AIR_DENSITY":  1.293,
    "GRAVITY": 9.81,
}

pd.options.display.float_format = '{:,g}'.format
plt.rcParams['text.usetex'] = True
plt.style.use(['ggplot'])

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("config",
                    help="Path to the Tephra2 configuration file used for \
                            forward simulation.")
parser.add_argument("add_params",
                    help="Path to the additional parameter file used in the \
                            forward modelling. See Readme.MD for usage.")
parser.add_argument("grid",
                    help="Path to the accumulation points to use in the final \
                            output.")
parser.add_argument("output",
                    help="Output file used to store completed forward run")
parser.add_argument("-i", "--inversion",
                    help="Calculate, use and report inversion parameters.\
                            (Use if this forward run will be inverted)",
                    action="store_true")
parser.add_argument("-v", "--verbose",
                    help="Print logs and data to command line.",
                    action="store_true")

args = parser.parse_args()


if __name__ == "__main__":
    # Read Tephra2 Con fig
    config = io.read_tephra2_config(args.config)

    # Read Additional Configuration Parameters
    add_params = io.read_add_param_config(args.add_params)

    if args.verbose:
        io.print_table(config)
        io.print_table(add_params)
        io.print_table(globs)

    # Calculate and report inversion steps
    if args.inversion:
        layer_thickness = (
            (config["PLUME_HEIGHT"] - config["VENT_ELEVATION"]) /
            config["COL_STEPS"])
        inversion_steps = np.round((config["COL_STEPS"] *
                                    (add_params["MAXIMUM_COLUMN_HEIGHT"] -
                                    config["VENT_ELEVATION"])) /
                                   (config["PLUME_HEIGHT"] -
                                    config["VENT_ELEVATION"]))

        closest_H = ((inversion_steps*(config["PLUME_HEIGHT"] -
                      config["VENT_ELEVATION"])) /
                     config["COL_STEPS"]) + config["VENT_ELEVATION"]

        if args.verbose:
            print("INVERSION PARAMETER CALCULATION")
            print("===============================")
            print("Use this number of columns steps to for perfect inversion:")
            print(inversion_steps)
            print("(This number needs to be low enough to invert efficiently)")
            print("If not, decrease COL_STEPS or MAXIMUM_COLUMN_HEIGHT")
            print("Closest Possible Theoretical Max Column Height:")
            print(closest_H)

        add_params["INV_STEPS"] = int(inversion_steps)
        add_params["THEO_MAX_COL"] = closest_H
    else:
        add_params["INV_STEPS"] = int(config["COL_STEPS"])
        add_params["THEO_MAX_COL"] = int(config["MAXIMUM_COLUMN_HEIGHT"])

    # Read Grid

    grid = io.read_grid(args.grid)

    if args.verbose:
        io.print_table(grid)

    # Calculate Phi Steps
    theo_phi_steps = inv.get_phi_steps(config["MIN_GRAINSIZE"],
                                       config["MAX_GRAINSIZE"],
                                       config["PART_STEPS"],
                                       config["MEDIAN_GRAINSIZE"],
                                       config["STD_GRAINSIZE"],
                                       globs["LITHIC_DIAMETER_THRESHOLD"],
                                       globs["PUMICE_DIAMETER_THRESHOLD"],
                                       config["LITHIC_DENSITY"],
                                       config["PUMICE_DENSITY"])
    step_width = (config["MAX_GRAINSIZE"] -
                  config["MIN_GRAINSIZE"])/config["PART_STEPS"]

    if args.verbose:
        io.print_table(pd.DataFrame(theo_phi_steps))

    # Perform Forward Run

    u = add_params["WIND_SPEED"]*np.cos(add_params["WIND_ANGLE"])
    v = add_params["WIND_SPEED"]*np.sin(add_params["WIND_ANGLE"])

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

    if args.verbose:
        io.print_table(forward_df)

    # Save output file
    forward_df.to_csv(args.output, index=False)
