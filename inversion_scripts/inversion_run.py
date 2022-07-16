import argparse
import json
from scipy.stats import truncnorm, norm, uniform
import numpy as np
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

parser = argparse.ArgumentParser()
parser.add_argument("data", help="Path to the data to be inverted")

parser.add_argument("eruption_parser", help="Eruption method to use for \
                    data import.", choices=["tephra2", "colima", "CN", "pulu"])

parser.add_argument("inv_params", help="Path to a file containing inversion \
                                        parameters for the inversion run.")

parser.add_argument("-t", "--tgsd_method", help="TGSD method to use for \
                    inversion.", choices=["naive", "voronoi"],
                    default="naive")
parser.add_argument("tgsd_config", help="Path to TGSD parameter config file \
                                         used to construct initial TGSD. \
                                         extracted from the data using the \
                                         'tgsd_method', the default of which \
                                         is 'naive'.")
parser.add_argument("param_config", help="JSON file specifying the \
                                         configuration of the inversion \
                                         parameters")
parser.add_argument("output", help="Path to output file in which to store \
                                    solution candidates.")
parser.add_argument("-g", "--grid",
                    help="Path to the accumulation points to use in the final \
                            output. If unspecified, the grid is extracted from\
                            the data file.")
parser.add_argument("-v", "--verbose",
                    help="Print logs and data to command line.",
                    action="store_true")

args = parser.parse_args()


def col_truncnorm(mean, top=45000):
    bottom = 5000
    std = (top - bottom)/4
    standard_a, standard_b = (bottom - mean) / std, (top - mean) / std
    return truncnorm.rvs(standard_a, standard_b, loc=mean, scale=std)


def lognorm(prior, bottom=0):
    std_norm = norm.rvs()
    std_lognorm = np.exp(std_norm)
    lognorm = bottom + (prior-bottom)*std_lognorm
    return lognorm


def normal(prior):
    std_norm = norm.rvs()
    normal = prior*std_norm
    return normal


def uninformed(bottom, top):
    unif = uniform.rvs(loc=bottom, scale=(top-bottom))
    return unif


methods = {
        "col_truncnorm": col_truncnorm,
        "lognorm": lognorm,
        "normal": normal,
        "uniform": uninformed
        }

if __name__ == "__main__":
    # Read in data file

    filename = args.data
    if args.eruption_parser == "CN":
        data, grid_raw = io.import_cerronegro(filename)
    elif args.eruption_parser == "colima":
        data, grid_raw = io.import_colima(filename)
    elif args.eruption_parser == "pulu":
        data, grid_raw = io.import_pululagua(filename)
    elif args.eruption_parser == "tephra2":
        data, grid_raw = io.import_tephra2(filename)

    if args.verbose:
        io.print_table(data)

    # Read in grid file (if grid is specified)
    if args.grid is not None:
        grid = io.read_grid(args.grid)
    else:
        grid = grid_raw

    if args.verbose:
        io.print_table(grid)

    # Read in inversion parameters
    config = io.read_config_file(args.inv_params)

    config["INV_STEPS"] = int(config["INV_STEPS"])
    config["THEO_MAX_COL"] = float(config["THEO_MAX_COL"])
    config["VENT_ELEVATION"] = float(config["VENT_ELEVATION"])
    config["ELEVATION"] = float(config["ELEVATION"])
    config["ERUPTION_MASS"] = float(config["ERUPTION_MASS"])
    config["PRE_SAMPLES"] = int(config["PRE_SAMPLES"])
    config["SOL_ITER"] = int(config["SOL_ITER"])
    config["MAX_ITER"] = int(config["MAX_ITER"])
    config["TOL"] = float(config["TOL"])
    config["ADJUST_TGSD"] = bool(config["ADJUST_TGSD"])
    config["RUNS"] = int(config["RUNS"])

    if args.verbose:
        io.print_table(config)

    # Read or construct TGSD
    if args.tgsd_config is not None:
        tgsd_config = io.read_config_file(args.tgsd_config)

        tgsd_config["MIN_GRAINSIZE"] = float(tgsd_config["MIN_GRAINSIZE"])
        tgsd_config["MAX_GRAINSIZE"] = float(tgsd_config["MAX_GRAINSIZE"])
        if "MEDIAN_GRAINSIZE" in tgsd_config:
            tgsd_config["MEDIAN_GRAINSIZE"] = \
                        float(tgsd_config["MEDIAN_GRAINSIZE"])
        if "STD_GRAINSIZE" in tgsd_config:
            tgsd_config["STD_GRAINSIZE"] = float(tgsd_config["STD_GRAINSIZE"])
        tgsd_config["PART_STEPS"] = int(tgsd_config["PART_STEPS"])
        tgsd_config["LITHIC_DENSITY"] = float(tgsd_config["LITHIC_DENSITY"])
        tgsd_config["PUMICE_DENSITY"] = float(tgsd_config["PUMICE_DENSITY"])
        if args.tgsd_method == "naive":
            phi_steps = inv.get_naive_tgsd(data, tgsd_config["MIN_GRAINSIZE"],
                                           tgsd_config["MAX_GRAINSIZE"],
                                           tgsd_config["PART_STEPS"],
                                           globs["LITHIC_DIAMETER_THRESHOLD"],
                                           globs["PUMICE_DIAMETER_THRESHOLD"],
                                           tgsd_config["LITHIC_DENSITY"],
                                           tgsd_config["PUMICE_DENSITY"])
        elif args.tgsd_method == "uniform":
            phi_steps = \
                    inv.get_uniform_tgsd(tgsd_config["MIN_GRAINSIZE"],
                                         tgsd_config["MAX_GRAINSIZE"],
                                         tgsd_config["PART_STEPS"],
                                         globs["LITHIC_DIAMETER_THRESHOLD"],
                                         globs["PUMICE_DIAMETER_THRESHOLD"],
                                         tgsd_config["LITHIC_DENSITY"],
                                         tgsd_config["PUMICE_DENSITY"])
        elif args.tgsd_method == "voronoi":
            print("Voronoi TGSD estimation method not yet implemented")
        else:
            if "MEDIAN_GRAINSIZE" not in tgsd_config:
                print("MEDIAN_GRAINSIZE parameter not specified, required for \
                       theoretical construction of TGSD")
            if "STD_GRAINSIZE" not in tgsd_config:
                print("STD_GRAIN SIZE parameter not specified, required for \
                       theoretical construction of TEST")

            phi_steps = inv.get_phi_steps(config["MIN_GRAINSIZE"],
                                          config["MAX_GRAINSIZE"],
                                          config["PART_STEPS"],
                                          config["MEDIAN_GRAINSIZE"],
                                          config["STD_GRAINSIZE"],
                                          globs["LITHIC_DIAMETER_THRESHOLD"],
                                          globs["PUMICE_DIAMETER_THRESHOLD"],
                                          config["LITHIC_DENSITY"],
                                          config["PUMICE_DENSITY"])
    else:
        print("TGSD parameters need to be specified in a TGSD config file, \
                regardless of TGSD method used.")

    # Read in Param Config
    with open(args.param_config) as json_file:
        param_config = json.load(json_file)

    if args.verbose:
        print(json.dumps(param_config, indent=4))

    for i, (key, val) in enumerate(param_config.items()):
        val["sample_function"] = methods[val["sample_function"]]

    # Perform Inversion
    layer_thickness = (
        (config["THEO_MAX_COL"]-config["VENT_ELEVATION"])/config["INV_STEPS"])

    out = inv.gaussian_stack_multi_run(
        data,
        len(data),
        config["INV_STEPS"],
        config["VENT_ELEVATION"],
        config["THEO_MAX_COL"],
        config["ELEVATION"],
        phi_steps,
        config["ERUPTION_MASS"],
        param_config,
        runs=config["RUNS"],
        column_cap=config["THEO_MAX_COL"],
        pre_samples=config["PRE_SAMPLES"],
        sol_iter=config["SOL_ITER"],
        max_iter=config["MAX_ITER"],
        tol=config["TOL"],
        termination=config["TERMINATION_METHOD"],
        adjust_TGSD=config["ADJUST_TGSD"],
        adjust_mass=False,
        gof=config["MISFIT_METHOD"],
        adjustment_factor=None,
        abort_method="too_slow",
        file_prefix=config["FILE_PREFIX"]
    )
    inverted_masses_list, misfit_list, params_list, priors_list, heights, \
        tgsd_list, mass_list, status_list, message_list = out

    if args.verbose:
        print(misfit_list)

    # Save output to file


