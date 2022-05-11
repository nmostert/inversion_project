import argparse

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
parser.add_argument("aux_params", help="Path to a file containing auxiliary \
                                        parameters for the inversion run.")
parser.add_argument("tgsd_method", help="TGSD method to use for inversion.")
parser.add_argument("param_config", help="JSON (?) file specifying the \
                                         configuration of the inversion \
                                         parameters")
parser.add_argument("output", help="Path to output file in which to store \
                                    solution candidates."
parser.add_argument("-g", "--grid",
                    help="Path to the accumulation points to use in the final \
                            output. If unspecified, the grid is extracted from\
                            the data file.")
parser.add_argument("-v", "--verbose",
                    help="Print logs and data to command line.",
                    action="store_true")

args = parser.parse_args()

if __name__ == "__main__":
    # Read in data file
     
    raw_df, grid_raw = io.import_cerronegro(filename, layers=["A", "M"])

grid = io.read_grid("../data/cerronegro/cerronegro_grid_layers_AM.csv")

io.print_table(raw_df["Mass"])

io.print_table(grid)
io.print_table(grid_raw)
print(len(grid))
print(len(grid_raw))

    # Read in grid file (if grid is specified)

    # Read in auxiliary parameters

    # Read or construct TGSD

    # Read in Param Config

    # Perform Inversion

    # Save output to file


