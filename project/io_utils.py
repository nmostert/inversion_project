import pandas as pd
import numpy as np
import re
from tabulate import tabulate

pd.options.display.float_format = '{:,g}'.format


def read_tephra2(filename):
    # Reads output from Tephra2 into a Pandas DataFrame
    # Extracts phi-classes from header constructs column names

    headers = pd.read_csv(filename, delim_whitespace=True,
                          header=None, nrows=1)
    phi_labels = []
    phi_limits = []
    phi_centroids = []
    for name in headers[headers.columns[4:-1]].values[0]:
        m1 = re.search(r'[-+]?[0-9]*\.?[0-9]+(?=->)', name)
        m2 = re.search(r'[-+]?[0-9]*\.?[0-9]+(?=\))', name)
        phi_labels.append("[%.3g,%.3g)" %
                          (float(m1.group(0)), float(m2.group(0))))
        phi_limits.append((m1.group(0), m2.group(0)))
        phi_centroids.append((float(m1.group(0)) + float(m2.group(0))) / 2)
    col_names = ["Northing", "Easting", "Elevation",
                 "MassArea"] + phi_labels + ["Percent"]

    df = pd.read_csv(filename, delim_whitespace=True, header=None,
                     names=col_names, skiprows=1)
    df = df.dropna(axis=1, how='all')
    df = df.fillna(0)

    df["radius"] = np.sqrt(df["Easting"]**2 + df["Northing"]**2)
    df = df.sort_values(by=['radius'])

    return df, phi_labels, phi_limits, phi_centroids


def read_config_file(filename):
    config = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line == "" and not line.startswith("#"):
                (key, val) = line.split()
                config[str(key)] = val

    return config


def import_cerronegro(filename, layers=["A", "M"]):
    raw_df = pd.read_csv(filename)

    phi_labels = [
        "[-4.5,-4)",
        "[-4,-3.5)",
        "[-3.5,-3)",
        "[-3,-2.5)",
        "[-2.5,-2)",
        "[-2,-1.5)",
        "[-1.5,-1)",
        "[-1,-0.5)",
        "[-0.5,0)",
        "[0,0.5)",
        "[0.5,1)",
        "[1,1.5)",
        "[1.5,2)",
        "[2,2.5)",
        "[2.5,3)",
        "[3,3.5)",
    ]

    # phi_labels = [
    #     "[-5,-4)",
    #     "[-4,-3)",
    #     "[-3,-2)",
    #     "[-2,-1)",
    #     "[-1,0)",
    #     "[0,1)",
    #     "[1,2)",
    #     "[2,3)",
    #     "[3,4)",
    # ]

    ventx = 532596
    venty = 1381862

    raw_df["Easting"] = raw_df["Easting"] - ventx
    raw_df["Northing"] = raw_df["Northing"] - venty

    for phi in phi_labels:
        raw_df[phi] = (raw_df[phi].values)*100

    raw_df["radius"] = np.sqrt(raw_df["Easting"]**2 + raw_df["Northing"]**2)
    raw_df = raw_df.sort_values(by=['radius'])

    grid = raw_df[["Easting", "Northing"]].copy()
    grid["Elevation"] = np.ones(len(grid))

    df = raw_df[raw_df["Layer"].isin(layers)]
    grid = grid[raw_df["Layer"].isin(layers)]

    return df, grid


def import_colima(filename):
    raw_df = pd.read_csv(filename)

    phi_labels = [
        "[-5,-4)",
        "[-4,-3)",
        "[-3,-2)",
        "[-2,-1)",
        "[-1,0)",
        "[0,1)",
        "[1,2)",
        "[2,3)",
        "[3,4)"
    ]

    ventx = 645110
    venty = 2158088

    raw_df["Easting"] = raw_df["Easting"] - ventx
    raw_df["Northing"] = raw_df["Northing"] - venty

    for phi in phi_labels:
        raw_df[phi] = (raw_df[phi].values)*100

    raw_df["radius"] = np.sqrt(raw_df["Easting"]**2 + raw_df["Northing"]**2)
    raw_df = raw_df.sort_values(by=['radius'])

    grid = raw_df[["Easting", "Northing"]].copy()
    grid["Elevation"] = np.zeros(len(grid))

    return raw_df, grid


def import_pululagua(filename):
    raw_df = pd.read_csv(filename)

    phi_labels = [
        "[-8,-7)",
        "[-7,-6)",
        "[-6,-5)",
        "[-5,-4)",
        "[-4,-3)",
        "[-3,-2)",
        "[-2,-1)",
        "[-1,0)",
        "[0,1)",
        "[1,2)",
        "[2,3)",
        "[3,4)",
        "[4,5)",
        "[5,6)",
        "[6,7)",
        "[7,8)",
        "[8,9)",
        "[9,10)",
        "[10,11)"
    ]

    ventx = 780000
    venty = 10005500

    raw_df["Easting"] = raw_df["Easting"] - ventx
    raw_df["Northing"] = raw_df["Northing"] - venty

    raw_df["radius"] = np.sqrt(raw_df["Easting"]**2 + raw_df["Northing"]**2)
    raw_df = raw_df.sort_values(by=['radius'])

    grid = raw_df[["Easting", "Northing"]].copy()
    grid["Elevation"] = np.zeros(len(grid))
    return raw_df, grid


def read_grid(filename, columns=None):
    """Reads csv grid file as pandas dataframe

    Parameters
    ----------
    filename :
        filename of .csv grid file

    Returns
    -------
    grid_df :
        grid in Pandas DataFrame
    """
    if columns is None:
        columns = ["Easting", "Northing", "Elev."]
    grid_df = pd.read_csv(filename, names=columns,
                          sep=r'\s+')
    return grid_df


def print_table(data, tablefmt="fancy_grid"):
    """Prints a pretty table from a dataframe.

    Parameters
    ----------
    data :
        data
    format :
        format
    """
    if isinstance(data, pd.DataFrame):
        print(tabulate(data, headers="keys", tablefmt=tablefmt))
    if isinstance(data, dict):
        print(tabulate(pd.DataFrame(data, index=["Values"]).T,
              tablefmt=tablefmt))


def log_table(data, title="", tablefmt="plain"):
    """Prepares a pretty table from a dataframe.

    # TODO: Apparently multiline logging is bad practice because it breaks
    log processors.#
    Parameters
    ----------
    data :
        data
    format :
        format
    """
    if isinstance(data, pd.DataFrame):
        return title + "\n" + tabulate(data, headers="keys", tablefmt=tablefmt,
                                       floatfmt=".4g")
    if isinstance(data, dict):
        return title + "\n" + tabulate(pd.DataFrame(data, index=["Values"]).T,
                                       tablefmt=tablefmt)
