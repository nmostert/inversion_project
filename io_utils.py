import os
import pandas as pd
import numpy as np
import re

def read_tephra2(filename):
    # Reads output from Tephra2 into a Pandas DataFrame
    # Extracts phi-classes from header constructs column names

    headers = pd.read_csv(filename, delim_whitespace=True, header=None, nrows=1)
    phi_labels = []
    phi_limits = []
    phi_centroids = []
    for name in headers[headers.columns[4:-1]].values[0]:
        m1 = re.search(r'[-+]?[0-9]*\.?[0-9]+(?=->)', name)
        m2 = re.search(r'[-+]?[0-9]*\.?[0-9]+(?=\))', name)
        phi_labels.append("[%.3g,%.3g)"%(float(m1.group(0)), float(m2.group(0))))
        phi_limits.append((m1.group(0), m2.group(0)))
        phi_centroids.append((float(m1.group(0)) + float(m2.group(0))) / 2)
    col_names = ["Easting", "Northing", "Elevation",
                 "MassArea"] + phi_labels + ["Percent"]

    df = pd.read_csv(filename, delim_whitespace=True, header=None,
                     names=col_names, skiprows=1)
    df = df.dropna(axis=1, how='all')
    df = df.fillna(0)

    df["radius"] = np.sqrt(df["Easting"]**2 + df["Northing"]**2)
    df = df.sort_values(by=['radius'])

    return df, phi_labels, phi_limits, phi_centroids

def read_tephra2_config(filename):
    config = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line=="" and not line.startswith("#"):
                (key, val) = line.split()
                config[str(key)] = float(val)

    config["COL_STEPS"] = int(config["COL_STEPS"])
    config["PART_STEPS"] = int(config["PART_STEPS"])

    return config

def import_colima(filename):
    raw_df = pd.read_csv(filename)

    phi_labels = [
        "[-5,-4)",
        "[-4,-3)" ,
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
