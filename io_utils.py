import os
import pandas as pd
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

    return df, phi_labels, phi_limits, phi_centroids