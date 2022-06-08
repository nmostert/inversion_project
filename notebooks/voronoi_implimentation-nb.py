# %%
"""
# Voronoi TGSD estimation method implimentation
"""

# %%
import copy
from tabulate import tabulate
# from pandas import ExcelWriter
from functools import reduce
from time import process_time
from scipy.optimize import minimize
from scipy.sparse.linalg import lsqr
from scipy.stats import beta, expon, truncnorm, norm, uniform
from scipy.spatial import Voronoi, ConvexHull, voronoi_plot_2d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import project.vis_utils as vis
import project.io_utils as io
import project.inversion as inv
import matplotlib.colors as colors
import math


pd.options.display.float_format = '{:,g}'.format
plt.style.use(['ggplot'])

import logging

logging.basicConfig(level=logging.INFO)


# %%
""" ### Observation Data

Reading in Colima observation dataset and extracting grid
"""

# %%
filename = "../data/colima/colima_real_data.csv"

raw_df, grid = io.import_colima(filename)


# grid.to_csv("../data/colima/colima_grid.csv",
#             sep=" ", header=False, index=False)
grid = io.read_grid("../data/colima/colima_grid.csv")

io.print_table(raw_df.head())

io.print_table(grid)

# %%

x = grid["Easting"].values
y = grid["Northing"].values

points = np.array(list(zip(x, y)))
print(points)
vor = Voronoi(points)
print(vor.regions)

fig = voronoi_plot_2d(vor)
plt.show()

# %%


def voronoi_volumes(points):
    v = Voronoi(points)
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices:  # some regions can be opened
            vol[i] = np.inf
        else:
            vol[i] = ConvexHull(v.vertices[indices]).volume
    return vol


print(voronoi_volumes(points))


# %%

hull = ConvexHull(points)

plt.plot(points[:, 0], points[:, 1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
plt.show()                                                                        


def PointsInCircum(eachPoint, r, n=100):
    return [(eachPoint[0] + math.cos(2*math.pi/n*x)*r, eachPoint[1] +
             math.sin(2 * math.pi/n*x)*r) for x in range(0, n+1)]


def bufferPoints(inPoints, stretchCoef, n):
    newPoints = []
    for eachPoint in inPoints:
        newPoints += PointsInCircum(eachPoint, stretchCoef, n)
    newPoints = np.array(newPoints)
    newBuffer = ConvexHull(newPoints)

    return newPoints[newBuffer.vertices]


# %%

dilateCoef = 3000
pointsDilated = bufferPoints(points[hull.vertices], dilateCoef, n=10)
plt.scatter(points[:, 0], points[:, 1], color='b')
plt.scatter(pointsDilated[:, 0], pointsDilated[:, 1], color='r')
plt.show()

# %%

all_points = np.append(points, pointsDilated, axis=0)

print(all_points)

all_vor = Voronoi(all_points)

fig = voronoi_plot_2d(all_vor)
plt.show()


# %%


vor_volumes = voronoi_volumes(all_points)


raw_df["vor_areas"] = vor_volumes[:len(raw_df)]

print(raw_df)

# %%


def get_voronoi_tgsd(
    df, min_grainsize, max_grainsize, part_steps,
    lithic_diameter_threshold, pumice_diameter_threshold,
    lithic_density, pumice_density
):
    """Calculate naive estimation of TGSD from a given deposit.

    Parameters
    ----------
    df : Dataframe
        Pandas data frame of deposit dataset.
    min_grainsize : float
        Smallest grainsize (largest phi)
    max_grainsize : float
        Largest grainsize (smallest phi)
    part_steps : float
        TGSD bin width in phi.
    lithic_diameter_threshold : float
        Particles smaller (or higher in phi) than this threshold are given
        lithic density.
    pumice_diameter_threshold : float
        Particles larger (or lower in phi) than this threshold are given
        pumice density.
    lithic_density : float
        lithic density (in kg/m^3)
    pumice_density : float
        pumice density (in kg/m^3)

    Returns
    -------
    phi_steps :
        List of dicts with descriptions of each size class:
        "lower": Smallest grainsize in bin (largest phi)
        "upper": Largest grain size in bin (smallest phi)
        "interval": Interval notation string of phi class.
        "centroid": Centroid of grain size class in phi.
        "density": Density of particles in phi class.
        "probability": Probability of pdf bin.
    """
    part_section_width = min_grainsize - max_grainsize
    part_step_width = part_section_width / part_steps

    phi_steps = []

    y = max_grainsize
    for i in range(part_steps):
        if y > lithic_diameter_threshold:
            particle_density = lithic_density
        elif y < pumice_diameter_threshold:
            particle_density = pumice_density
        else:
            particle_density = lithic_density - \
                (lithic_density - pumice_density) * \
                (y - lithic_diameter_threshold) /\
                (pumice_diameter_threshold - lithic_diameter_threshold)
        interval = "[%.3g,%.3g)" % (y, y + part_step_width)

        prob = sum((df[interval]/100)*df["MassArea"]*df["vor_areas"])

        phi_class = {
            "lower": y,
            "upper": y + part_step_width,
            "interval": interval,
            "centroid": (y + (y + part_step_width))/2,
            "density": particle_density,
            "probability": prob
        }
        phi_steps.append(phi_class)
        y += part_step_width

    # Normalise
    total_prob = sum([phi["probability"] for phi in phi_steps])
    for phi in phi_steps:
        phi["probability"] = phi["probability"] / total_prob

    return phi_steps


# %%

vor_phi_steps = get_voronoi_tgsd(
        raw_df,
        4,
        -5,
        9,
        7.,
        -1.,
        2700,
        1024)

step_width = (-5 - 4)/9

io.print_table(pd.DataFrame(vor_phi_steps))


probs = [phi["probability"] for phi in vor_phi_steps]
x = [phi["centroid"] for phi in vor_phi_steps]
labels = [phi["interval"] for phi in vor_phi_steps]
fig, ax = plt.subplots(facecolor='w', edgecolor='k')
ax.bar(x, probs, width=1, align="center")
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.xlabel("Phi Interval")
plt.show()

# %%

naive_phi_steps = inv.get_naive_tgsd(
        raw_df,
        4,
        -5,
        9,
        7.,
        -1.,
        2700,
        1024)

step_width = (-5 - 4)/9

io.print_table(pd.DataFrame(vor_phi_steps))


probs = [phi["probability"] for phi in vor_phi_steps]
x = [phi["centroid"] for phi in vor_phi_steps]
labels = [phi["interval"] for phi in vor_phi_steps]
fig, ax = plt.subplots(facecolor='w', edgecolor='k')
ax.bar(x, probs, width=1, align="center", color="b")
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.xlabel("Phi Interval")
plt.show()
