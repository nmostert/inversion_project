# %%
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import beta, expon, truncnorm, norm, uniform
from scipy.sparse.linalg import lsqr
from scipy.optimize import minimize
from time import process_time
from functools import reduce
from pandas import ExcelWriter
from pandas import ExcelFile
import xlrd
import re
import copy
from matplotlib import cm

import sys
sys.path.append("..")
import project.vis_utils as vis
import project.io_utils as io
import project.inversion as inv
import matplotlib.colors as colors

pd.options.display.float_format = '{:,g}'.format
plt.style.use(['ggplot'])

# %%
filename = "../data/colima/colima_real_data.csv"

raw_df, grid = io.import_colima(filename)


# grid.to_csv("../data/colima/colima_grid.csv",
#             sep=" ", header=False, index=False)
grid = io.read_grid("../data/colima/colima_grid.csv")

io.print_table(raw_df.head())

io.print_table(grid)
# plt.savefig("colima/all_points.png", dpi=200, format='png')

obs_df = raw_df.copy()

# obs_df = obs_df.drop([16, 18, 17, 37, 31])
#This is going to be 100 across. Just needs to be the same as the other dfs.
obs_df["Residual"] = obs_df["MassArea"]/obs_df["MassArea"]*100

obs_df["radius"] = np.sqrt(obs_df["Easting"]**2 + obs_df["Northing"]**2)
obs_df = obs_df.sort_values(by=['radius'])

vis.plot_sample(obs_df, vent=(0,0), log=False, title="Preprocessed Data",
        cbar_label="Mass/Area")
# plt.savefig("colima/All_trans.png", dpi=200, format='png')
plt.show()

# %%
""" ## Set up inversion configuration

- Read in Tephra2 config file
- Set global parameters
- Set other inversion parameters

# Internal consistency requirements

In order to demonstrate perfect internal consistency we need our forward model
to invert perfectly, i.e. return the exact input parameters that simulated it.
Such a perfect inversion requires that the release points in the column be in
exactly the same place in the forward model as in the inversion.

The forward model evenly spaces the release points between the vent height and
the plume height, putting a point at each. In order to demonstrate internal
consistency, those same points need to be used for inversion. To this end, a
theoretical maximum column height and number of inversion column steps is
calculated that places release points for inversion at those same points in the
column that would be used in the forward model, while adding evenly spaced
points above the actual plume height.
If internal consistency is not required these values can be set for efficient
inversion.
"""

# %%
config = io.read_tephra2_config("../data/colima/colima_config.txt")

globs = {
    "LITHIC_DIAMETER_THRESHOLD": 7.,
    "PUMICE_DIAMETER_THRESHOLD": -1.,
    "AIR_VISCOSITY": 0.000018325,
    "AIR_DENSITY":  1.293,
    "GRAVITY": 9.81,
}

# Update parameters
# COL STEPS need to be small enough so the
# layer height can be kept for an inversion with a high H
config["COL_STEPS"] = 20
theoretical_max = 45000  # The H value will be as close as possible to this
layer_thickness = (
    (config["PLUME_HEIGHT"]-config["VENT_ELEVATION"])/config["COL_STEPS"])

inversion_steps = np.round((config["COL_STEPS"]*(theoretical_max -
                            config["VENT_ELEVATION"])) /
                           (config["PLUME_HEIGHT"]-config["VENT_ELEVATION"]))
closest_H = ((inversion_steps*(config["PLUME_HEIGHT"] -
             config["VENT_ELEVATION"])) /
             config["COL_STEPS"]) + config["VENT_ELEVATION"]
print("This number needs to be low enough to invert efficiently:")
print(inversion_steps)
print("If not, decrease COL_STEPS or theoretical max")
print("Closest Possible Theoretical Max Column Height:")
print(closest_H)

config["PART_STEPS"] = 9

config["MAX_GRAINSIZE"] = -5
config["MIN_GRAINSIZE"] = 4

add_params = {
    # Constant wind speed (m/s)
    "WIND_SPEED": 10.345736331007556,
    # Constant wind angle (radians anti-clockwise from Easting)
    "WIND_ANGLE": np.radians(40.7454850322874),
    "INV_STEPS": int(inversion_steps),
    "THEO_MAX_COL": closest_H
}
# To ensure monotonicity:
# config["DIFFUSION_COEFFICIENT"] = 1.7*config["FALL_TIME_THRESHOLD"]

print("INPUT PARAMETERS:")
io.print_table(config, tablefmt="latex")
io.print_table(globs, tablefmt="latex")
io.print_table(add_params, tablefmt="latex")

# %%
""" ## Phi class calculations

# Theoretical phi parameters

The function `get_phi_steps` generates phi classes using Tephra2 input
variables in the exact same way as Tephra2.
"""

# %%
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

io.print_table(pd.DataFrame(theo_phi_steps))


probs = [phi["probability"].copy() for phi in theo_phi_steps]
x = [phi["centroid"] for phi in theo_phi_steps]
labels = [phi["interval"] for phi in theo_phi_steps]
fig, ax = plt.subplots(facecolor='w', edgecolor='k')
ax.bar(x, probs, width=1, align="center")
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.xlabel("Phi Interval")
plt.show()

# %%
sep_phis = []

fig, axs = plt.subplots(3,3, figsize=(
            15, 15), facecolor='w', edgecolor='k')
axs = axs.ravel()

for i, phi in enumerate(theo_phi_steps):
    df, _, _, _ = io.read_tephra2("data/colima/colima_gs_trial/colima_%d_out.txt"%i)
    sep_phis += [df]
    vis.plot_sample(sep_phis[i], vent=(0,0), ax=axs[i], 
                    title="%s"%phi["interval"], cbar_label="Mass/Area")
plt.tight_layout()
plt.show()

print(obs_df.index)

# %%
observation_phis = []
posterior_phis = []

prior_phi_steps = theo_phi_steps.copy()

for j, phi_step in enumerate(theo_phi_steps):
    phi_obs = obs_df.copy()
    phi_obs["MassArea"] = phi_obs["MassArea"]*phi_obs[phi_step["interval"]]/100
    phi_obs[phi_step["interval"]] = 100
    bounds=(0, 200)

    observation_phis += [phi_obs]

    phi_post = sep_phis[j].copy()
    phi_post["radius"] = np.sqrt(phi_post["Easting"]**2 + phi_post["Northing"]**2)
    phi_post = phi_post.sort_values(by=['radius'])
    phi_post["Residual"] = phi_post["MassArea"].values/phi_obs["MassArea"].values
    
    posterior_phis += [phi_post]

fig, axs = plt.subplots(3,3, figsize=(
            10, 10), facecolor='w', edgecolor='k')
axs = axs.ravel()
for j, phi_step in enumerate(prior_phi_steps):
    vis.plot_sample(observation_phis[j], vent=(0,0), log=False,#bounds=bounds, 
                   title="Obs %s"%phi_step["interval"], cbar_label="Mass/Area", ax=axs[j])

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(3,3, figsize=(
            10, 10), facecolor='w', edgecolor='k')
axs = axs.ravel()

for j, phi_step in enumerate(prior_phi_steps):
    vis.plot_sample(posterior_phis[j], vent=(0,0), log=False,#bounds=bounds, 
                   title="Post %s"%phi_step["interval"], cbar_label="Mass/Area", ax=axs[j])
    
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(3,3, figsize=(
            15, 15), facecolor='w', edgecolor='k')
axs = axs.ravel()

for j, phi_step in enumerate(prior_phi_steps):
    vis.plot_residuals(posterior_phis[j], vent=(0,0), plot_type="size",
                   title="Post %s"%phi_step["interval"], ax=axs[j])
    
plt.tight_layout()
plt.show()

# %%
fig, axs = plt.subplots(3,3, figsize=(
            15, 15), facecolor='w', edgecolor='k')
axs = axs.ravel()

for j, phi_step in enumerate(prior_phi_steps):
    min_mass = min((min(observation_phis[j]["MassArea"].values), min(posterior_phis[j]["MassArea"].values)))
    max_mass = max((max(observation_phis[j]["MassArea"].values), max(posterior_phis[j]["MassArea"].values)))
    axs[j].scatter(np.sqrt(observation_phis[j]["MassArea"].values), np.sqrt(posterior_phis[j]["MassArea"].values))
    axs[j].plot([-100,100],[-100,100], "k:")
    axs[j].set_xlabel("SQRT(Observed)")
    axs[j].set_ylabel("SQRT(Calculated)")
    axs[j].set_aspect('equal', 'box')
    axs[j].set_title(phi_step["interval"])
    axs[j].set_xlim([np.sqrt(min_mass)-1, np.sqrt(max_mass)+1])
    axs[j].set_ylim([np.sqrt(min_mass)-1, np.sqrt(max_mass)+1])
    
plt.tight_layout()
plt.show()

# %%
fig, axs = plt.subplots(3,3, figsize=(
            15, 12), facecolor='w', edgecolor='k')
axs = axs.ravel()

for j, phi_step in enumerate(prior_phi_steps):
    axs[j].plot(observation_phis[j]["radius"].values, observation_phis[j]["MassArea"].values, 'C2o-', label="Observation Data")
    axs[j].plot(posterior_phis[j]["radius"].values, posterior_phis[j]["MassArea"].values, 'C1o', label="Posterior Simulation")
    axs[j].legend()
    axs[j].set_title(phi_step["interval"])
    axs[j].set_xlabel("Distance from vent (m)")
    axs[j].set_ylabel("Mass/Area")
    axs[j].set_xlim(1000, 13000)
    
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(3,3, figsize=(
            15, 12), facecolor='w', edgecolor='k')
axs = axs.ravel()

for j, phi_step in enumerate(prior_phi_steps):
    axs[j].plot(posterior_phis[j]["radius"].values, posterior_phis[j]["Residual"].values*100, 'C1o', label="Post/Obs")
    axs[j].axhline(100, linestyle="--", lw=1, c="gray")  
    axs[j].legend()
    axs[j].set_xlabel("Distance from vent (m)")
    axs[j].set_title(phi_step["interval"])
    axs[j].set_ylabel("Sim as % of Real")
    axs[j].set_xlim(1000, 13000)
    
plt.tight_layout()
plt.show()

# %%
def misfit_sse(observed, expected):
    return ((observed - expected)**2)/expected

contributions = np.zeros((len(prior_phi_steps), len(observation_phis[0])))
RMSE_contributions = np.zeros((len(prior_phi_steps), len(observation_phis[0])))
print(contributions.shape)
for j, phi_step in enumerate(prior_phi_steps):
    observation_phis[j]["MassArea"].values
    for i in range(len(posterior_phis[j])):
        observed = observation_phis[j]["MassArea"].values[i]
        expected = posterior_phis[j]["MassArea"].values[i]
        misfit = ((observed - expected)**2)/expected
        RMSE = ((observed - expected)**2)
        contributions[j,i] = misfit
        RMSE_contributions[j,i] = RMSE
    
RMSE_point_contributions = np.sqrt(np.sum(RMSE_contributions, 0)/38)
RMSE_phi_contributions = np.sqrt(np.sum(RMSE_contributions, 1)/9)

point_contributions = np.sum(contributions, 0)
phi_contributions = np.sum(contributions, 1)

total_misfit = np.sum(np.sum(contributions))

total_RMSE = np.sqrt(np.sum(np.sum(RMSE_contributions))/38)

total_NRMSE = total_RMSE/(np.max(obs_df["MassArea"].values) - np.min(obs_df["MassArea"].values))
display.display(phi_contributions)
print(total_misfit)
print(total_RMSE)
print(total_NRMSE)

# %%
fig, axs = plt.subplots(3,3, figsize=(
            15, 12), facecolor='w', edgecolor='k')
axs = axs.ravel()
for j, phi_step in enumerate(prior_phi_steps):
    axs[j].plot(observation_phis[j]["radius"], contributions[j,:], "C0o-")
    axs[j].set_xlabel("Distance from Vent")
    axs[j].set_ylabel("SSE Contribution")
    axs[j].set_title(phi_step["interval"])
    axs[j].set_xlim(1000, 13000)

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(3,3, figsize=(
            15, 15), facecolor='w', edgecolor='k')
axs = axs.ravel()
for j, phi_step in enumerate(prior_phi_steps):
    observation_phis[j]["Contributions"] = contributions[j,:]
    vis.plot_sample(observation_phis[j], vent=(0,0), log=False, values="Contributions",
                title="SSE Contributions", cbar_label="SSE Contributions", ax = axs[j], cmap="hot")
    axs[j].set_xlim([-3000, 6500])
    axs[j].set_ylim([-500, 12000])
    axs[j].set_title(phi_step["interval"])

plt.tight_layout()
plt.show()

# %%
fig, axs = plt.subplots(3,3, figsize=(
            15, 12), facecolor='w', edgecolor='k')
axs = axs.ravel()
for j, phi_step in enumerate(prior_phi_steps):
    axs[j].plot(observation_phis[j]["radius"], np.sqrt(RMSE_contributions[j,:]/len(observation_phis[j])), "C0o-")
    axs[j].set_xlabel("Distance from Vent")
    axs[j].set_ylabel("RMSE* Contribution")
    axs[j].set_title(phi_step["interval"])
    axs[j].set_xlim(1000, 13000)

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(3,3, figsize=(
            15, 15), facecolor='w', edgecolor='k')
axs = axs.ravel()
for j, phi_step in enumerate(prior_phi_steps):
    observation_phis[j]["Contributions"] = np.sqrt(RMSE_contributions[j,:]/len(observation_phis[j]))
    vis.plot_sample(observation_phis[j], vent=(0,0), log=False, values="Contributions", cbar_label="RMSE* Contributions", ax = axs[j], cmap="hot")
    axs[j].set_xlim([-3000, 6500])
    axs[j].set_ylim([-500, 12000])
    axs[j].set_title(phi_step["interval"])A

plt.tight_layout()
plt.show()

# %%
row_labels = [phi["interval"] for phi in prior_phi_steps]

num_phis = len(theo_phi_steps)

cont_df = pd.DataFrame(data=contributions.T, columns=row_labels, index = obs_df.index)

min_phis = np.array([min(observation_phis[i]["MassArea"].values) for i in range(num_phis)])
print(min_phis)
max_phis = np.array([max(observation_phis[i]["MassArea"].values) for i in range(num_phis)])
print(max_phis)

#Total sum per column: 
cont_df.loc['Phi Total (RMSE)',:9]= np.sqrt(cont_df.sum(axis=0)/num_phis)
cont_df.loc['Phi Total (NRMSE)',:9] = np.sqrt(cont_df.sum(axis=0)/num_phis)/(max_phis-min_phis)

#Total sum per row: 
cont_df.loc[:38,'Point Total (RMSE*)'] = np.sqrt(cont_df.sum(axis=1)/38)
# cont_df.loc[:,'Point Total (NRMSE)'] = np.sqrt(cont_df.sum(axis=1)/38)

display.display(cont_df)

# %%
print(cont_df.loc['Phi Total (NRMSE)'][:9])

print(sum(cont_df.loc['Phi Total (NRMSE)'].values[:9]))

# %%
disp_cols = ["Easting",
            "Northing",
            "MassArea",
            "radius"]

display.display(observation_phis[-1][disp_cols].head())

display.display(posterior_phis[-1][disp_cols].head())

display.display(obs_df.head())

# %%
