#  %%
"""
# Colima Inversion Examples


"""

# %%
import matplotlib
import re
import xlrd
from pandas import ExcelFile
from pandas import ExcelWriter
from functools import reduce
from time import process_time
from scipy.optimize import minimize
from scipy.sparse.linalg import lsqr
from scipy.stats import beta, expon
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vis_utils as vis
import io_utils as io
from utils import *
%matplotlib inline
pd.options.display.float_format = '{:,g}'.format
plt.style.use(['ggplot'])

matplotlib.use('module://matplotlib-backend-kitty')

# %%


# %%
filename = "./data/colima/colima_real_data.csv"

raw_df, grid = io.import_colima(filename)

grid.to_csv("data/colima/colima_grid.csv", sep=" ", header=False, index=False)

display(raw_df)

# %%
"""
## Preprossessing Steps
"""

# %%
vis.plot_sample(raw_df, vent=(0, 0), log=False,
                title="Colima 1913 data", cbar_label="Mass/Area")
plt.show()
# plt.savefig("colima/all_points.png", dpi=200, format='png')

obs_df = raw_df.copy()

# This is going to be 100 across. Just needs to be the same as the other dfs.
obs_df["Residual"] = obs_df["MassArea"]/obs_df["MassArea"]

vis.plot_sample(obs_df, vent=(0, 0), log=False,
                title="Preprocessed Data", cbar_label="Mass/Area")
# plt.savefig("colima/All_trans.png", dpi=200, format='png')
plt.show()

# %%
grid.to_csv("data/colima/colima_grid.csv", sep=" ", header=False, index=False)

t2_df, _, _, _ = io.read_tephra2("data/colima/colima_tephra2_sim_data.txt")

t2_df["Residual"] = t2_df["MassArea"].values/obs_df["MassArea"].values

display(t2_df.head())

# %%
t2_const_df, _, _, _ = io.read_tephra2(
    "data/colima/colima_tephra2_const_wind_sim_data.txt")

t2_const_df["Residual"] = t2_const_df["MassArea"]/obs_df["MassArea"]

display(t2_const_df.head())

# %%
config = io.read_tephra2_config("data/colima/colima_config.txt")

globs = {
    "LITHIC_DIAMETER_THRESHOLD": 7.,
    "PUMICE_DIAMETER_THRESHOLD": -1.,
    "AIR_VISCOSITY": 0.000018325,
    "AIR_DENSITY":  1.293,
    "GRAVITY": 9.81,
}

# Update parameters
config["COL_STEPS"] = 20
config["PART_STEPS"] = 9

config["MAX_GRAINSIZE"] = -5
config["MIN_GRAINSIZE"] = 4

# Additional parameter: Constant wind speed
config["WIND_SPEED"] = 10

print("INPUT PARAMETERS:")
display(config)
display(globs)

# %%
phi_steps = get_phi_steps(config["MIN_GRAINSIZE"], config["MAX_GRAINSIZE"],
                          config["PART_STEPS"],
                          config["MEDIAN_GRAINSIZE"], config["STD_GRAINSIZE"],
                          globs["LITHIC_DIAMETER_THRESHOLD"],
                          globs["PUMICE_DIAMETER_THRESHOLD"],
                          config["LITHIC_DENSITY"], config["PUMICE_DENSITY"])
step_width = (config["MAX_GRAINSIZE"] -
              config["MIN_GRAINSIZE"])/config["PART_STEPS"]

display(pd.DataFrame(phi_steps))


probs = [phi["probability"] for phi in phi_steps]
x = [phi["centroid"] for phi in phi_steps]
labels = [phi["interval"] for phi in phi_steps]
fig, ax = plt.subplots()
ax.bar(x, probs, width=1, align="center")
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.xlabel("Phi Interval")
plt.show()

# %%
t = process_time()
t_tot = process_time()
df_list = []

q_dist = beta(config["ALPHA"], config["BETA"])

grid = obs_df[["Easting", "Northing"]]
wind_angle = np.radians(55)

u = config["WIND_SPEED"]*np.cos(wind_angle)
v = config["WIND_SPEED"]*np.sin(wind_angle)
for phi_step in phi_steps:
    mass_in_phi = config["ERUPTION_MASS"] * phi_step["probability"]
    input_table, gsm_df, sig, vv, tft = gaussian_stack_single_phi(
        grid, 20, config["VENT_ELEVATION"],
        config["PLUME_HEIGHT"],
        (config["ALPHA"], config["BETA"]),
        mass_in_phi, (u, v),
        phi_step["lower"], phi_step["density"], 2500,
        config["DIFFUSION_COEFFICIENT"],
        config["EDDY_CONST"],
        config["FALL_TIME_THRESHOLD"])
    df_list.append(gsm_df.rename(columns={"MassArea": phi_step["interval"]}))


elapsed_time = process_time() - t
print("Forward Sim time: %.5f seconds" % elapsed_time)

t = process_time()
df_merge = reduce(lambda x, y: pd.merge(
    x, y, on=['Northing', 'Easting']), df_list)
elapsed_time = process_time() - t
print("Dataframe Merge time: %.5f seconds" % elapsed_time)

t = process_time()
df_merge["MassArea"] = np.sum(df_merge[labels], 1)
elapsed_time = process_time() - t
print("Tot M/A calc time: %.5f seconds" % elapsed_time)

t = process_time()
for label in labels:
    # This operation is much faster
    df_merge[label] = df_merge.apply(lambda row: (
        row[label]/row["MassArea"])*100, axis=1)

elapsed_time = process_time() - t
print("Phi Wt perc calc time: %.5f seconds" % elapsed_time)

total_time = process_time() - t_tot
print("Total time: %.5f seconds" % total_time)

forward_df = df_merge

# forward_df["MassArea"] = forward_df["MassArea"]*(6991/9469)

forward_df["radius"] = np.sqrt(
    forward_df["Easting"]**2 + forward_df["Northing"]**2)
forward_df = forward_df.sort_values(by=['radius'])
forward_df["Residual"] = forward_df["MassArea"].values / \
    obs_df["MassArea"].values
display(forward_df.head())

# %%

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

vis.plot_sample(obs_df, vent=(0, 0), log=True, bounds=(50, 1500),
                title="Colima Observations", cbar_label="Mass/Area", ax=ax)
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(1, 3, figsize=(
    18, 6), facecolor='w', edgecolor='k')
axs = axs.ravel()

vis.plot_sample(forward_df, vent=(0, 0), log=True, bounds=(50, 1500),
                show_cbar=False, title="Forward Simulation (Const. Wind)",
                cbar_label="Mass/Area", ax=axs[0])
vis.plot_sample(t2_const_df, vent=(0, 0), log=True, bounds=(50, 1500),
                show_cbar=False,
                title="Tephra2 Simulation (Const. Wind)",
                cbar_label="Mass/Area", ax=axs[1])
vis.plot_sample(t2_df, vent=(0, 0), log=True, bounds=(50, 1500),
                title="Tephra2 Simulation (NOAA Wind)", cbar_label="Mass/Area",
                ax=axs[2])
plt.show()


# %%
fig, axs = plt.subplots(1, 3, figsize=(
    15, 6), facecolor='w', edgecolor='k')
axs = axs.ravel()


vis.plot_residuals(forward_df, vent=(0, 0), values="Residual",
                   title="Forward Simulation (Const. Wind)",
                   show_legend=False,
                   plot_type="size", ax=axs[0])
vis.plot_residuals(t2_const_df, vent=(0, 0), values="Residual",
                   title="Tephra2 Simulation (Const. Wind)", show_legend=False,
                   plot_type="size", ax=axs[1])
vis.plot_residuals(t2_df, vent=(0, 0), values="Residual",
                   title="Tephra2 Simulation (NOAA Wind)",  show_legend=True,
                   plot_type="size", ax=axs[2])


plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(
    18, 6), facecolor='w', edgecolor='k')
axs = axs.ravel()


vis.plot_residuals(forward_df, vent=(0, 0), values="Residual",
                   title="Forward Simulation (Const. Wind)",
                   plot_type="cmap", ax=axs[0], show_cbar=False)
vis.plot_residuals(t2_const_df, vent=(0, 0), values="Residual",
                   title="Tephra2 Simulation (Const. Wind)",
                   plot_type="cmap", ax=axs[1], show_cbar=False)
vis.plot_residuals(t2_df, vent=(0, 0), values="Residual",
                   title="Tephra2 Simulation (NOAA Wind)",
                   plot_type="cmap", ax=axs[2], show_cbar=True)


# plt.tight_layout()
plt.show()

# %%
fig, axs = plt.subplots(2, 1, figsize=(
    8, 10), facecolor='w', edgecolor='k')
axs = axs.ravel()

axs[0].plot(obs_df["radius"].values, obs_df["MassArea"].values,
            'C0o-', label="Observations")
# plt.plot(rotdf["radius"].values, rotdf["MassArea"].values, 'C1o-')
axs[0].plot(forward_df["radius"].values, forward_df["MassArea"].values,
            'C1o', label="Forward (Const. Wind)")
axs[0].plot(t2_const_df["radius"].values,
            t2_const_df["MassArea"].values, 'C2o',
            label="Tephra2 (Const. Wind)")
axs[0].plot(t2_df["radius"].values, t2_df["MassArea"].values,
            'C3o', label="Tephra2 (NOAA Wind)")
axs[0].legend()
axs[0].set_xlabel("Distance from vent (m)")
axs[0].set_ylabel("Mass/Area")

axs[1].plot(forward_df["radius"].values, forward_df["Residual"].values *
            100, 'C1o', label="Forward (Const. Wind)")
axs[1].plot(t2_const_df["radius"].values, t2_const_df["Residual"].values *
            100, 'C2o', label="Tephra2 (Const. Wind)")
axs[1].plot(t2_df["radius"].values, t2_df["Residual"].values *
            100, 'C3o', label="Tephra2 (NOAA Wind)")
axs[1].axhline(100, linestyle="--", lw=1, c="gray")
axs[1].legend()
axs[1].set_xlabel("Distance from vent (m)")
axs[1].set_ylabel("Sim as % of Real")
plt.show()

# %%
priors_vals = {
    "a": 1.1,
    "b": 1.1,
    "h1": config["PLUME_HEIGHT"],
    "u": u,
    "v": v,
    "D": config["DIFFUSION_COEFFICIENT"],
    "ftt": config["FALL_TIME_THRESHOLD"],
    "M": config["ERUPTION_MASS"]
}

invert_params = {
    "a": True,
    "b": True,
    "h1": False,
    "u": False,
    "v": False,
    "D": False,
    "ftt": False,
    "M": False
}

H = 30000

wind_angle = np.radians(55.5)

u = config["WIND_SPEED"]*np.cos(wind_angle)
v = config["WIND_SPEED"]*np.sin(wind_angle)
names = ["Const. Wind Simulation", "T2 Const. Wind Simulation",
         "T2 NOAA Wind Simulation", "Observation Data"]
data_sets = [forward_df, t2_const_df, t2_df, obs_df]
inverted_masses_list = []
params_list = []
for name, df in zip(names, data_sets):
    print("========%s========" % name)
    out = gaussian_stack_plume_inversion(
        df, len(df), 20,
        config["VENT_ELEVATION"], H, 2500,
        phi_steps, config["EDDY_CONST"],
        invert_params=invert_params,
        priors=priors_vals,
        column_cap=H)
    inversion_table, params, sol, sse, trace, _, sse_trace = out

    inv_mass = inversion_table["Suspended Mass"].values
    inverted_masses_list += [inv_mass]
    params_list += [params]
    display(inversion_table)
    trace = np.array(trace)
    fig, axs = plt.subplots(2, 2, figsize=(
        11, 8), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    axs[0].plot(trace[:, 0], linewidth=.8)
    axs[0].set_title("a")

    axs[1].plot(trace[:, 1], linewidth=.8)
    axs[1].set_title("b")

    axs[2].plot(trace[:, 2], linewidth=.8)
    axs[2].set_title("h1")

    axs[3].plot(np.log(sse_trace), linewidth=.8)
    axs[3].set_title("log(sse)")

    plt.tight_layout()
    plt.show()


# %%
fig, ax1 = plt.subplots(1, 1, figsize=(
    8, 6), facecolor='w', edgecolor='k')

q_mass = mass_dist_in_plume(config["ALPHA"], config["BETA"],
                            config["VENT_ELEVATION"],
                            config["PLUME_HEIGHT"],
                            inversion_table["Height"],
                            config["ERUPTION_MASS"])

ax1.plot(q_mass, inversion_table["Height"], label="Simulation Parameters")
for name, mass in zip(names, inverted_masses_list):
    ax1.plot(mass,
             inversion_table["Height"],
             '--', label=name)
ax1.legend()
# ax1.set_title("Mass in Column as inverted from various datasets")
ax1.set_ylabel("Height")
ax1.set_xlabel("Mass/Area")

plt.tight_layout()
plt.show()


# %%
for name, params, mass, in_data in zip(names, params_list,
                                       inverted_masses_list, data_sets):
    print("========%s========" % name)
    q_dist = beta(params["a"], params["b"])

    grid = obs_df[["Easting", "Northing"]]

    post_df = gaussian_stack_forward(
        grid, int(config["COL_STEPS"]), config["VENT_ELEVATION"],
        params["h1"], 2500, phi_steps, (params["a"],
                                        params["b"]), config["ERUPTION_MASS"],
        (u, v), config["DIFFUSION_COEFFICIENT"], config["EDDY_CONST"],
        config["FALL_TIME_THRESHOLD"]
    )

    post_df["radius"] = np.sqrt(post_df["Easting"]**2 + post_df["Northing"]**2)
    post_df = post_df.sort_values(by=['radius'])
    post_df["Residual"] = post_df["MassArea"].values/obs_df["MassArea"].values
    post_df["Change"] = post_df["MassArea"].values/in_data["MassArea"].values

    fig, axs = plt.subplots(5, 1, figsize=(
        5, 20), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    bounds = (25, 350)
    vis.plot_sample(in_data, vent=(0, 0), log=False, bounds=bounds,
                    title="Input Dataset (%s)" % name, cbar_label="Mass/Area",
                    ax=axs[0])
    vis.plot_sample(post_df, vent=(0, 0), log=False, bounds=bounds,
                    title="Posterior Simulation", cbar_label="Mass/Area",
                    ax=axs[1])
    vis.plot_residuals(post_df, vent=(0, 0), values="Change", plot_type="size",
                       title="Change from Input Dataset", ax=axs[2])
    vis.plot_residuals(in_data, vent=(0, 0), values="Residual",
                       plot_type="size",
                       title="Residual wrt. Obs. (Before Inv.)", ax=axs[3])
    vis.plot_residuals(post_df, vent=(0, 0), values="Residual",
                       plot_type="size",
                       title="Residual wrt. Obs. (After Inv.)", ax=axs[4])

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(
        8, 9), facecolor='w', edgecolor='k')
    axs = axs.ravel()

    axs[0].plot(obs_df["radius"].values, obs_df["MassArea"].values,
                'C2o-', label="Observation Data")
    axs[0].plot(in_data["radius"].values, in_data["MassArea"].values,
                'C0o', label="Input Data (%s)" % name)
    axs[0].plot(post_df["radius"].values, post_df["MassArea"].values,
                'C1o', label="Posterior Simulation")
    axs[0].legend()
    axs[0].set_xlabel("Distance from vent (m)")
    axs[0].set_ylabel("Mass/Area")

    axs[1].plot(in_data["radius"].values,
                in_data["Residual"].values*100, 'C0o', label="Input/Obs")
    axs[1].plot(post_df["radius"].values,
                post_df["Residual"].values*100, 'C1o', label="Post/Obs")
    axs[1].axhline(100, linestyle="--", lw=1, c="gray")
    axs[1].legend()
    axs[1].set_xlabel("Distance from vent (m)")
    axs[1].set_ylabel("Sim as % of Real")
    plt.show()


# %%


# %%
priors_vals = {
    "a": 2,
    "b": 5,
    "h1": 18000,
    "u": 5,
    "v": 6,
    "D": 4000,
    "ftt": 6000,
    "M": config["ERUPTION_MASS"]
}

invert_params = {
    "a": True,
    "b": True,
    "h1": True,
    "u": True,
    "v": True,
    "D": True,
    "ftt": True,
    "M": False
}

H = 30000

t_tot = process_time()
single_run_time = 0
wind_angle = np.radians(55.5)

names = ["Const. Wind Simulation",
         "T2 Const. Wind Simulation",
         "T2 NOAA Wind Simulation",
         "Observation Data"]
data_sets = [forward_df, t2_const_df, t2_df, obs_df]

inverted_masses_list = []
params_list = []
for name, df in zip(names, data_sets):
    t = process_time()

    print("========%s========" % name)
    out = gaussian_stack_inversion(
        df, len(df), 20, config["VENT_ELEVATION"],
        H, 2500, phi_steps,
        invert_params=invert_params,
        priors=priors_vals,
        column_cap=H)
    inversion_table, params, sol, sse, plume_trace, param_trace, sse_trace = out
    inv_mass = inversion_table["Suspended Mass"].values
    inverted_masses_list += [inv_mass]
    params_list += [params]
    display(inversion_table)
    plume_trace = np.array(plume_trace)
    param_trace = np.array(param_trace)
    fig, axs = plt.subplots(3, 3, figsize=(
        11, 9), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    axs[0].plot(plume_trace[:, 0], linewidth=.8)
    axs[0].set_title("a")

    axs[1].plot(plume_trace[:, 1], linewidth=.8)
    axs[1].set_title("b")

    axs[2].plot(plume_trace[:, 2], linewidth=.8)
    axs[2].set_title("h1")

    axs[3].plot(param_trace[:, 0], linewidth=.8)
    axs[3].set_title("u")

    axs[4].plot(param_trace[:, 1], linewidth=.8)
    axs[4].set_title("v")

    axs[5].plot(param_trace[:, 2], linewidth=.8)
    axs[5].set_title("Diffusion Coefficient")

    axs[6].plot(param_trace[:, 3], linewidth=.8)
    axs[6].set_title("Fall Time Threshold")

    axs[7].plot(param_trace[:, 4], linewidth=.8)
    axs[7].set_title("Total Mass")

    axs[8].plot(sse_trace, linewidth=.8)
    axs[8].set_title("SSE")

    plt.tight_layout()
    # plt.savefig("colima/real_trace.png", dpi=200, format='png')
    plt.show()

    run_time = process_time() - t
    print("%s Run Time: %.5f minutes\n\n" % (name, run_time/60))


total_run_time = process_time() - t_tot

print("Total Run Time: %.5f minutes" % (total_run_time/60))

# %%
fig, ax1 = plt.subplots(1, 1, figsize=(
    8, 6), facecolor='w', edgecolor='k')

q_dist = beta(config["ALPHA"], config["BETA"])

q_mass = mass_dist_in_plume(config["ALPHA"], config["BETA"],
                            config["VENT_ELEVATION"],
                            config["PLUME_HEIGHT"],
                            inversion_table["Height"],
                            config["ERUPTION_MASS"])

ax1.plot(q_mass, inversion_table["Height"], label="Simulation Parameters")
for name, mass in zip(names, inverted_masses_list):
    ax1.plot(mass,
             inversion_table["Height"],
             '--', label=name)
ax1.legend()
# ax1.set_title("Mass in Column as inverted from various datasets")
ax1.set_ylabel("Height")
ax1.set_xlabel("Mass/Area")

plt.tight_layout()
plt.show()
params_df = pd.DataFrame(params_list)
params_df["Dataset"] = names
params_df = params_df[["Dataset", "a", "b", "h1", "u", "v", "D", "ftt", "M"]]
display(params_df)

# %%
for name, params, mass, in_data in zip(names, params_list,
                                       inverted_masses_list, data_sets):
    print("========%s========" % name)
    q_dist = beta(params["a"], params["b"])

    grid = obs_df[["Easting", "Northing"]]

    post_df = gaussian_stack_forward(
        grid, int(config["COL_STEPS"]), config["VENT_ELEVATION"],
        params["h1"], 2500, phi_steps, (params["a"],
                                        params["b"]), config["ERUPTION_MASS"],
        (u, v), config["DIFFUSION_COEFFICIENT"], config["EDDY_CONST"],
        config["FALL_TIME_THRESHOLD"]
    )

    post_df["radius"] = np.sqrt(post_df["Easting"]**2 + post_df["Northing"]**2)
    post_df = post_df.sort_values(by=['radius'])
    post_df["Residual"] = post_df["MassArea"].values/obs_df["MassArea"].values
    post_df["Change"] = post_df["MassArea"].values/in_data["MassArea"].values

    fig, axs = plt.subplots(5, 1, figsize=(
        6, 25), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    bounds = (50, 1500)
    vis.plot_sample(in_data, vent=(0, 0), log=True, bounds=bounds,
                    title="Input Dataset (%s)" % name, cbar_label="Mass/Area",
                    ax=axs[0])
    vis.plot_sample(post_df, vent=(0, 0), log=True, bounds=bounds,
                    title="Posterior Simulation", cbar_label="Mass/Area",
                    ax=axs[1])
    vis.plot_residuals(post_df, vent=(0, 0), values="Change", plot_type="size",
                       title="Change from Input Dataset", ax=axs[2])
    vis.plot_residuals(in_data, vent=(0, 0), values="Residual",
                       plot_type="size",
                       title="Residual wrt. Obs. (Before Inv.)", ax=axs[3])
    vis.plot_residuals(post_df, vent=(0, 0), values="Residual",
                       plot_type="size",
                       title="Residual wrt. Obs. (After Inv.)", ax=axs[4])

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(
        8, 9), facecolor='w', edgecolor='k')
    axs = axs.ravel()

    axs[0].plot(obs_df["radius"].values, obs_df["MassArea"].values,
                'C2o-', label="Observation Data")
    axs[0].plot(in_data["radius"].values, in_data["MassArea"].values,
                'C0o', label="Input Data (%s)" % name)
    axs[0].plot(post_df["radius"].values, post_df["MassArea"].values,
                'C1o', label="Posterior Simulation")
    axs[0].legend()
    axs[0].set_xlabel("Distance from vent (m)")
    axs[0].set_ylabel("Mass/Area")

    axs[1].plot(in_data["radius"].values,
                in_data["Residual"].values*100, 'C0o', label="Input/Obs")
    axs[1].plot(post_df["radius"].values,
                post_df["Residual"].values*100, 'C1o', label="Post/Obs")
    axs[1].axhline(100, linestyle="--", lw=1, c="gray")
    axs[1].legend()
    axs[1].set_xlabel("Distance from vent (m)")
    axs[1].set_ylabel("Sim as % of Real")
    plt.show()


# %%


# %%
