# %%
"""
# Testing the Beta Distribution Parity
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import project.vis_utils as vis
import project.io_utils as io
import project.inversion as inv
import math

import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
pd.options.display.float_format = '{:,g}'.format
plt.style.use(['ggplot'])

# %%

config = io.read_tephra2_config("../tests/test_data/test_config.txt")

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

config["INV_STEPS"] = int(inversion_steps)
config["THEO_MAX_COL"] = closest_H
config["PART_STEPS"] = 9

config["MAX_GRAINSIZE"] = -5
config["MIN_GRAINSIZE"] = 4

# Additional parameter: Constant wind speed
config["WIND_SPEED"] = 10

# To ensure monotonicity:
# config["DIFFUSION_COEFFICIENT"] = 1.7*config["FALL_TIME_THRESHOLD"]

print("INPUT PARAMETERS:")
io.print_table(config)
io.print_table(globs)

# %%


def plume_pdf2(x_norm, step, alpha, beta, sum_prob):

    print(x_norm, step, alpha, beta,
          sum_prob)
    i = 0
    x = x_norm
    a1 = alpha - 1.0
    b1 = beta - 1.0
    probability = 0.0
    if not sum_prob:
        for i in range(config["COL_STEPS"]):
            x += step
            if (x <= 0):
                # step is the small slice of the column as a fraction of the
                # whole
                x1 = x + 0.001
                x2 = 1.0 - x1
                prob = (x1 ** a1) * (x2 ** b1)
            elif (x >= 1):
                x1 = x - 0.001
                x2 = 1.0 - x1
                prob = (x1 ** a1) * (x2 ** b1)
            else:
                x1 = 1.0 - x
                prob = (x ** a1) * (x1 ** b1)
            print("[sum_prob=0][%d] x=%g step=%g prob=%g" % (i, x, step, prob))
            probability += prob

    # Just calculate the probability for one column step
    else:
        x1 = 1.0 - x
        probability = (x ** a1) * (x1 ** b1)
    if (np.isnan(probability)):
        probability = 0
    return probability

# %%


z = np.linspace(config["VENT_ELEVATION"] + layer_thickness,
                config["THEO_MAX_COL"], config["INV_STEPS"])
q_mass = inv.beta_plume(config["ALPHA"], config["BETA"],
                        config["PLUME_HEIGHT"],
                        config["ERUPTION_MASS"],
                        z,
                        config["VENT_ELEVATION"],
                        config["THEO_MAX_COL"])
plt.plot(q_mass, 'ro')
print(sum(q_mass))
release_prob = np.zeros(len(z))

ht_section_width = config["PLUME_HEIGHT"] - config["VENT_ELEVATION"]
ht_step_width = ht_section_width / config["COL_STEPS"]
step_norm = ht_step_width / ht_section_width
x_norm = 0.0
total_P_col = 0.0
print(x_norm, step_norm, config["ALPHA"], config["BETA"],
      total_P_col)
cum_prob_col = plume_pdf2(x_norm, step_norm, config["ALPHA"], config["BETA"],
                          total_P_col)

total_P_col = cum_prob_col
for i in range(config["COL_STEPS"]):
    x_norm += step_norm
    if x_norm <= 0:
        release_prob[i] = plume_pdf2(x_norm + 0.001,
                                     step_norm,
                                     config["ALPHA"],
                                     config["BETA"],
                                     total_P_col)
    elif x_norm >= 1:
        release_prob[i] = plume_pdf2(x_norm - 0.001,
                                     step_norm,
                                     config["ALPHA"],
                                     config["BETA"],
                                     total_P_col)
    else:
        release_prob[i] = plume_pdf2(x_norm,
                                     step_norm,
                                     config["ALPHA"],
                                     config["BETA"],
                                     total_P_col)
print(sum(release_prob))

sus_mass = config["ERUPTION_MASS"] * release_prob / total_P_col
plt.plot(release_prob, 'bo')
plt.plot(sus_mass, 'g*')
print(sum(sus_mass))
print(sum(q_mass))
plt.show()
# %%

# %%


