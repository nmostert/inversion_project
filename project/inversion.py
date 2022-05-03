import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta
from scipy.special import betainc
from time import process_time
import matplotlib.pyplot as plt
import random
import sys
sys.path.append("..")
import project.io_utils as io
from tabulate import tabulate

# Suppress warning from scipy.optimize.minimize. This function is purposely
# terminated early as part of our custom optimisation method.
import warnings
warnings.filterwarnings("ignore",
                        message="Warning: Maximum number of iterations "
                                "has been exceeded.")
np.seterr(invalid='ignore')
np.seterr(divide='ignore')

# Are globals really that bad? If they are I have a problem.
PARAM_TRACE = []
MISFIT_TRACE = []
MULTI_MISFIT_TRACE = []
FINAL_MISFIT_TRACE = []
TGSD_TRACE = []
MASS_TRACE = []
TOTAL_ITER = 0

# Abortion globals
ABORT_FLAG = False
ABORT_MESSAGE = None
ABORT_ITER = 0
ABORT_MISFIT = None

# Possibly unused?
BOUND_FACTOR = False
SPIKE_TERM = False

LITHIC_DIAMETER_THRESHOLD = 7.
PUMICE_DIAMETER_THRESHOLD = -1.
AIR_VISCOSITY = 0.000018325
AIR_DENSITY = 1.293
GRAVITY = 9.81


def pdf_grainsize(part_mean, part_sigma, part_max_grainsize, part_step_width):
    """Calculates probability of single phi class in a normal TGSD.

    Function taken directly from Tephra2.

    Parameters
    ----------
    part_mean : float
        The mean of the normal grainsize pdf in phi.
    part_sigma : float
        The standard deviation of the normal grainsize pdf in phi.
    part_max_grainsize : float
        The top (smallest phi) of the grainsize class/bin.
    part_step_width : float
        The bin width in phi.

    Returns
    -------
    func_rho : float
        The probability value of that grain size class/pdf bin.
    """
    temp1 = 1.0 / (2.506628 * part_sigma)
    temp2 = np.exp(-(part_max_grainsize - part_mean)**2
                   / (2*part_sigma*part_sigma))
    func_rho = temp1 * temp2 * part_step_width
    return func_rho


def get_phi_steps(
    min_grainsize, max_grainsize, part_steps, median_grainsize, std_grainsize,
    lithic_diameter_threshold, pumice_diameter_threshold, lithic_density,
    pumice_density
):
    """Returns the internally used grain size configuration based on
    the normal TGSD parameter inputs.

    Particles with grain sizes in between the lithic and pumice thresholds are
    given an interpolated density between the two values.

    Parameters
    ----------
    min_grainsize : float
        Smallest grainsize (largest phi)
    max_grainsize : float
        Largest grainsize (smallest phi)
    part_steps : float
        TGSD bin width in phi.
    median_grainsize : float
        Median of normal grainsize pdf in phi.
    std_grainsize : float
        Standard deviation of normal grainsize in phi.
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

        prob = pdf_grainsize(
            median_grainsize,
            std_grainsize, y,
            part_step_width)

        phi_class = {
            "lower": y,
            "upper": y + part_step_width,
            "interval": "[%.3g,%.3g)" % (y, y + part_step_width),
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


def get_tgsd(df, phi_steps):
    """Calculate naive estimation of TEST from a given deposit.

    # TODO: Add support for more sophisticated methods of TEST calculation #
    # TODO: Take in interval strings directly. No need to pass phi_steps #

    Parameters
    ----------
    df : Dataframe
        Pandas data frame of deposit dataset.
    phi_steps : list(dict)
        List of dicts calculated by get_phi_steps function, used only to
        provide interval notation labels.

    Returns
    -------
    test : List containing pdf probabilities calculated in a naive sense.
    """
    tgsd = []
    for phi in phi_steps:
        tgsd += [sum((df[phi["interval"]]/100)*df["MassArea"])/len(df)]
    tgsd = np.array(tgsd)/sum(tgsd)
    return tgsd


def sample(df, n, weight="Mass Area", alpha=0.5):
    """Randomly select n data points from deposit data frame.

    # TODO: UNUSED #

    Parameters
    ----------
    df : Dataframe
        Pandas data frame of deposit dataset.
    n : int
        Number of points to select from dataset.
    weight : str
        Data frame column name to use as weight values.
    alpha : float
        Weight scaling factor.
    """
    weights = df[weight].copy()  # Get values to be used as weights
    weights = weights**(alpha)  # Apply scaling factor as w^alpha
    probs = weights/np.sum(weights)  # Normalise to sum up to one
    chosen = np.random.choice(df.index, n, replace=False, p=probs)
    # Randomly choose n points
    return df.loc[chosen]


def d2phi(d):
    """Convert diameter (mm) to phi.

    Parameters
    ----------
    d : float
        particle diameter (mm)
    """
    return - np.log2(d)


def phi2d(phi):
    """Convert phi to diameter (mm)

    Parameters
    ----------
    phi : float
        phi-scale value of particle
    """
    return 2**(-phi)


def column_spread_fine(height):
    """Column spread of particles with fall time above FT parameter.

    Parameters
    ----------
    height : float
        Particle release height (m).
    """
    return (0.2*(height**2))**(2/5)


def column_spread_coarse(height, diffusion_coefficient):
    """Column spread of particles with fall time below FT parameter.

    Parameters
    ----------
    height : float
        Particle release height (m).
    diffusion_coefficient : float
        Diffusion coefficient input parameter K.
    """
    return 0.0032 * (height**2) / diffusion_coefficient


def fall_time(terminal_velocity, release_height):
    """Calculates the fall time of a particle.

    DEPRECATED. Use part_fall_time.

    Parameters
    ----------
    terminal_velocity : float
        Terminal velocity of a particle (m/s).
    release_height : float
        Particle release height (m) above ground level.
    """
    return release_height/terminal_velocity


def func2(x, y, sigma_sqr, x_bar, y_bar):
    """The analytical solution of the mass conservation equation for emission
    from a point source.
    (Eq 5 from Bonadonna et al 2005, doi:10.1029/2003JB002896)

    THIS FUNCTION IS UNUSED. It has been replaced by strat_average, which is
    a direct transcription of the equivalent function in Tephra2.

    Parameters
    ----------
    x : float
        The x coordinate of the accumulation point.
    y : float
        The y coordinate of the accumulation point.
    sigma_sqr : float
        The variance of the Gaussian-like distribution function, calculated
        by the sigma_squared function.
    x_bar : float
        The x coordinate of the center of mass of the suspended mass cloud.
    y_bar : float
        The y coordinate of the centre of mass of the suspended mass cloud.
    """
    return 1/(np.pi*sigma_sqr) * \
        np.exp(-((x - x_bar)**2 + (y - y_bar)**2)/(2*sigma_sqr))


def sigma_squared(
    height, fall_time, diff_coef, spread_coarse, spread_fine, eddy_const,
    fall_time_thres, diff_two=0
):
    """The variance of the Gaussian-like distribution function.

    (Eqs 6 and 8 of Bonadonna et al 2005, doi:10.1029/2003JB002896)

    Parameters
    ----------
    height : float
        Particle release height (m).
    fall_time : float
        Particle fall time (s).
    diff_coef : float
        Diffusion coefficient input parameter, K.
    spread_coarse : float
        Column "spread" for particles with low fall times.
    spread_fine : float
        Column "spread" for particles with high fall times.
    eddy_const : float
        Eddy constant input parameter.
    fall_time_thres : float
        Fall time threshold (s) parameter by which coarse and fine particles
        are separates.
    diff_two : float
        Additional crosswind diffusion parameter (Deprecated)
    """
    if fall_time < fall_time_thres:
        ss = 4*diff_coef*(fall_time + spread_coarse)
    else:
        ss = ((8*eddy_const)/5) * ((fall_time + spread_fine)**(5/2))
    if ss <= 0:
        ss += 1e-9
    return ss + diff_two


def mass_dist_in_plume(a, b, z_min, z_max, z_points, total_mass):
    """ Discretised Beta mass distribution suspended in the plume.
    DEPRECATED: use beta_plume in stead.
    Parameters
    ----------
    a : float
        alpha parameter of Beta distribution.
    b : float
        beta parameter of Beta distribution.
    z_min : float
        Bottom of suspended plume (m).
    z_max : float
        Top of suspended plume.
    z_points : float
        Particle release points in the plume.
    total_mass : float
        Total suspended mass in plume.

    Returns
    -------
    mass_dist :
        List of masses suspended at points corresponding to z_points.
    """
    pdf = beta.pdf(z_points, a=a, b=b, loc=z_min, scale=(z_max-z_min))
    mass_dist = (pdf/sum(pdf))*total_mass
    return mass_dist


def construct_grid_dataframe(deposit, easting, northing):
    """This constructs a dataframe that works with the visualisation library.

    Parameters
    ----------
    deposit : list(float)
        List of mass/area values.
    easting : list(float)
        Easting points (m).
    northing : list(float)
        Northing points (m).
    """
    df = pd.DataFrame(deposit.T, columns=northing, index=easting)
    df.reset_index(inplace=True)
    df = pd.melt(df, id_vars=["index"])
    df = df.rename(columns={
        "index": "Easting",
        "variable": "Northing",
        "value": "MassArea"
    })
    return df


def construct_dataframe(deposit, easting, northing):
    """This constructs a dataframe from simulated values for return.

    # TODO: Is this used in any notebooks? #

    Parameters
    ----------
    deposit : list(float)
        List of mass/area values.
    easting : list(float)
        Easting points (m).
    northing : list(float)
        Northing points (m).
    """
    data = {
        "Northing": northing,
        "Easting": easting,
        "MassArea": deposit
    }

    df = pd.DataFrame(data)
    return df


# def random_sample(n, df, sample_dev, K):
#     # TODO: Possibly unused. Check notebooks.  #
#     transect = df[df["Northing"] == 0].values
#     max_point = np.argmax(transect)
#     samp_x = df["Easting"].values
#     return samp_x, samp_y


def part_fall_time(
    particle_ht, layer, ashdiam, part_density, air_density, gravity,
    air_viscosity
):
    """Calculate particle fall time through an atmospheric layer, as well as
    its terminal velocity. This function is directly transcribed from Tephra2.

    (Eqs 7 and 9 of Bonadonna et al 2005, doi:10.1029/2003JB002896
    as well as Eq 2 of Connor et al 2019,
    https://doi.org/10.1007/978-3-642-25911-1_3)

    Parameters
    ----------
    particle_ht :
        Height of particle above sea level.
    layer :
        Vertical "thickness" of the current layer that the particle is moving
        through.
    ashdiam :
        Diameter of the particle (mm?)
    part_density :
        Density of the particle.
    air_density :
        Density of the air.
    gravity :
        Gravitational constant.
    air_viscosity :
        Viscosity of the air.
    """
    hz = particle_ht  # height of particle above sea level
    particle_fall_time = 0.0

    # rho is the density of air (kg/m^3) at the elevation of the current
    # particle
    rho = air_density * np.exp(-hz/8200.0)

    #  (friction due to the air) :
    #  vtl is terminal velocity (m/s) in laminar regime RE<6
    #  vti is terminal velocity (m/s) in intermediate regime 6<RE<500
    #  vtt is terminal velocity (m/s) in turbulent regime RE>500

    vtl = part_density * gravity * ashdiam * ashdiam / (air_viscosity * 18.0)

    reynolds_number = ashdiam * rho * vtl / air_viscosity
    particle_term_vel = vtl
    temp0 = ashdiam * rho

    if reynolds_number >= 6.0:
        temp1 = 4.0 * gravity * gravity * part_density * part_density\
            / (air_viscosity * 225.0 * rho)
        vti = ashdiam * (temp1 ** (1./3.))
        reynolds_number = temp0 * vti / air_viscosity
        particle_term_vel = vti

    # c...if intermediate RE>500 (turbulent regime),
    # RE is calculated again considering vtt

    if reynolds_number >= 500.0:
        vtt = np.sqrt(3.1 * part_density * gravity * ashdiam / rho)
        reynolds_number = temp0 * vtt / air_viscosity
        particle_term_vel = vtt

    particle_fall_time = layer / particle_term_vel

    return (particle_fall_time, particle_term_vel)


def strat_average(
    average_wind_direction, average_windspeed, xspace, yspace, total_fall_time,
    sigma
):
    """Forms part of the analytical solution of the mass conservation equation
    for emission from a point source.
    (Eq 5 from Bonadonna et al 2005, doi:10.1029/2003JB002896)

    In practice, this function performs a rotational transformation, aligning
    the xprime direction to the average wind direction, and then calculates the
    exponential term of the function.

    Parameters
    ----------
    average_wind_direction :
        Average wind direction from release point to the accumulation plane.
    average_windspeed :
        Average windspeed from release point to the accumulation plane.
    xspace :
        Accumulation grid point x-coordinate.
    yspace :
        Accumulation grid point y-coordinate.
    total_fall_time :
        The total fall time of the particle from release point to accumulation
        plane.
    sigma :
        This is actually the sigma_sqr parameter that serves as the variance of
        the gaussian-like plume function.
    """
    temp0 = np.cos(average_wind_direction)
    temp1 = np.sin(average_wind_direction)

    xprime = xspace * temp0 + yspace * temp1
    yprime = yspace * temp0 - xspace * temp1

    temp0 = xprime - average_windspeed * total_fall_time
    demon1 = temp0 * temp0 + yprime * yprime
    demon3 = np.exp(-demon1/sigma)
    # where sigma is calculated for the total fall time

    return demon3


def gaussian_stack_single_phi(
    grid, column_steps, z_min, z_max,
    beta_params, tot_mass, wind, phi, particle_density, elevation,
    diffusion_coefficient, eddy_constant, fall_time_threshold
):
    """A forward model simulation for a single phi class.

    This function aims to replicate exactly the function of Tephra2 when using
    a constant wind speed.

    Parameters
    ----------
    grid : Dataframe
        Pandas dataframe with Northing and Easting columns.
    column_steps : list(float)
        Particle release heights (m) in the column.
    z_min : float
        Bottom of the plume (usually the vent height).
    z_max : float
        Top of the plume (Column Height).
    beta_params : (float, float)
        Alpha and beta parameters of the Beta distribution for the suspended
        mass.
    tot_mass : float
        Total mass in phi class.
    wind : (float, float)
        Wind u (Easting) and v (Northing) parameters (m/s).
    phi : float
        Phi grain size of particle.
    particle_density : float
        Density used for the particle phi class.
    elevation : float
        Elevation of the particle accumulation plane.
    diffusion_coefficient : float
        Diffusion coefficient input parameter K.
    eddy_constant : float
        Eddy constant input parameter.
    fall_time_threshold : float
        Threshold that separates coarse and fine particles by grain size.
        (see part_fall_time).
    """

    global AIR_DENSITY, GRAVITY, AIR_VISCOSITY
    u, v = wind
    # Here I convert this to azimuth (clockwise from North)
    wind_angle = np.arctan2(v, u)
    wind_speed = np.sqrt(u**2 + v**2)

    # Release points in column
    layer_thickness = ((z_max-z_min)/column_steps)
    z = np.linspace(z_min + layer_thickness, z_max, column_steps)

    height_above_vent = z - z_min

    # TODO: This should be generalized to take point elevation into
    # account. At the moment it only uses
    distance_below_vent = z_min - elevation

    # Adjustment that accounts for the particle travel below the vent height.
    windspeed_adj = (wind_speed*elevation)/z_min
    u_wind_adj = np.cos(wind_angle)*windspeed_adj
    v_wind_adj = np.sin(wind_angle)*windspeed_adj

    plume_diffusion_fine_particle = [
        column_spread_fine(ht) for ht in height_above_vent]
    plume_diffusion_coarse_particle = [column_spread_coarse(
        ht, diffusion_coefficient) for ht in height_above_vent]

    d = phi2d(phi)/1000

    # Adjustment that accounts for the particle travel below the vent height.
    if distance_below_vent > 0:
        fall_time_adj = part_fall_time(
            z_min, distance_below_vent,
            d, particle_density,
            AIR_DENSITY,
            GRAVITY,
            AIR_VISCOSITY
        )[0]
    else:
        fall_time_adj = 0.

    fall_values = [part_fall_time(
        zk,
        layer_thickness,
        d, particle_density,
        AIR_DENSITY,
        GRAVITY,
        AIR_VISCOSITY
    ) for zk in z]

    # Terminal velocities and fall times.
    vv = [-e[1] for e in fall_values]
    ft = [e[0] for e in fall_values]

    # Mass distribution in the plume
    alpha, beta = beta_params

    # q_mass = mass_dist_in_plume(alpha, beta, z_min, z_max, z, tot_mass)
    q_mass = beta_plume(alpha, beta, z_max, tot_mass, z, z_min, z_max)

    xx = grid["Easting"].values
    yy = grid["Northing"].values
    dep_mass = np.zeros(xx.shape)

    wind_sum_x = 0
    wind_sum_y = 0

    sig = []
    dep_mass_list = []
    wind_speed_list = []
    wind_angle_list = []
    avg_wind_x_list = []
    avg_wind_y_list = []
    wind_sum_x_list = []
    wind_sum_y_list = []
    total_fall_time_list = []
    x_adj_list = []
    y_adj_list = []

    # For each vertical height in the column
    for k, zh in enumerate(z):

        # Adjust the total fall time by the time it
        # takes to fall below the vent height to the ground.

        total_fall_time = sum(ft[:k+1]) + fall_time_adj

        # Here we would put a proper wind field (u[k], v[k]). If we had one.

        x_adj = u_wind_adj*fall_time_adj
        y_adj = v_wind_adj*fall_time_adj

        wind_sum_x += ft[k]*u
        wind_sum_y += ft[k]*v

        average_windspeed_x = (wind_sum_x + x_adj)/total_fall_time
        average_windspeed_y = (wind_sum_y + y_adj)/total_fall_time

        # converting back to degrees
        average_wind_direction = np.arctan2(
            average_windspeed_y, average_windspeed_x)

        average_windspeed = np.sqrt(average_windspeed_x**2 +
                                    average_windspeed_y**2)

        s_sqr = sigma_squared(
            zh, total_fall_time,
            diffusion_coefficient,
            plume_diffusion_coarse_particle[k],
            plume_diffusion_fine_particle[k],
            eddy_constant,
            fall_time_threshold
        )

        total_spread = s_sqr

        dist = strat_average(
            average_wind_direction,
            average_windspeed,
            xx, yy,
            total_fall_time, total_spread)

        calc_mass = (q_mass[k]/(total_spread*np.pi))*dist

        dep_mass += calc_mass
        # logging.info(io.log_table(pd.DataFrame(calc_mass,
        #                                        columns=["ash_fall"]),
        #                           title="Ash fall by point,"
        #                           "height=%g" % (zh)))
        # logging.info(q_mass[k])
        # logging.info(total_spread)

        sig.append(s_sqr)
        dep_mass_list.append(dep_mass.sum())
        total_fall_time_list.append(total_fall_time)
        x_adj_list.append(x_adj)
        y_adj_list.append(y_adj)
        avg_wind_x_list.append(average_windspeed_x)
        avg_wind_y_list.append(average_windspeed_y)
        wind_sum_x_list.append(wind_sum_x)
        wind_sum_y_list.append(wind_sum_y)
        wind_speed_list.append(average_windspeed)
        wind_angle_list.append(average_wind_direction)

    dep_df = construct_dataframe(dep_mass, xx, yy)

    # This table is for logging, debugging and display purposes.
    input_data = np.asarray([
        # z,
        np.asarray(q_mass),
        np.asarray(q_mass)/tot_mass,
        dep_mass_list,
        # [d]*len(z),
        # [particle_density]*len(z),
        # ft,
        total_fall_time_list,
        # [fall_time_adj]*len(z),
        # x_adj_list,
        # y_adj_list,
        vv,
        # plume_diffusion_coarse_particle,
        # plume_diffusion_fine_particle,
        # sig,
        wind_angle_list,
        wind_speed_list,
        # avg_wind_x_list,
        # avg_wind_y_list,
        # wind_sum_x_list,
        # wind_sum_y_list,
        [windspeed_adj]*len(z),
        [u_wind_adj]*len(z),
        [v_wind_adj]*len(z)
    ]).T

    input_table = pd.DataFrame(
        input_data,
        columns=[
            # "Release Height (z)",
            "Suspended Mass (q)",
            "Release Probability ",
            "Deposited Mass",
            # "Ash Diameter",
            # "Particle Density",
            # "Fall Time",
            "Total Fall Time",
            # "Fall Time Adj",
            # "X Adj",
            # "Y Adj",
            "Terminal Velocity",
            # "Col Spead Coarse",
            # "Col Spead Fine",
            # "Diffusion",
            "Avg. Wind Angle",
            "Avg. Wind Speed",
            # "Avg. Wind Speed x",
            # "Avg. Wind Speed y",
            # "Wind Sum x",
            # "Wind Sum y",
            "Windspeed Adj",
            "U wind adj",
            "V wind adj"
        ])

    return input_table, dep_df, sig, vv, ft


def gaussian_stack_forward(
    grid, column_steps, z_min, z_max, elevation, phi_steps,
    beta_params, tot_mass, wind, diffusion_coefficient,
    eddy_constant, fall_time_threshold
):
    """Full multi-phi forward model. This sets up and utilises the
    gaussian_stack_single_phi for each phi class.

    Parameters
    ----------
    grid : Dataframe
        Pandas dataframe with Northing and Easting columns.
    column_steps : list(float)
        Particle release heights (m) in the column.
    z_min : float
        Bottom of the plume (usually the vent height).
    z_max : float
        Top of the plume (Column Height).
    elevation : float
        Elevation of the particle accumulation plane.
    phi_steps : list(dict)
        List of dicts of phi-class properties, as created by get_phi_steps.
    beta_params : (float, float)
        Alpha and beta parameters of the Beta distribution for the suspended
        mass.
    tot_mass : float
        Total erupted mass (kg).
    wind : (float, float)
        Wind u (Easting) and v (Northing) parameters (m/s).
    diffusion_coefficient : float
        Diffusion coefficient input parameter K.
    eddy_constant : float
        Eddy constant input parameter.
    fall_time_threshold : float
        Threshold that separates coarse and fine particles by grain size.
        (see part_fall_time).
    """
    df_list = []
    phi_norm = sum([phi["probability"] for phi in phi_steps])
    for phi_step in phi_steps:
        mass_in_phi = tot_mass * phi_step["probability"]
        input_table, gsm_df, sig, vv, tft = gaussian_stack_single_phi(
            grid, column_steps, z_min, z_max,
            beta_params, mass_in_phi, wind,
            phi_step["lower"], phi_step["density"], elevation,
            diffusion_coefficient, eddy_constant, fall_time_threshold
        )
        logging.info(io.log_table(gsm_df,
                     title=phi_step["interval"] + " dataframe"))
        logging.info(io.log_table(input_table,
                     title=phi_step["interval"] + " dataframe"))
        df_list.append(
            gsm_df.rename(columns={"MassArea": phi_step["interval"]}))
    df_merge = df_list[0]

    logging.info(io.log_table(df_merge,
                 title="dataframe"))
    labels = [phi_step["interval"] for phi_step in phi_steps]
    for df, lab in zip(df_list[1:], labels[1:]):
        df_merge[lab] = df[lab]
    df_merge["MassArea"] = np.sum(df_merge[labels], 1)/phi_norm
    for label in labels:
        df_merge[label] = df_merge.apply(
            lambda row: (row[label]/row["MassArea"])*100, axis=1)
    return df_merge


def beta_function(z, a, b, h0, h1):
    # TODO: Possibly unused, check notebooks. #
    return beta.pdf(z, a, b, h0, h1)

# def beta_plume(a_star, b_star, h0_star, h1_star, tot_mass, z):
#     a, b, h0, h1 = plume_transform(a_star, b_star, h0_star, h1_star)
#     dist = beta.pdf(z, a, b, h0, h1)
#     return (dist/sum(dist))*tot_mass


def beta_plume(a, b, h1, tot_mass, z, z_min, H):
    """This is the beta function used to model the suspended mass
    distribution in the plume.

    Parameters
    ----------
    a : float
        Alpha parameter of Beta distribution.
    b : float
        Beta parameter of Beta distribution.
    h1 : float
        Plume height (m).
    tot_mass : float
        Total erupted mass (kg).
    z : list(float)
        Particle release heights.
    z_min : float
        Bottom of the eruption column (usually the vent height).
    H : float
        Theoretical maximum plume height.
    """
    # Subset of height levels that fall within the plume.
    heights = z[(z >= z_min) & (z <= h1)]

    # TODO: Figure out what I did here. #
    x_k = [(z_k-z_min)/(h1-z_min) for z_k in z]
    x_k[len(heights)-1] = 1 - 0.001

    # TODO: ...and here #
    dist = np.zeros(len(x_k))
    for i in range(len(x_k)):
        dist[i] = (x_k[i] ** (a - 1)) * ((1.0 - x_k[i]) ** (b - 1))

    plume_data = np.asarray([
        z,
        x_k,
        dist,
    ]).T

    plume_table = pd.DataFrame(
        plume_data,
        columns=[
            "Release heights",
            "Norm heights",
            "Release prob"
        ])
    # logging.info(io.log_table(plume_table,
    #                           title="Plume table"))

    plume = np.zeros(len(z))
    # Insert the suspended probabilities in the height levels.
    plume[(z >= z_min) & (z <= h1)] = dist[(z >= z_min) & (z <= h1)]
    # plume[(z >= z_min) & (z <= h1)] = dist

    # Scale the probabilities by the total mass.

    q = (plume/sum(plume))*tot_mass

    column_data = np.asarray([
        z,
        plume,
        q,
    ]).T

    column_table = pd.DataFrame(
        column_data,
        columns=[
            "Column heights",
            "Release prob",
            "Suspended Mass"
        ])

    # logging.info(io.log_table(column_table,
    #                           title="Full column table"))
    return q


def plume_pdf(a, b, h1, tot_mass, z_list, z_min, H):
    """This is the beta function used to model the suspended mass
    distribution in the plume, according to Tephra2.

    Parameters
    ----------
    a : float
        Alpha parameter of Beta distribution.
    b : float
        Beta parameter of Beta distribution.
    h1 : float
        Plume height (m).
    tot_mass : float
        Total erupted mass (kg).
    z_list : list(float)
        Particle release heights.
    z_min : float
        Bottom of the eruption column (usually the vent height).
    H : float
        Theoretical maximum plume height.
    """

    # Subset of height levels that fall within the plume.
    heights = z_list[(z_list >= z_min) & (z_list < h1)]

    # For each height, calculate the normalised height value between 0 and 1.
    x_k = [(z_k-z_min)/(h1-z_min) for z_k in heights]

    # for each height value, (including a 0?), calculate the release
    # probability values at each height level. 
    dist = [betainc(a, b, x_k[i+1]) - betainc(a, b, x_k[i])
            for i in range(len(x_k)-1)] + [0]
    return None

def param_transform(p_star):
    """General parameter transform function."""
    return np.exp(p_star)


def param_inv_transform(p):
    """General parameter inverse transform function."""
    return np.log(p)


def plume_inv_transform(a, b, h1, H):
    """Inverse transformation function for plume parameters.

    Parameters
    ----------
    a : float
        Alpha parameter of Beta distribution.
        Transformation effectively constrains the parameter between to be
        larger than 1.
    b : float
        Beta parameter of Beta distribution.
        Transformation effectively constrains the parameter between to be
        larger than 1.
    h1 : float
        Column height (m)
        Transformation effectively constrains the parameter between to be
        between 0 and H.
    H : float
        Theoretical maximum column height, often set to around 45,000 m.
    """
    a_star = np.log(a - 1)
    b_star = np.log(b - 1)
    h1_star = -np.log(-np.log(h1/H))
    return a_star, b_star, h1_star


def plume_transform(a_star, b_star, h1_star, H):
    """Transformation function for plume parameters.

        See plume_inv_transform for details.
    """
    a = np.exp(a_star) + 1
    b = np.exp(b_star) + 1
    h1 = H*np.exp(-np.exp(-h1_star))
    return a, b, h1


def misfit(a, b, h1, A, z, m, tot_mass, z_min, H, gof="chi-sqr"):
    """Misfit calculation function for a single phi class.

    Calculates the amount of deviation between the observed masses in m, and
    the masses as predicted by a linearised forward model.

    Parameters
    ----------
    a : float
        Alpha parameter of Beta distribution.
    b : float
        Beta parameter of Beta distribution.
    h1 : float
        Column height (m)
    A : 2D array(float)
        The plume matrix as generated by get_plume_matrix.
    z : list(float)
        Particle release heights.
    m : list(float)
        Observation array. The mass/area values in this phi class.
    tot_mass : float
        Total erupted mass (kg) in phi class.
    z_min : float
        Bottom of the eruption column (usually the vent height).
    H : float
        Theoretical maximum column height (m).
    gof : str
        The Goodness-of-fit measure to use in the error calculations.
        Options:
        "chi-sqr" : The chi-squared misfit function as described in Eq ? of
            Connor and Connor (2006) [https://doi.org/10.1144/IAVCEI001.18].
        "RMSE" : The Root-mean-squared-error as described in Eq 16 of
            Connor et. al. (2019) [https://doi.org/10.1007/978-3-642-25911-1_3]
            Note that the calculation here is only the numerator part of the
            RMSE function. The rest is calculated outside of this loop.

    Returns
    -------
    misfit : float
        The error result.
    misfit_contributions : list(float)
        The error contribution of each data point.
    fit : list(float)
        The predicted masses.
    """

    # Calculate suspended masses based on current plume parameters
    q = beta_plume(a, b, h1, tot_mass, z, z_min, H)

    # Calculate predicted deposit masses as a linear combination of the
    # suspended masses and the plume matrix.
    fit = np.matmul(A, q)

    misfit_contributions = []

    # A factor that penalises the plume parameters if they get too close to
    # their bounds.
    # factor_a = (1/((a-1)*(b-1)*(H-h1)))

    # for each i in the n observation masses (m)
    for i in range(len(m)):
        if gof == "chi-sqr":
            if fit[i] < 0:
                frac = (m[i] - fit[i])**2/(fit[i])
            else:
                frac = (m[i] - fit[i])**2/(fit[i] + 1e-20)
            misfit_contributions += [frac]
        elif gof == "RMSE":
            SE = (m[i] - fit[i])**2
            misfit_contributions += [SE]
        else:
            logging.error("UNKNOWN MISFIT MEASURE: %s" % gof)

    logging.debug("------misfit calculations---------")
    logging.debug("a", a)
    logging.debug("b", b)
    logging.debug("h1", h1)
    logging.debug("z values", z)
    logging.debug("z_min", z_min)
    logging.debug("H", H)
    logging.debug("tot_mass", tot_mass)
    logging.debug("q values", q)
    logging.debug("A Matrix", A)
    logging.debug("q values", q)
    logging.debug("Fit Values", fit)
    logging.debug("Obs Values", m)
    logging.debug("Misfit Values", misfit_contributions)
    logging.debug("----------------------------------")
    # This factor forces a and b closer together
    # factor_b = 1+(np.abs(a-b)/100)

    misfit = sum(misfit_contributions)  # +factor_b

    return misfit, misfit_contributions, fit


def total_misfit(
    k, setup, z, gof="chi-sqr", TGSD=None, total_mass=None, transformed=True,
    trace=False
):
    """The total misfit across all phi classes for a deposit.

    This function calculates and accumulates the error contribution of each
    phi class based on a set of parameter.

    Parameters
    ----------
    k : list(float)
        A list of invertable input parameters used to perform the linearized
        deposit prediction. This function requires all parameters, not just
        the ones being inverted.
        This list contains, in order:
        [a, b, h1, u, v, diffusion_coefficient, fall_time_threshold]
    setup : list(list)
        A list of lists of non-invertable parameters generated in the inversion
        function. Each entry in the outer list is a set of auxiliary parameters
        that correspond to a phi class.
        For each phi class, the list contains:
        m : list
            the observation dataset (array of mass/area values) for that phi
            class.
        n : int
            The number of points in the dataset.
            # TODO: I guess this is redundant #
        p : int
            The number of particle release points in the plume.
        z : list(float)
            The particle release heights in the plume.
        z_min : float
            The bottom of the eruption column (usually the vent elevation).
        elev : float
            The elevation of the accumulation plane.
        ft : list(float):
            A list of fall times of particles of a phi class from each particle
            release height.
        eddy_constant : float
            The eddy constant input parameter.
        samp_df: Dataframe
            The complete observation dataset for a phi class. Used in
            get_plume_matrix to extract the datapoint locations.
            # TODO: Could probably be replaced with the points themselves. #
        H : float
            Theoretical maximum release height (m).
        fall_time_adj : float
            The adjustment to fall time that accounts for the fall distance
            below the vent.
    z : list(float)
        Particle release heights.
    gof : str
        The Goodness-of-fit measure to use in the error calculations.
        Options:
        "chi-sqr" : The chi-squared misfit function as described in Eq ? of
            Connor and Connor (2006) [https://doi.org/10.1144/IAVCEI001.18].
        "RMSE" : The Root-mean-squared-error as described in Eq 16 of
            Connor et. al. (2019) [https://doi.org/10.1007/978-3-642-25911-1_3]
            Note that the calculation here is only the numerator part of the
            RMSE function. The rest is calculated outside of this loop.
    TGSD : list(float)
        The total grain size distribution.
    total_mass : float
        Total suspended mass in plume (kg).
    transformed : bool
        Flag that indicates whether the parameters in k are transformed.
        If True, all parameters are back-transformed with their appropriate
        transformation functions before being used.
    trace : bool
        Used internally for saving the trace to memory. Should only be true
        when used inside of Nelder-Mead solver function.
    """
    global PARAM_TRACE, MISFIT_TRACE, TGSD_TRACE, MASS_TRACE
    tot_sum = 0
    contributions = []
    pred_masses = []

    # If no TGSD is supplied, use the latest TGSD in the global trace.
    # TODO: This seems like a bad idea. #
    if TGSD is None:
        TGSD = TGSD_TRACE[-1]
    else:
        TGSD = TGSD

    # If no total mass supplied, use latest mass in the global trace.
    if total_mass is None:
        total_mass = MASS_TRACE[-1]

    # If values are transformed, back-transform using their appropriate
    # functions.
    if transformed:
        a_star = k[0]
        b_star = k[1]
        h1_star = k[2]
        a, b, h1 = plume_transform(a_star, b_star, h1_star, setup[0][9])
        u = k[3]
        v = k[4]
        diffusion_coefficient = param_transform(k[5])
        fall_time_threshold = param_transform(k[6])
    else:
        a, b, h1, u, v, \
            diffusion_coefficient, \
            fall_time_threshold = k

    # Add parameters to the global trace.
    PARAM_TRACE += [[a, b, h1, u, v, diffusion_coefficient,
                     fall_time_threshold, total_mass]]

    # for each phi class
    for stp, phi_prob in zip(setup, TGSD):
        # Unzip the aux params for this phi.
        m, n, p, z, z_min, elev, ft, \
            eddy_constant, samp_df, H, \
            fall_time_adj = stp

        # Calculate the mass proportion for this phi.
        phi_mass = total_mass*phi_prob

        logging.debug("Fall Time:")
        # logging.debug(tabulate(ft, headers="keys", tablefmt="fancy_grid"))
        logging.debug("Fall Time Adj: %g" % (fall_time_adj))

        # Construct the plume matrix for this phi.
        A = get_plume_matrix(
            u, v, n, p, z, z_min,
            elev, ft, diffusion_coefficient, fall_time_threshold,
            eddy_constant, samp_df, H, fall_time_adj
        )

        # get misfit contributions.
        phi_sum, phi_contributions, fit = misfit(
            a, b, h1, A, z, m, phi_mass, z_min, H, gof=gof)

        pred_masses += [fit]

        contributions += [phi_contributions]

        # If RMSE is used, the final sqrt and denominator part needs to be
        # applied.
        if gof == "chi-sqr":
            tot_sum += phi_sum
        elif gof == "RMSE":
            tot_sum += np.sqrt(phi_sum/n)
        else:
            logging.error("UNKNOWN MISFIT MEASURE: %s" % gof)
    if trace:
        MISFIT_TRACE += [tot_sum]

    return tot_sum, contributions, pred_masses


def custom_minimize(
    func, old_k, old_TGSD, old_total_mass, setup, z, H, include_idxes,
    exclude_idxes, all_params, gof="chi-sqr", sol_iter=10, max_iter=200,
    tol=0.1, termination="std", adjustment_factor=None, adjust_mass=True,
    adjust_TGSD=True, abort_method=None
):
    """A recursive custom minimization function that aims to optimise a set of
    parameters using a modified Nelder-Mead downhill-simplex scheme.

    Parameters
    ----------
    func : function
        The function to be passed into the downhill simplex optimiser.
        This function is dynamically generated by the inversion function based
        on the parameters that are being optimised.
    old_k : list(float)
        A list of invertable input parameters used to perform the linearized
        deposit prediction. This function requires all parameters, not just
        the ones being inverted.
        This list contains, in order:
        [a, b, h1, u, v, diffusion_coefficient, fall_time_threshold]
    old_TGSD : list(float)
        The total grain size distribution.
    old_total_mass : float
        The total erupted mass (kg).
    setup :
        A list of lists of non-invertable parameters generated in the inversion
        function. Each entry in the outer list is a set of auxiliary parameters
        that correspond to a phi class.
        For each phi class, the list contains:
        m : list
            the observation dataset (array of mass/area values) for that phi
            class.
        n : int
            The number of points in the dataset.
            # TODO: I guess this is redundant #
        p : int
            The number of particle release points in the plume.
        z : list(float)
            The particle release heights in the plume.
        z_min : float
            The bottom of the eruption column (usually the vent elevation).
        elev : float
            The elevation of the accumulation plane.
        ft : list(float):
            A list of fall times of particles of a phi class from each particle
            release height.
        eddy_constant : float
            The eddy constant input parameter.
        samp_df: Dataframe
            The complete observation dataset for a phi class. Used in
            get_plume_matrix to extract the datapoint locations.
            # TODO: Could probably be replaced with the points themselves. #
        H : float
            Theoretical maximum release height (m).
        fall_time_adj : float
            The adjustment to fall time that accounts for the fall distance
            below the vent.
    z : list(float)
        Particle release heights. (DOES THIS NEED TO BE IN setup?)
    H : float
        Theoretical maximum column height.
    include_idxes : list(int)
        list of indices corresponding to parameters that are marked for
        optimisation.
    exclude_idxes :
        list of indices corresponding to parameters that are marked for
        optimisation.
        # TODO: This can be calculated from include_idxes, right? #
    all_params :
        A list of all the parameters and their current values.
        # TODO: How is this different from old_k? #
    gof : str
        The Goodness-of-fit measure to use in the error calculations.
        Options:
        "chi-sqr" : The chi-squared misfit function as described in Eq ? of
            Connor and Connor (2006) [https://doi.org/10.1144/IAVCEI001.18].
        "RMSE" : The Root-mean-squared-error as described in Eq 16 of
            Connor et. al. (2019) [https://doi.org/10.1007/978-3-642-25911-1_3]
            Note that the calculation here is only the numerator part of the
            RMSE function. The rest is calculated outside of this loop.
    sol_iter : int
        Number of iterations to allow the downhill simplex solver to run.
    max_iter : int
        Maximum number of iterations/recursions to allow for this entire
        optimisation scheme.
    tol : float
        Tolerance of the convergence check.
    termination : str
        Termination method to use.
        Options:
        "std": standard deviation of the last N misfit values, where N is
            calculated as 2/3 of sol_iter (to account for the break-in curve)
        "norm_diff" : the absolute value of the normalised difference between
            the past two misfit values.
    adjustment_factor : float
        This is a step-size factor for the TGSD adjustment. If not specified,
        this will be dynamically estimated by the mass proportion in the
        phi-class.
    adjust_mass : bool
        Toggle for mass adjustment.
    adjust_TGSD : bool
        Toggle for TGSD adjustment.
    abort_method : str
        Method of early abortion to use.
        Options:
        "too_slow" : If the solution does not fall below one std of the mean
        within half the total conversions, abort.
    """
    global TOTAL_ITER, TGSD_TRACE, MASS_TRACE, MISFIT_TRACE, \
        FINAL_MISFIT_TRACE, ABORT_FLAG, ABORT_ITER, ABORT_MESSAGE, \
        ABORT_MISFIT, MULTI_MISFIT_TRACE

    # Using a global iteration counter to keep track of the recursions.
    TOTAL_ITER += 1

    ######################################################
    # Calculate Old Misfit.
    ######################################################
    # Update the full parameter list with the updated parameters that
    # were marked for optimisation.
    all_params[include_idxes] = np.array(old_k, dtype=np.float64)

    # Calculate the old misfit.
    old_misfit, contributions_old, \
        pred_masses_old = total_misfit(
            all_params, setup, z, TGSD=old_TGSD, gof=gof,
            total_mass=old_total_mass, transformed=False)

    if TOTAL_ITER == 1:
        FINAL_MISFIT_TRACE += [old_misfit]
    ######################################################
    # Adjust TGSD
    ######################################################

    # Extracting observation masses from setup list.
    obs_masses = [stp[0] for stp in setup]

    new_TGSD = old_TGSD.copy()
    new_mass_in_phi = []
    adjustment_list = []
    observed = []
    predicted = []

    logging.debug("Phi: \tTheo Mass: \tPred Mass: \tObs Mass: "
                  "\tAdjustment: \tFactor:")

    # for each phi class j
    for j in range(len(setup)):
        # Calculate the mass proportion of phi class j
        mass_in_phi = old_total_mass*old_TGSD[j]

        # If the masses are much smaller than the error, then the adjustment
        # blows up.
        # So here I divide by the error, so that a large error has less effect
        # if the masses are small?
        # adjustment = np.log(sum(obs_masses[j])+1) /\
        #     np.log(sum(pred_masses_old[j])+2)
        # EDIT: This is no longer how I handle things

        adjustment = sum(obs_masses[j])/sum(pred_masses_old[j])
        adjustment_list += [adjustment]

        observed += [sum(obs_masses[j])]
        predicted += [sum(pred_masses_old[j])]

        if adjustment_factor is None:
            adj = sum(obs_masses[j])/sum(sum(obs_masses))
        else:
            adj = adjustment_factor
        # new_mass_in_phi += [mass_in_phi + 0.5 *
        #                     (sum(obs_masses[j]) - sum(pred_masses_old[j]))]
        new_mass_in_phi += [(1-adj)*mass_in_phi +
                            adj*mass_in_phi*adjustment]
        logging.debug("%s \t%.2g \t%.2g \t%.2g \t%.2g \t%.2g" %
                      (str(j), mass_in_phi,
                       sum(pred_masses_old[j]),
                       sum(obs_masses[j]),
                       adjustment,
                       adj))

    # Calculate how total mass is affected by TGSD adjustment.
    sum_all_phis = sum(new_mass_in_phi)

    if adjust_mass:
        new_total_mass = sum_all_phis
    else:
        new_total_mass = old_total_mass

    # TGSD adjustment is actually applied here. It is normalised so that
    # the mass of all phis add up to the total mass.
    if adjust_TGSD:
        new_TGSD = [new_phi/sum_all_phis for new_phi in new_mass_in_phi]
    else:
        new_TGSD = old_TGSD

    TGSD_TRACE += [new_TGSD]

    MASS_TRACE += [new_total_mass]

    logging.debug("Old Total Mass: %g" % old_total_mass)
    logging.debug("New Total Mass: %g" % new_total_mass)

    logging.debug("Nelder-Mead Solver:")
    logging.debug("\t \t|a: \t\tb: \t\th1: \t\tu: \t\tv: \t\tD: \t\tftt: ")
    logging.debug("-------------------------------"
                  "-------------------------------")
    logging.debug("Before: \t|%g \t%g \t%g \t%g \t%g \t%g \t%g " %
                  (all_params[0],
                   all_params[1],
                   all_params[2],
                   all_params[3],
                   all_params[4],
                   all_params[5],
                   all_params[6]))

    ######################################################
    # perform nelder mead Downhill Simplex optimization
    ######################################################

    # constrainment transformations
    a = all_params[0]
    b = all_params[1]
    h1 = all_params[2]
    u = all_params[3]
    v = all_params[4]
    D = all_params[5]
    ftt = all_params[6]
    k_star = list(plume_inv_transform(a, b, h1, H))
    k_star += [u, v,
               param_inv_transform(D),
               param_inv_transform(ftt)]

    sol = minimize(func, np.array(k_star, dtype=np.float64)[
                   include_idxes], method="Nelder-Mead",
                   options={'maxiter': sol_iter, 'disp': True})
    logging.debug("New Total Mass: %g" % new_total_mass)
    logging.debug("Total Nelder-Mead iterations: %g" % sol.nit)

    # Untransform all parameters
    new_k_star = np.array(k_star)
    new_k_star[include_idxes] = sol.x

    a_star = new_k_star[0]
    b_star = new_k_star[1]
    h1_star = new_k_star[2]
    u_star = new_k_star[3]
    v_star = new_k_star[4]
    D_star = new_k_star[5]
    ftt_star = new_k_star[6]
    new_k = list(plume_transform(a_star, b_star, h1_star, H))
    new_k += [u_star, v_star,
              param_transform(D_star),
              param_transform(ftt_star)]

    logging.debug("After: \t\t|%g \t%g \t%g \t%g \t%g \t%g \t%g" %
                  (new_k[0],
                   new_k[1],
                   new_k[2],
                   new_k[3],
                   new_k[4],
                   new_k[5],
                   new_k[6]))

    ######################################################
    # Calculate updated misfit and check for convergence
    ######################################################

    new_misfit, contributions_new, pred_masses_new = total_misfit(
            new_k, setup, z, gof=gof,
            TGSD=new_TGSD, total_mass=new_total_mass, transformed=False)

    logging.debug("Old Misfit: %g" % old_misfit)
    logging.debug("New Misfit: %g" % new_misfit)

    FINAL_MISFIT_TRACE += [new_misfit]

    # Calculate the standard deviation of the last 5 misfit values,
    # where N is two thirds of the number of Nelder Mead iterations being run
    # (To avoid the break-in curve)
    # TODO: Turn the termination window into a parameter #
    if termination == "norm_diff":
        term_crit = np.abs(new_misfit - old_misfit)/old_misfit
    elif termination == "std":
        if len(FINAL_MISFIT_TRACE) >= 5:
            term_crit = np.std(FINAL_MISFIT_TRACE[-5:])
        else:
            term_crit = tol+tol
    else:
        raise Exception("Unknown termination method: %s" % termination)

    # Allow for the specification of additional abortion criteria
    if ABORT_FLAG is False:
        if abort_method == "too_slow":
            if len(MULTI_MISFIT_TRACE) > 0:
                final_misfits = [m[-1] for m in MULTI_MISFIT_TRACE]
                lowest_misfit = min(final_misfits)
                logging.debug("Lowest Misfit: ", lowest_misfit)
                logging.debug("Current Misfit: ", new_misfit)
                logging.debug("Misfit Cutoff: ", 1.1*lowest_misfit)
                if TOTAL_ITER > 20:
                    if new_misfit > 1.1*lowest_misfit:
                        ABORT_FLAG = True
                        ABORT_MISFIT = new_misfit
                        ABORT_ITER = TOTAL_ITER

    logging.debug("Last 5 misfits: ", FINAL_MISFIT_TRACE[-5:])
    logging.debug("Termination Metric:\n%s\n" % str(term_crit))
    logging.debug("____________________")

    if sol.success is False and sol.status != 2:
        # The status code may depend on the solver.
        # I think 2 mostly means "max iterations", so we ignore that because
        # it will hit that max by design. We only want to know if something
        # else goes wrong.
        return new_k, new_TGSD, new_total_mass, \
            False, ("Downhill-simplex solver failed with message: "
                    + sol.message)
    elif TOTAL_ITER >= max_iter:
        # This refers to the function iterations, not the downhill simplex
        # iterations.
        if ABORT_FLAG:
            ret_msg = "Solution marked for abortion " \
                "after %d iterations, " \
                "with a %s misfit of %g, " \
                "using abort method %s." % (ABORT_ITER, gof, ABORT_MISFIT,
                                            abort_method)
        else:
            ret_msg = "Maximum number of iterations exceeded."
        return new_k, new_TGSD, new_total_mass, \
            False, ret_msg
    else:
        # Successful exit criteria
        if (term_crit < tol) or (new_misfit < 1e-5):
            if ABORT_FLAG:
                ret_msg = "Solution marked for abortion " \
                    "after %d iterations, " \
                    "with a %s misfit of %g, " \
                    "using abort method %s." % (ABORT_ITER, gof, ABORT_MISFIT,
                                                abort_method)
                success_flag = False
            else:
                success_flag = True
                ret_msg = "Successful convergence."
            return new_k, new_TGSD, new_total_mass, \
                success_flag, ret_msg
        else:
            # Recursion
            return custom_minimize(func,
                                   np.array(new_k,
                                            dtype=np.float64)[include_idxes],
                                   new_TGSD, new_total_mass, setup, z, H,
                                   include_idxes, exclude_idxes, all_params,
                                   gof=gof, sol_iter=sol_iter,
                                   max_iter=max_iter, tol=tol,
                                   termination=termination,
                                   adjustment_factor=adjustment_factor,
                                   adjust_mass=adjust_mass,
                                   adjust_TGSD=adjust_TGSD,
                                   abort_method=abort_method)


def get_plume_matrix(
    u, v, num_points, column_steps, z, z_min, elevation,
    ft, diffusion_coefficient, fall_time_threshold,
    eddy_constant, samp_df, column_cap, fall_time_adj
):
    """Get the plume matrix A to be used in the linearised forward modelling
    for the misfit calculation.

    Parameters
    ----------
    u :
        x (Easting) component of the wind vector.
    v :
        y (Northing) component of the wind vector.
    num_points : int
        The number of points in the dataset.
    column_steps : list(float)
        Particle release heights (m) in the column.
    z : list(float)
        Particle release heights.
    z_min : float
        Bottom of the plume (usually the vent height).
    elevation : float
        Elevation of the particle accumulation plane.
    ft : list(float):
        A list of fall times of particles of a phi class from each particle
        release height.
    diffusion_coefficient : float
        Diffusion coefficient input parameter K.
    fall_time_threshold : float
        Threshold that separates coarse and fine particles by grain size.
        (see part_fall_time).
    eddy_constant : float
        Eddy constant input parameter.
    samp_df : Dataframe
        Full observation dataset. Used only to extract datapoint locations.
        # TODO: Replace with datapoint locations. #
    column_cap : float
        Theoretical maximum column height.
    fall_time_adj : float
        The adjustment to fall time that accounts for the fall distance
        below the vent.
    """

    height_above_vent = z - z_min

    plume_diffusion_fine_particle = [
        column_spread_fine(ht) for ht in height_above_vent]
    plume_diffusion_coarse_particle = [column_spread_coarse(
        ht, diffusion_coefficient) for ht in height_above_vent]

    wind_angle = np.arctan2(v, u)

    wind_speed = np.sqrt(u**2 + v**2)

    # Adjustment for wind below the vent.
    windspeed_adj = (wind_speed*elevation)/z_min
    u_wind_adj = np.cos(wind_angle)*windspeed_adj
    v_wind_adj = np.sin(wind_angle)*windspeed_adj
    x_adj = u_wind_adj*fall_time_adj
    y_adj = v_wind_adj*fall_time_adj

    # Extracting the sample points from the df.
    samp_x = samp_df['Easting'].values
    samp_y = samp_df["Northing"].values

    A = np.zeros((num_points, column_steps))

    wind_sum_x = 0
    wind_sum_y = 0

    for k in range(column_steps):
        total_fall_time = sum(ft[:k+1]) + fall_time_adj

        # assumes constant u and v
        wind_sum_x += ft[k]*u
        wind_sum_y += ft[k]*v

        average_windspeed_x = (wind_sum_x + x_adj)/total_fall_time
        average_windspeed_y = (wind_sum_y + y_adj)/total_fall_time

        average_wind_direction = np.arctan2(
            average_windspeed_y, average_windspeed_x)

        average_windspeed = np.sqrt(average_windspeed_x**2 +
                                    average_windspeed_y**2)

        s_sqr = sigma_squared(
            z[k], total_fall_time,
            diffusion_coefficient,
            plume_diffusion_coarse_particle[k],
            plume_diffusion_fine_particle[k],
            eddy_constant,
            fall_time_threshold
        )

        full_variance = s_sqr

        logging.debug("Sigma Squared: %.5g" % s_sqr)
        logging.debug("Full Variance: %.5g" % full_variance)
        for i in range(num_points):
            dist = strat_average(
                average_wind_direction,
                average_windspeed,
                samp_x[i], samp_y[i],
                total_fall_time, full_variance
            )
            A[i, k] = (1/(full_variance*np.pi))*dist

    return A


def gaussian_stack_inversion(
    samp_df, num_points, column_steps,
    z_min, z_max, elevation,
    phi_steps, total_mass, eddy_constant=.04, priors=None,
    invert_params=None, column_cap=45000,
    sol_iter=10, max_iter=200, tol=0.1, termination="std",
    adjust_TGSD=True,
    adjust_mass=True,
    adjustment_factor=None,
    abort_method=None,
    gof="chi-sqr",
    multi_trace=False
):
    """Perform a single run of the optimisation scheme to invert a set of
    flagged parameters.

    Parameters
    ----------
    samp_df : Dataframe
        Full observation dataset.
    num_points : int
        The number of points in the dataset.
    column_steps : list(float)
        Particle release heights (m) in the column.
    z_min : float
        Bottom of the plume (usually the vent height).
    z_max : float
        Top of the plume (Column Height).
    elevation : float
        Elevation of the particle accumulation plane.
    phi_steps : list(dict)
        List of dicts of phi-class properties, as created by get_phi_steps.
    total_mass : float
        Total suspended mass in plume (kg).
    eddy_constant : float
        Eddy constant input parameter.
    priors : dict
        Dict containing the prior values for the invertible parameters.
        Key is the string parameter name, the value is the prior guess used for
        the inversion.
    invert_params : dict
        Dict containing inversion toggles for each parameter.
        Key is the string parameter name, value is a bool indicating whether or
        not the parameter is to be inverted.
    column_cap : float
        Theoretical maximum column height.
    sol_iter : int
        Number of iterations to allow the downhill simplex solver to run.
    max_iter : int
        Maximum number of iterations/recursions to allow for this entire
        optimisation scheme.
    tol : float
        Tolerance of the convergence check.
    termination : str
        Termination method to use.
        Options:
        "std": standard deviation of the last N misfit values, where N is
            calculated as 2/3 of sol_iter (to account for the break-in curve)
        "norm_diff" : the absolute value of the normalised difference between
            the past two misfit values.
    adjust_TGSD : bool
        Toggle for TGSD adjustment.
    adjust_mass : bool
        Toggle for mass adjustment.
    adjustment_factor : float
        This is a step-size factor for the TGSD adjustment. If not specified,
        this will be dynamically estimated by the mass proportion in the
        phi-class.
    abort_method : str
        Method of early abortion to use.
        Options:
        "too_slow" : If the solution does not fall below one std of the mean
        within half the total conversions, abort.
    gof : str
        The Goodness-of-fit measure to use in the error calculations.
        Options:
        "chi-sqr" : The chi-squared misfit function as described in Eq ? of
            Connor and Connor (2006) [https://doi.org/10.1144/IAVCEI001.18].
        "RMSE" : The Root-mean-squared-error as described in Eq 16 of
            Connor et. al. (2019) [https://doi.org/10.1007/978-3-642-25911-1_3]
            Note that the calculation here is only the numerator part of the
            RMSE function. The rest is calculated outside of this loop.

    Returns
    -------
    inversion_table : Dataframe
        A dataframe containing inversion information for display purposes.
    params : dict
        A parameter dict in the same format as the prior inputs.
    misfit : float
        The final misfit result.
    status : str
        The status message returned by the solver.
    param_trace : list(list(float))
        The parameter trace throughout the optimisation.
    misfit_trace : list(float)
        The misfit trace throughout the optimisation.
    tgsd_trace : list(list(float))
        The TGSD trace throughout the optimisation.
    mass_trace : list(float)
        The mass trace throughout the optimisation.
    """
    global AIR_VISCOSITY, GRAVITY, AIR_DENSITY, ABORT_FLAG, ABORT_MESSAGE, \
        ABORT_ITER, ABORT_MISFIT

    ABORT_FLAG = False
    ABORT_MESSAGE = None
    ABORT_ITER = 0
    ABORT_MISFIT = None

    ######################################################
    # Load the non-invertible (auxiliary) parameters.
    ######################################################
    layer_thickness = ((z_max-z_min)/column_steps)
    z = np.linspace(z_min + layer_thickness, z_max, column_steps)

    # TODO: This could be generalized to take point elevation into
    # account
    distance_below_vent = z_min - elevation

    setup = []

    # for each phi class
    for phi_step in phi_steps:
        d = phi2d(phi_step["lower"])/1000

        # calculate fall time below vent height
        if distance_below_vent > 0:
            fall_time_adj = part_fall_time(
                z_min, distance_below_vent,
                d, phi_step["density"],
                AIR_DENSITY,
                GRAVITY,
                AIR_VISCOSITY
            )[0]
        else:
            fall_time_adj = 0.

        # calculate fall times through each layer.
        fall_values = [part_fall_time(
            zk,
            layer_thickness,
            d, phi_step["density"],
            AIR_DENSITY,
            GRAVITY,
            AIR_VISCOSITY
        ) for zk in z]

        fall_times = [e[0] for e in fall_values]

        # extract mass/area values from df.
        m = samp_df["MassArea"].values \
            * (samp_df[phi_step["interval"]].values / 100)

        # This list containst all non-invertible parameters.
        setup.append([
            m, num_points,
            column_steps, z, z_min, elevation, fall_times,
            eddy_constant, samp_df, column_cap,
            fall_time_adj
        ])

    ######################################################
    # Set up the inversion parameters.
    ######################################################

    # These are meant to be default (uninformed) guesses if none are specified.
    guesses = {
        "a": 2,
        "b": 2,
        "h1": z_max,
        "u": 3,
        "v": 3,
        "D": 4000,
        "ftt": 6000,
    }

    # Ordering keys into same order as default guesses above
    for key in guesses.keys():
        priors[key] = priors.pop(key)
        invert_params[key] = invert_params.pop(key)

    # Add in specified priors.
    if priors is not None:
        guesses.update(priors)

    # these are the indexes of the parameters to be inverted.
    include = list(invert_params.values())
    # this could always be calculated from include, which may be better.
    exclude = [not val for val in invert_params.values()]

    keys = list(guesses.keys())

    include_idxes = [keys.index(key) for key in np.array(keys)[include]]
    exclude_idxes = [keys.index(key) for key in np.array(keys)[exclude]]
    logging.info("Guesses")
    logging.info(guesses)
    # parameters are transformed to enforce optimisation boundaries.
    # The plume parameters are handled differently
    trans_vals = list(plume_inv_transform(guesses["a"],
                                          guesses["b"],
                                          guesses["h1"],
                                          column_cap))
    # u and v are not transformed. They can both vary as real numbers.
    trans_vals += [guesses["u"],
                   guesses["v"],
                   param_inv_transform(guesses["D"]),
                   param_inv_transform(guesses["ftt"])]

    k0 = np.array(list(guesses.values()), dtype=np.float64)[include_idxes]

    all_params = np.array(list(guesses.values()), dtype=np.float64)

    # uses: keys, include_idxes, exclude_idxes, trans_vals

    # This inversion function is dynamically created based on the parameters to
    # be inverted.
    def func(k):
        kt = np.zeros(len(keys))
        kt[include_idxes] = np.array(k, dtype=np.float64)
        kt[exclude_idxes] = np.array(trans_vals,
                                     dtype=np.float64)[exclude_idxes]
        mf, _, _ = total_misfit(kt, setup, z, gof=gof, trace=True)
        return mf

    # Loading global trace lists.
    global PARAM_TRACE, MISFIT_TRACE, TGSD_TRACE, MASS_TRACE, TOTAL_ITER, \
        FINAL_MISFIT_TRACE
    PARAM_TRACE = []
    MISFIT_TRACE = []
    FINAL_MISFIT_TRACE = []
    TOTAL_ITER = 0

    TGSD_TRACE = [[phi_step["probability"] for phi_step in phi_steps]]
    MASS_TRACE = [total_mass]

    ######################################################
    # Perform optimisation.
    ######################################################

    ret = custom_minimize(func, k0, TGSD_TRACE[0], MASS_TRACE[0],
                          setup, z, column_cap, include_idxes, exclude_idxes,
                          all_params, gof=gof, sol_iter=sol_iter,
                          max_iter=max_iter, tol=tol, termination=termination,
                          adjustment_factor=adjustment_factor,
                          adjust_mass=adjust_mass, adjust_TGSD=adjust_TGSD,
                          abort_method=abort_method)

    sol_vals, new_tgsd, new_total_mass, status, message = ret

    iterations = TOTAL_ITER

    # extract optimised parameters
    params = dict(zip(keys, sol_vals))

    # back-transform the transformed params.
    # TODO: I think all transformations should happen outside of this function
    q_inv_mass = beta_plume(params["a"],
                            params["b"],
                            params["h1"],
                            new_total_mass, z, z_min,
                            column_cap)

    # Obtain final misfit data.
    misfit, contributions, pred_masses = total_misfit(
        sol_vals, setup, z, gof=gof, TGSD=new_tgsd, total_mass=new_total_mass,
        transformed=False)

    logging.debug("a = %.5f\tb = %.5f\t\
            h1 = %.5f\tu = %.5f\tv = %.5f\t\
            D = %.5f\tftt = %.5f\tTM = %.5f" % (
            params["a"],
            params["b"],
            params["h1"],
            params["u"],
            params["v"],
            params["D"],
            params["ftt"],
            new_total_mass))
    logging.debug("Success: " + str(status))
    logging.debug("Status: " + str(message))
    logging.debug("Iterations: " + str(iterations))
    logging.debug("Misfit: " + str(misfit))

    param_trace = PARAM_TRACE.copy()
    misfit_trace = MISFIT_TRACE.copy()
    tgsd_trace = TGSD_TRACE.copy()
    mass_trace = MASS_TRACE.copy()

    # Set up output tables.
    inversion_data = np.asarray([np.asarray(z), q_inv_mass]).T
    inversion_table = pd.DataFrame(inversion_data,
                                   columns=["Height", "Suspended Mass"])
    return inversion_table, params, misfit, status, message, param_trace, \
        misfit_trace, tgsd_trace, mass_trace


def gaussian_stack_multi_run(
    data, num_points, column_steps,
    z_min, z_max, elevation,
    phi_steps, total_mass, param_config, eddy_constant=.04,
    column_cap=45000, runs=5, pre_samples=1,
    sol_iter=20, max_iter=200, tol=0.01, termination="std",
    adjust_TGSD=True, adjust_mass=True,
    adjustment_factor=None,
    abort_method=None,
    gof="chi-sqr"
):
    """gaussian_stack_multi_run.

    # TODO: Complete comments on this function. #
    Parameters
    ----------
    data : Dataframe
        Full observation dataset.
    num_points : int
        The number of points in the dataset.
    column_steps : list(float)
        Particle release heights (m) in the column.
    z_min : float
        Bottom of the plume (usually the vent height).
    z_max : float
        Top of the plume (Column Height).
    elevation : float
        Elevation of the particle accumulation plane.
    phi_steps : list(dict)
        List of dicts of phi-class properties, as created by get_phi_steps.
    total_mass : float
        Total suspended mass in plume (kg).
    param_config : Dict
        Dict containing the configuration for each parameter. The key is the
        parameter string name, and the value is a dict containing:
        "value" : is a list of distribution parameters used in the sample
            function. These will be splat into the sample function during
            sampling, so they need to be in the correct order.
        "invert" : is a boolean value indicating if the parameter should be
            inverted or not. If True, the parameter will be optimized during
            the downhill-simplex phase. Otherwise the parameter will be kept
            fixed.
        "sample_function" : is the prior distribution to sample from during the
            initial sampling phase.
    eddy_constant : float
        Eddy constant input parameter.
    column_cap : float
        Theoretical maximum column height.
    runs : int
        Number of full inversion runs to perform. The runs are collated in a
        list for comparison.
    pre_samples : int
        During the sampling phase, a number of priors will be chosen from their
        sampling functions. These will be compared using the specified misfit
        function, and the best prior will be chosen.
    sol_iter : int
        Number of iterations to allow the downhill simplex solver to run.
    max_iter : int
        Maximum number of iterations/recursions to allow for this entire
        optimisation scheme.
    tol : float
        Tolerance of the convergence check.
    termination : str
        Termination method to use.
        Options:
        "std": standard deviation of the last N misfit values, where N is
            calculated as 2/3 of sol_iter (to account for the break-in curve)
        "norm_diff" : the absolute value of the normalised difference between
            the past two misfit values.
    adjust_TGSD : bool
        Toggle for TGSD adjustment.
    adjust_mass : bool
        Toggle for mass adjustment.
    adjustment_factor : float
        This is a step-size factor for the TGSD adjustment. If not specified,
        this will be dynamically estimated by the mass proportion in the
        phi-class.
    abort_method : str
        Method of early abortion to use.
        Options:
        "too_slow" : If the solution does not fall below one std of the mean
        within half the total conversions, abort.
    gof : str
        The Goodness-of-fit measure to use in the error calculations.
        Options:
        "chi-sqr" : The chi-squared misfit function as described in Eq ? of
            Connor and Connor (2006) [https://doi.org/10.1144/IAVCEI001.18].
        "RMSE" : The Root-mean-squared-error as described in Eq 16 of
            Connor et. al. (2019) [https://doi.org/10.1007/978-3-642-25911-1_3]
            Note that the calculation here is only the numerator part of the
            RMSE function. The rest is calculated outside of this loop.
    """
    global MULTI_MISFIT_TRACE
    t_tot = process_time()

    inverted_masses_list = []
    priors_list = []
    params_list = []
    misfit_list = []
    tgsd_list = []
    mass_list = []
    status_list = []
    message_list = []

    i = 0

    while i < runs:
        t = process_time()

        logging.info("Run %d%s" % (i, '='*(80-5)))

        invert = {}

        # Extract the inversion parameters from the param config
        for key, val in param_config.items():
            invert[key] = val["invert"]

        # Creating initial sampled population.
        pre_priors_list, pre_misfit_list = generate_param_samples(
            pre_samples, param_config, data, num_points, column_steps,
            z_min, z_max, elevation, phi_steps,
            total_mass, eddy_constant, column_cap, gof=gof
        )
        logging.info("PRIORS")
        logging.info(pre_priors_list)
        logging.info(pre_misfit_list)

        # RMSE GoF requires a normalisation denominator applied to all points
        # and grain sizes in order to produce a single final GoF. For this
        # reason a "partial" RMSE is not a true RMSE.
        if gof == "RMSE":
            norm_denom = max(data["MassArea"]) - min(data["MassArea"])
            pre_misfit_list = [pm/norm_denom for pm in pre_misfit_list]

        # TODO: I'm not sure why this is necessary #
        for prior in pre_priors_list:
            if "Misfit" in prior:
                del prior["Misfit"]

        # Select prior with lowest misfit.
        best_prior = np.argsort(pre_misfit_list)[0]

        # Perform inversion
        output = gaussian_stack_inversion(
            data,
            num_points,
            column_steps,
            z_min,
            z_max,
            elevation,
            phi_steps,
            total_mass,
            invert_params=invert,
            priors=pre_priors_list[best_prior],
            column_cap=column_cap,
            sol_iter=sol_iter,
            max_iter=max_iter,
            tol=tol,
            termination=termination,
            adjust_TGSD=adjust_TGSD,
            adjust_mass=adjust_mass,
            adjustment_factor=adjustment_factor,
            gof=gof,
            abort_method=abort_method
            )
        inversion_table, params, new_misfit, status, message, param_trace, \
            misfit_trace, tgsd_trace, mass_trace = output

        MULTI_MISFIT_TRACE += [misfit_trace]

        # if status is False:
        #     # Convergence failed
        #     logging.info("DID NOT CONVERGE")
        #     logging.info(tabulate(pd.DataFrame([pre_priors_list[best_prior],
        #                  params],
        #                  index=["Priors", "Posteriors"]).T, headers="keys",
        #                  tablefmt="fancy_grid"))
        #     logging.info("Prior Misfit: %g,\t Post Misfit: %g" %
        #                  (pre_misfit_list[best_prior], new_misfit))
        # else:

        # Convergence succeeded, set of optimised parameters returned.
        priors_list += [pre_priors_list[best_prior]]

        logging.info("converged: %s" % str(status))
        logging.info(message)
        logging.info("Prior Misfit: %g,\t Post Misfit: %g" %
                     (pre_misfit_list[best_prior], new_misfit))

        fig, axs = plt.subplots(3, 3, figsize=(
            11, 9), facecolor='w', edgecolor='k')
        axs = axs.ravel()

        param_trace = np.array(param_trace)
        axs[0].plot(param_trace[:, 0], linewidth=.8)
        axs[0].set_title("a")

        axs[1].plot(param_trace[:, 1], linewidth=.8)
        axs[1].set_title("b")

        axs[2].plot(param_trace[:, 2], linewidth=.8)
        axs[2].set_title("h1")

        axs[3].plot(param_trace[:, 3], linewidth=.8)
        axs[3].set_title("u")

        axs[4].plot(param_trace[:, 4], linewidth=.8)
        axs[4].set_title("v")

        axs[5].plot(param_trace[:, 5], linewidth=.8)
        axs[5].set_title("Diffusion Coefficient")

        axs[8].plot(misfit_trace, linewidth=.8)
        axs[8].set_title("SSE")
        plt.show()

        # logging.info(tabulate(pd.DataFrame([pre_priors_list[best_prior],
        #              params],
        #              index=["Priors", "Posteriors"]).T, headers="keys",
        #              tablefmt="fancy_grid"))

        inverted_masses_list += [inversion_table["Suspended Mass"].values]
        params_list += [params]
        misfit_list += [new_misfit]
        tgsd_list += [tgsd_trace[-1]]
        mass_list += [mass_trace[-1]]
        status_list += [status]
        message_list += [message]

        i += 1
        run_time = process_time() - t
        iter_left = runs - (i+1)
        avg_time_per_run = (process_time() - t_tot)/(i+1)
        logging.info("Run %d Time: %.3f minutes\n\n" % (i, run_time/60))
        logging.info("Estimated remaining run time: %.3f minutes\n\n" %
                     (avg_time_per_run*iter_left/60))
    total_run_time = process_time() - t_tot
    logging.info("Total Run Time: %.5f minutes" % (total_run_time/60))

    return inverted_masses_list, misfit_list, params_list, priors_list, \
        inversion_table["Height"].values, tgsd_list, mass_list, status_list, \
        message_list


def gaussian_stack_genetic_optimisation(
    data, num_points, column_steps,
    z_min, z_max, elevation,
    phi_steps, total_mass, param_config, eddy_constant=.04,
    column_cap=45000, generations=5,
    sol_iter=20, max_iter=200, tol=0.01,
    adjust_TGSD=True, adjust_mass=True,
    adjustment_factor=None,
    population_size=50, mating_pool_size=10,
    crossover_method="weighted-selection",
    num_offspring=10, gof="chi-sqr"
):
    """gaussian_stack_genetic_optimisation.

    # TODO: Complete comments on this function. #

    Parameters
    ----------
    data : Dataframe
        Full observation dataset.
    num_points : int
        The number of points in the dataset.
    column_steps : list(float)
        Particle release heights (m) in the column.
    z_min : float
        Bottom of the plume (usually the vent height).
    z_max : float
        Top of the plume (Column Height).
    elevation : float
        Elevation of the particle accumulation plane.
    phi_steps : list(dict)
        List of dicts of phi-class properties, as created by get_phi_steps.
    total_mass : float
        Total suspended mass in plume (kg).
    param_config : Dict
        Dict containing the configuration for each parameter. The key is the
        parameter string name, and the value is a dict containing:
        "value" : is a list of distribution parameters used in the sample
            function. These will be splat into the sample function during
            sampling, so they need to be in the correct order.
        "invert" : is a boolean value indicating if the parameter should be
            inverted or not. If True, the parameter will be optimized during
            the downhill-simplex phase. Otherwise the parameter will be kept
            fixed.
        "sample_function" : is the prior distribution to sample from during the
            initial sampling phase.
    eddy_constant : float
        Eddy constant input parameter.
    column_cap : float
        Theoretical maximum column height.
    generations :
        generations
    sol_iter : int
        Number of iterations to allow the downhill simplex solver to run.
    max_iter : int
        Maximum number of iterations/recursions to allow for this entire
        optimisation scheme.
    tol : float
        Tolerance of the convergence check.
    adjust_TGSD : bool
        Toggle for TGSD adjustment.
    adjust_mass : bool
        Toggle for mass adjustment.
    adjustment_factor : float
        This is a step-size factor for the TGSD adjustment. If not specified,
        this will be dynamically estimated by the mass proportion in the
        phi-class.
    population_size :
        population_size
    mating_pool_size :
        mating_pool_size
    crossover_method :
        crossover_method
    num_offspring :
        num_offspring
    gof : str
        The Goodness-of-fit measure to use in the error calculations.
        Options:
        "chi-sqr" : The chi-squared misfit function as described in Eq ? of
            Connor and Connor (2006) [https://doi.org/10.1144/IAVCEI001.18].
        "RMSE" : The Root-mean-squared-error as described in Eq 16 of
            Connor et. al. (2019) [https://doi.org/10.1007/978-3-642-25911-1_3]
            Note that the calculation here is only the numerator part of the
            RMSE function. The rest is calculated outside of this loop.
    """
    t_tot = process_time()

    invert = {}

    for key, val in param_config.items():
        invert[key] = val["invert"]

    i = 0

    # CREATE INITIAL POPULATION
    pop_misfit_list = []
    pop_list = []
    for p in range(population_size):
        pop_samples = {}
        for key, val in param_config.items():
            if val["invert"]:
                pop_samples[key] = val["sample_function"](*val["value"])
            else:
                pop_samples[key] = val["value"][0]

        misfit, _, _ = get_error_contributions(
            data, num_points, column_steps,
            z_min, z_max, elevation, phi_steps,
            pop_samples, total_mass, gof=gof, eddy_constant=eddy_constant,
            column_cap=column_cap)

        pop_samples["Misfit"] = misfit
        pop_list += [pop_samples]
        pop_misfit_list += [misfit]

    # DRAW INITIAL MATING POOL

    mating_pool = pop_list
    misfit_list = pop_misfit_list
    mating_logs = []
    misfit_logs = []
    column_logs = []
    tgsd_logs = []
    mass_logs = []
    mutated_logs = []
    offspring_logs = []

    while i < generations:
        t = process_time()
        logging.info("Generation %d%s" % (i, '='*(80-5)))

        # CROSSOVER/RECOMBINATION/Getting the new mating pool

        mating_pool_idx = np.argsort(misfit_list)[0:mating_pool_size]

        best_misfit_last_gen = misfit_list[mating_pool_idx[0]]

        offspring_list, tgsd_offspring,\
            mass_offspring = weighted_selection_crossover(
                mating_pool, misfit_list, mating_pool_idx, param_config,
                num_offspring, adjust_TGSD, adjust_mass)
        misfit_list = []
        tgsd_list = []
        mass_list = []
        column_list = []
        mutated_list = []

        for offspring, in offspring_list:
            # logging.info(tabulate(offspring, tgsd_offspring, mass_offspring,
            #              headers="keys", tablefmt="fancy_grid"))

            output = gaussian_stack_inversion(
                data, num_points, column_steps, z_min,
                z_max, elevation, phi_steps, total_mass,
                invert_params=invert,
                priors=offspring,
                column_cap=column_cap,
                sol_iter=sol_iter, max_iter=max_iter, tol=tol,
                adjust_TGSD=adjust_TGSD, adjust_mass=adjust_mass,
                adjustment_factor=adjustment_factor)

            inversion_table, params, new_misfit, status, param_trace, \
                misfit_trace, tgsd_trace, mass_trace = output

            if status is False:
                logging.info("DID NOT CONVERGE")

            # logging.info(tabulate(pd.DataFrame([offspring, params],
            #              index=["Priors", "Posteriors"]).T, headers="keys",
            #              tablefmt="fancy_grid"))
            logging.info("Prev Gen Best Misfit: %g,\t This Misfit: %g" %
                         (best_misfit_last_gen, new_misfit))

            column_list += [inversion_table["Suspended Mass"].values]
            params["Misfit"] = new_misfit
            mutated_list += [params]
            misfit_list += [new_misfit]
            tgsd_list += [tgsd_trace[-1]]
            mass_list += [mass_trace[-1]]

        offspring_logs += [offspring_list]
        mating_logs += [mating_pool]
        mutated_logs += [mutated_list]
        misfit_logs += [misfit_list]
        tgsd_logs += [tgsd_list]
        mass_logs += [mass_list]
        column_logs += [column_list]

        mating_pool = mutated_list

        i += 1
        run_time = process_time() - t
        logging.info("Generation %d Time: %.3f minutes\n\n" % (i, run_time/60))
        iter_left = generations - (i+1)
        avg_time_per_run = (process_time() - t_tot)/(i+1)
        logging.info("Estimated remaining run time: %.3f minutes\n\n" %
                     (avg_time_per_run*iter_left/60))
    total_run_time = process_time() - t_tot
    logging.info("Total Run Time: %.5f minutes" % (total_run_time/60))
    return column_logs, misfit_logs, mating_logs, offspring_logs, \
        mutated_logs, inversion_table["Height"].values, tgsd_logs, mass_logs


def annealed_gaussian_stack_genetic_optimisation(
    data, num_points, column_steps,
    z_min, z_max, elevation,
    phi_steps, total_mass, param_config, eddy_constant=.04,
    column_cap=45000, generations=5,
    sol_iter=20, max_iter=200, tol=0.01,
    adjust_TGSD=True, adjust_mass=True,
    adjustment_factor=None,
    population_size=50, mating_pool_decrement=10, migrants=5,
    crossover_method="weighted-selection", gof="chi-sqr"
):
    """annealed_gaussian_stack_genetic_optimisation.

    Parameters
    ----------
    data : Dataframe
        Full observation dataset.
    num_points : int
        The number of points in the dataset.
    column_steps : list(float)
        Particle release heights (m) in the column.
    z_min : float
        Bottom of the plume (usually the vent height).
    z_max : float
        Top of the plume (Column Height).
    elevation : float
        Elevation of the particle accumulation plane.
    phi_steps : list(dict)
        List of dicts of phi-class properties, as created by get_phi_steps.
    total_mass : float
        Total suspended mass in plume (kg).
    param_config : Dict
        Dict containing the configuration for each parameter. The key is the
        parameter string name, and the value is a dict containing:
        "value" : is a list of distribution parameters used in the sample
            function. These will be splat into the sample function during
            sampling, so they need to be in the correct order.
        "invert" : is a boolean value indicating if the parameter should be
            inverted or not. If True, the parameter will be optimized during
            the downhill-simplex phase. Otherwise the parameter will be kept
            fixed.
        "sample_function" : is the prior distribution to sample from during the
            initial sampling phase.
    eddy_constant : float
        Eddy constant input parameter.
    column_cap : float
        Theoretical maximum column height.
    generations :
        generations
    sol_iter : int
        Number of iterations to allow the downhill simplex solver to run.
    max_iter : int
        Maximum number of iterations/recursions to allow for this entire
        optimisation scheme.
    tol : float
        Tolerance of the convergence check.
    adjust_TGSD : bool
        Toggle for TGSD adjustment.
    adjust_mass : bool
        Toggle for mass adjustment.
    adjustment_factor : float
        This is a step-size factor for the TGSD adjustment. If not specified,
        this will be dynamically estimated by the mass proportion in the
        phi-class.
    population_size :
        population_size
    mating_pool_decrement :
        mating_pool_decrement
    migrants :
        migrants
    crossover_method :
        crossover_method
    """
    t_tot = process_time()

    invert = {}

    for key, val in param_config.items():
        invert[key] = val["invert"]

    i = 0

    # CREATE INITIAL POPULATION
    pop_list, pop_misfit_list = generate_hypercube_samples(
        population_size, param_config, data, num_points, column_steps,
        z_min, z_max, elevation, phi_steps,
        total_mass, eddy_constant, column_cap, gof=gof
    )

    # DRAW INITIAL MATING POOL

    mating_pool = pop_list
    misfit_list = pop_misfit_list
    mating_logs = []
    misfit_logs = []
    column_logs = []
    tgsd_logs = []
    mass_logs = []
    mutated_logs = []
    offspring_logs = []

    while i < generations:
        t = process_time()
        logging.info("Generation %d%s" % (i, '='*(80-5)))

        # CROSSOVER/RECOMBINATION/Getting the new mating pool
        if (len(misfit_list) <= mating_pool_decrement):
            i = generations
            break

        num_offspring = len(misfit_list) - mating_pool_decrement
        mating_pool_idx = np.argsort(misfit_list)[0:num_offspring]

        best_misfit_last_gen = misfit_list[mating_pool_idx[0]]

        # MIGRANT WAVE
        migrant_list, migrant_misfit_list = generate_hypercube_samples(
            migrants, param_config, data, num_points, column_steps,
            z_min, z_max, elevation, phi_steps,
            total_mass, eddy_constant, column_cap, gof=gof
        )

        offspring_list, tgsd_offspring,\
            mass_offspring = weighted_selection_crossover(
                mating_pool, misfit_list, mating_pool_idx, param_config,
                num_offspring, adjust_TGSD, adjust_mass, migrants=migrant_list)
        misfit_list = []
        tgsd_list = []
        mass_list = []
        column_list = []
        mutated_list = []
        run_time = 0
        offspring_list += migrant_list
        for offspring in offspring_list:
            # logging.debug(tabulate(offspring, headers="keys",
            #               tablefmt="fancy_grid"))

            # The "Misfit" attribute messes with the inversion method.
            # It sneaks in with the migrants.
            # We must cleanse the migrants.
            prior = offspring
            if "Misfit" in prior:
                del prior["Misfit"]

            output = gaussian_stack_inversion(
                data, num_points, column_steps, z_min,
                z_max, elevation, phi_steps, total_mass,
                invert_params=invert,
                priors=prior,
                column_cap=column_cap,
                sol_iter=sol_iter, max_iter=max_iter, tol=tol,
                adjust_TGSD=adjust_TGSD, adjust_mass=adjust_mass,
                adjustment_factor=adjustment_factor)

            inversion_table, params, new_misfit, status, param_trace,\
                misfit_trace, tgsd_trace, mass_trace = output

            if status is False:
                logging.info("DID NOT CONVERGE")

            # logging.info(tabulate(pd.DataFrame([offspring, params],
            #              index=["Priors", "Posteriors"]).T, headers="keys",
            #              tablefmt="fancy_grid"))
            logging.info("Prev Gen Best Misfit: %g,\t This Misfit: %g" %
                         (best_misfit_last_gen, new_misfit))
            logging.info("Gen %d, current time %.3f minutes" %
                         (i, run_time/60))
            column_list += [inversion_table["Suspended Mass"].values]
            params["Misfit"] = new_misfit
            mutated_list += [params]
            misfit_list += [new_misfit]
            tgsd_list += [tgsd_trace[-1]]
            mass_list += [mass_trace[-1]]

        offspring_logs += [offspring_list]
        mating_logs += [mating_pool]
        mutated_logs += [mutated_list]
        misfit_logs += [misfit_list]
        tgsd_logs += [tgsd_list]
        mass_logs += [mass_list]
        column_logs += [column_list]

        mating_pool = mutated_list

        i += 1
        run_time = process_time() - t
        logging.info("Generation %d Time: %.3f minutes\n\n" % (i, run_time/60))
        iter_left = generations - (i+1)
        avg_time_per_run = (process_time() - t_tot)/(i+1)
        logging.info("Estimated remaining run time: %.3f minutes\n\n" %
                     (avg_time_per_run*iter_left/60))
    total_run_time = process_time() - t_tot
    logging.info("Total Run Time: %.5f minutes" % (total_run_time/60))
    return column_logs, misfit_logs, mating_logs, offspring_logs, \
        mutated_logs, inversion_table["Height"].values, tgsd_logs, mass_logs


def weighted_selection_crossover(
    pop_list, pop_misfit_list, mating_pool_idx, param_config, num_offspring,
    adjust_TGSD, adjust_mass, migrants=None, migration_prob=0.1
):
    """weighted_selection_crossover.

    Parameters
    ----------
    pop_list :
        pop_list
    pop_misfit_list :
        pop_misfit_list
    mating_pool_idx :
        mating_pool_idx
    param_config : Dict
        Dict containing the configuration for each parameter. The key is the
        parameter string name, and the value is a dict containing:
        "value" : is a list of distribution parameters used in the sample
            function. These will be splat into the sample function during
            sampling, so they need to be in the correct order.
        "invert" : is a boolean value indicating if the parameter should be
            inverted or not. If True, the parameter will be optimized during
            the downhill-simplex phase. Otherwise the parameter will be kept
            fixed.
        "sample_function" : is the prior distribution to sample from during the
            initial sampling phase.
    num_offspring :
        num_offspring
    adjust_TGSD : bool
        Toggle for TGSD adjustment.
    adjust_mass : bool
        Toggle for mass adjustment.
    migrants :
        migrants
    migration_prob :
        migration_prob
    """
    offspring_list = []
    tgsd_list = []
    mass_list = []
    misfit_pool = np.array(pop_misfit_list)[mating_pool_idx]
    fitness_pool = 1/misfit_pool
    norm_fitness_pool = fitness_pool/sum(fitness_pool)

    if migrants is not None:
        migrant_idx = list(range(0, len(migrants)))

    for i in range(num_offspring):
        offspring = {}
        for key, val in param_config.items():
            if val["invert"]:
                rand = np.random.random()
                if (migrants is not None) and (rand < migration_prob):
                    choice = random.choices(migrant_idx, k=1)[0]
                    offspring[key] = migrants[choice][key]
                else:
                    choice = random.choices(
                        mating_pool_idx, weights=norm_fitness_pool, k=1)[0]
                    offspring[key] = pop_list[choice][key]
            else:
                offspring[key] = val["value"][0]
        offspring_list += [offspring]
    return offspring_list, tgsd_list, mass_list


def generate_hypercube_samples(
    sample_size, param_config, data, num_points, column_steps,
    z_min, z_max, elevation, phi_steps,
    total_mass, eddy_constant, column_cap, gof="chi-sqr"
):
    """This function generates parameter samples based on a latin-hypercube
    algorithm. Each parameter distribution is divided into sections of equal
    size, and the algorithm ensures that each segment is sampled exactly once.
    Note that this method only works with uniform sampling distributions. 

    Parameters
    ----------
    sample_size :
        sample_size
    param_config : Dict
        Dict containing the configuration for each parameter. The key is the
        parameter string name, and the value is a dict containing:
        "value" : is a list of distribution parameters used in the sample
            function. These will be splat into the sample function during
            sampling, so they need to be in the correct order.
        "invert" : is a boolean value indicating if the parameter should be
            inverted or not. If True, the parameter will be optimized during
            the downhill-simplex phase. Otherwise the parameter will be kept
            fixed.
        "sample_function" : is the prior distribution to sample from during the
            initial sampling phase.
    data : Dataframe
        Full observation dataset.
    num_points : int
        The number of points in the dataset.
    column_steps : list(float)
        Particle release heights (m) in the column.
    z_min : float
        Bottom of the plume (usually the vent height).
    z_max : float
        Top of the plume (Column Height).
    elevation : float
        Elevation of the particle accumulation plane.
    phi_steps : list(dict)
        List of dicts of phi-class properties, as created by get_phi_steps.
    total_mass : float
        Total suspended mass in plume (kg).
    eddy_constant : float
        Eddy constant input parameter.
    column_cap : float
        Theoretical maximum column height.
    gof : str
        The Goodness-of-fit measure to use in the error calculations.
        Options:
        "chi-sqr" : The chi-squared misfit function as described in Eq ? of
            Connor and Connor (2006) [https://doi.org/10.1144/IAVCEI001.18].
        "RMSE" : The Root-mean-squared-error as described in Eq 16 of
            Connor et. al. (2019) [https://doi.org/10.1007/978-3-642-25911-1_3]
            Note that the calculation here is only the numerator part of the
            RMSE function. The rest is calculated outside of this loop.
    """
    sample_misfit_list = []
    sample_list = []
    # generate list of indices for each parameter

    for key, val in param_config.items():
        param_config[key]["subspaces"] = list(range(0, sample_size))

    for p in range(sample_size):
        samples = {}
        for key, val in param_config.items():

            if val["invert"]:
                bottom = val["value"][0]
                top = val["value"][1]
                hyperspace = np.linspace(bottom, top, sample_size+1)

                subspace = random.choice(val["subspaces"])

                samples[key] = val["sample_function"](
                    hyperspace[subspace], hyperspace[subspace+1])

                val["subspaces"].remove(subspace)
            else:
                samples[key] = val["value"][0]

        misfit, _, _ = get_error_contributions(
            data, num_points, column_steps,
            z_min, z_max, elevation, phi_steps,
            samples, total_mass, gof=gof, eddy_constant=eddy_constant,
            column_cap=column_cap)

        # samples["Misfit"] = misfit
        sample_list += [samples]
        sample_misfit_list += [misfit]
    return sample_list, sample_misfit_list


def generate_param_samples(
    sample_size, param_config, data, num_points, column_steps,
    z_min, z_max, elevation, phi_steps,
    total_mass, eddy_constant, column_cap, gof="chi-sqr"
):
    """This function generates parameters from their sampling distribution
    functions passed in through param_config.

    Parameters
    ----------
    sample_size :
        sample_size
    param_config : Dict
        Dict containing the configuration for each parameter. The key is the
        parameter string name, and the value is a dict containing:
        "value" : is a list of distribution parameters used in the sample
            function. These will be splat into the sample function during
            sampling, so they need to be in the correct order.
        "invert" : is a boolean value indicating if the parameter should be
            inverted or not. If True, the parameter will be optimized during
            the downhill-simplex phase. Otherwise the parameter will be kept
            fixed.
        "sample_function" : is the prior distribution to sample from during the
            initial sampling phase.
    data : Dataframe
        Full observation dataset.
    num_points : int
        The number of points in the dataset.
    column_steps : list(float)
        Particle release heights (m) in the column.
    z_min : float
        Bottom of the plume (usually the vent height).
    z_max : float
        Top of the plume (Column Height).
    elevation : float
        Elevation of the particle accumulation plane.
    phi_steps : list(dict)
        List of dicts of phi-class properties, as created by get_phi_steps.
    total_mass : float
        Total suspended mass in plume (kg).
    eddy_constant : float
        Eddy constant input parameter.
    column_cap : float
        Theoretical maximum column height.
    gof : str
        The Goodness-of-fit measure to use in the error calculations.
        Options:
        "chi-sqr" : The chi-squared misfit function as described in Eq ? of
            Connor and Connor (2006) [https://doi.org/10.1144/IAVCEI001.18].
        "RMSE" : The Root-mean-squared-error as described in Eq 16 of
            Connor et. al. (2019) [https://doi.org/10.1007/978-3-642-25911-1_3]
            Note that the calculation here is only the numerator part of the
            RMSE function. The rest is calculated outside of this loop.
    """
    sample_misfit_list = []
    sample_list = []
    for p in range(sample_size):
        samples = {}
        for key, val in param_config.items():
            if val["invert"]:
                samples[key] = val["sample_function"](*val["value"])
            else:
                samples[key] = val["value"][0]

        misfit, _, _ = get_error_contributions(
            data, num_points, column_steps,
            z_min, z_max, elevation, phi_steps,
            samples, total_mass, gof=gof, eddy_constant=eddy_constant,
            column_cap=column_cap)

        # samples["Misfit"] = misfit
        sample_list += [samples]
        sample_misfit_list += [misfit]
    return sample_list, sample_misfit_list


def get_error_contributions(
    data, num_points, column_steps,
    z_min, z_max, elevation,
    phi_steps, params, total_mass, gof="chi-sqr",
    eddy_constant=.04, column_cap=45000
):
    """get_error_contributions.

    Parameters
    ----------
    data : Dataframe
        Full observation dataset.
    num_points : int
        The number of points in the dataset.
    column_steps : list(float)
        Particle release heights (m) in the column.
    z_min : float
        Bottom of the plume (usually the vent height).
    z_max : float
        Top of the plume (Column Height).
    elevation : float
        Elevation of the particle accumulation plane.
    phi_steps : list(dict)
        List of dicts of phi-class properties, as created by get_phi_steps.
    params :
        params
    total_mass : float
        Total suspended mass in plume (kg).
    gof : str
        The Goodness-of-fit measure to use in the error calculations.
        Options:
        "chi-sqr" : The chi-squared misfit function as described in Eq ? of
            Connor and Connor (2006) [https://doi.org/10.1144/IAVCEI001.18].
        "RMSE" : The Root-mean-squared-error as described in Eq 16 of
            Connor et. al. (2019) [https://doi.org/10.1007/978-3-642-25911-1_3]
            Note that the calculation here is only the numerator part of the
            RMSE function. The rest is calculated outside of this loop.
    eddy_constant : float
        Eddy constant input parameter.
    column_cap : float
        Theoretical maximum column height.
    """
    global AIR_VISCOSITY, GRAVITY, AIR_DENSITY

    layer_thickness = ((z_max-z_min)/column_steps)
    z = np.linspace(z_min + layer_thickness, z_max, column_steps)

    # TODO: This should be generalized to take point elevation into
    # account
    distance_below_vent = z_min - elevation

    setup = []

    for phi_step in phi_steps:
        d = phi2d(phi_step["lower"])/1000

        if distance_below_vent > 0:
            fall_time_adj = part_fall_time(
                z_min, distance_below_vent,
                d, phi_step["density"],
                AIR_DENSITY,
                GRAVITY,
                AIR_VISCOSITY
            )[0]
        else:
            fall_time_adj = 0.

        fall_values = [part_fall_time(
            zk,
            layer_thickness,
            d, phi_step["density"],
            AIR_DENSITY,
            GRAVITY,
            AIR_VISCOSITY
        ) for zk in z]

        fall_times = [e[0] for e in fall_values]

        # Landing points of release point centers

        m = data["MassArea"].values \
            * (data[phi_step["interval"]].values / 100)

        setup.append([
            m, num_points,
            column_steps, z, z_min, elevation, fall_times,
            eddy_constant, data, column_cap,
            fall_time_adj
        ])

    TGSD = [phi_step["probability"] for phi_step in phi_steps]

    param_vals = np.array(list(params.values()), dtype=np.float64)

    misfit, contributions, _ = total_misfit(
        param_vals, setup, z, gof=gof, total_mass=total_mass, TGSD=TGSD,
        transformed=False)

    return misfit, contributions, setup
