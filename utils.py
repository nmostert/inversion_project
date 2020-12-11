import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta
from scipy.special import betainc
from matplotlib.colors import LogNorm
from functools import reduce
from time import process_time

PARAM_TRACE = []
MISFIT_TRACE = []
TGSD_TRACE = []
MASS_TRACE = []
TOTAL_ITER = 0

BOUND_FACTOR = False
SPIKE_TERM = False

LITHIC_DIAMETER_THRESHOLD = 7.
PUMICE_DIAMETER_THRESHOLD = -1.
AIR_VISCOSITY = 0.000018325
AIR_DENSITY =  1.293
GRAVITY = 9.81

def pdf_grainsize(part_mean, part_sigma, part_max_grainsize, part_step_width):
    temp1 = 1.0 / (2.506628 * part_sigma)
    temp2 = np.exp(-(part_max_grainsize - part_mean)**2 \
            / (2*part_sigma*part_sigma))
    func_rho = temp1 * temp2 * part_step_width
    return func_rho

def get_phi_steps(
    min_grainsize, max_grainsize, part_steps, median_grainsize, std_grainsize, 
    lithic_diameter_threshold, pumice_diameter_threshold, lithic_density, 
    pumice_density
):
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
            (y - lithic_diameter_threshold)/\
            (pumice_diameter_threshold - lithic_diameter_threshold)
        
        prob = pdf_grainsize(median_grainsize, 
            std_grainsize, y, part_step_width
            )
        
        phi_class = {
            "lower": y,
            "upper": y + part_step_width,
            "interval": "[%.3g,%.3g)"%(y, y + part_step_width),
            "centroid": (y + (y + part_step_width))/2,
            "density":particle_density, 
            "probability": prob
        }
        phi_steps.append(phi_class)
        y += part_step_width
        
    return phi_steps

def get_tgsd(df, phi_steps):
    tgsd = []
    for phi in phi_steps:
        tgsd += [sum((df[phi["interval"]]/100)*df["MassArea"])/len(df)]
    tgsd = np.array(tgsd)/sum(tgsd)
    return tgsd

def sample(df, n, weight="MassArea", alpha=0.5):
    weights = df[weight].copy() # Get values to be used as weights
    weights = weights**(alpha) # Apply scaling factor as w^alpha
    probs = weights/np.sum(weights) # Normalise to sum up to one
    chosen = np.random.choice(df.index, n, replace=False, p=probs) 
    # Randomly choose n points
    return df.loc[chosen]

def d2phi(d):
    return - np.log2(d)

def phi2d(phi):
    return 2**(-phi)

def column_spread_fine(height):
    return (0.2*(height**2))**(2/5)

def column_spread_coarse(height, diffusion_coefficient):
    return 0.0032 * (height**2) / diffusion_coefficient

def fall_time(terminal_velocity, release_height):
    return release_height/terminal_velocity

def func2(x, y, sigma_sqr, x_bar, y_bar):
    return 1/(2*np.pi*sigma_sqr) * \
        np.exp(-((x - x_bar)**2 + (y - y_bar)**2)/(2*sigma_sqr))

def func(x, y, sigma_sqr, x_bar, y_bar):
    return 1/(np.pi*sigma_sqr) * \
        np.exp(-((x - x_bar)**2 + (y - y_bar)**2)/(sigma_sqr))


def sigma_squared(height, fall_time, diff_coef, spread_coarse, spread_fine, eddy_const, fall_time_thres):
    if fall_time < fall_time_thres:
        ss = 4*diff_coef*(fall_time + spread_coarse)
    else:
        ss = ((8*eddy_const)/5) * ((fall_time + spread_fine)**(5/2))
    if ss <=0:
        ss += 1e-9
    return ss

def landing_point(x1, z1, ux, vt):
    m = vt/ux
    return x1 - (z1/m)

def mass_dist_in_plume(a, b, z_min, z_max, z_points, total_mass):
    pdf = beta.pdf(z_points, a=a, b=b, loc=z_min, scale=(z_max-z_min))
    mass_dist = (pdf/sum(pdf))*total_mass
    return mass_dist

def construct_grid_dataframe(deposit, easting, northing):
    df = pd.DataFrame(deposit.T, columns=northing, index=easting)
    df.reset_index(inplace=True)
    #display(df)
    df = pd.melt(df, id_vars=["index"])
    df = df.rename(columns={
        "index": "Easting", 
        "variable": "Northing", 
        "value":"MassArea"
        })
    return df

def construct_dataframe(deposit, easting, northing):
    data = {
        "Northing":northing,
        "Easting":easting,
        "MassArea":deposit
    }

    df = pd.DataFrame(data)
    return df

def random_sample(n, df, sample_dev, K):
    transect = df[df["Northing"]==0].values
    max_point = np.argmax(transect)
    samp_x = df["Easting"].values
    return samp_x, samp_y

def part_fall_time(
    particle_ht, layer, ashdiam, part_density, air_density, gravity, 
    air_viscosity
):
    hz = particle_ht # height of particle above sea level
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
        vtt = np.sqrt( 3.1 * part_density * gravity * ashdiam / rho)
        reynolds_number =  temp0 * vtt / air_viscosity
        particle_term_vel = vtt
    
    particle_fall_time = layer / particle_term_vel

    return (particle_fall_time, particle_term_vel)

def strat_average(
    average_wind_direction, average_windspeed, xspace, yspace, total_fall_time, 
    sigma
):
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
    global  AIR_DENSITY, GRAVITY, AIR_VISCOSITY
    u, v = wind
    # Here I convert this to azimuth (clockwise from North)
    # This is me giving up because I'm bad at trig
    wind_angle = np.arctan2(v, u)
    wind_speed = np.sqrt(u**2 + v**2)

    # Release points in column
    layer_thickness = ((z_max-z_min)/column_steps)
    z = np.linspace(z_min + layer_thickness, z_max, column_steps)

    height_above_vent = z - z_min

    # TODO: This should be generalized to take point elevation into
    # account
    distance_below_vent = z_min - elevation

    windspeed_adj = (wind_speed*elevation)/z_min
    u_wind_adj = np.cos(wind_angle)*windspeed_adj
    v_wind_adj = np.sin(wind_angle)*windspeed_adj

    plume_diffusion_fine_particle = [column_spread_fine(ht) for ht in height_above_vent]
    plume_diffusion_coarse_particle = [column_spread_coarse(ht, diffusion_coefficient) for ht in height_above_vent]

    d = phi2d(phi)/1000

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


    vv = [-e[1] for e in fall_values]
    ft = [e[0] for e in fall_values]
    
    #Mass distribution in the plume
    alpha, beta = beta_params
    
    # q_mass = mass_dist_in_plume(alpha, beta, z_min, z_max, z, tot_mass)
    q_mass = beta_plume(alpha, beta, z_max, tot_mass, z, z_min, z_max)
    

    xx = grid["Easting"].values
    yy = grid["Northing"].values
    dep_mass = np.zeros(xx.shape)
    

    wind_sum_x = 0
    wind_sum_y = 0

    sig = []
    wind_speed_list = []
    wind_angle_list = []
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

        #Here we will put a proper wind field (u[k], v[k])
        x_adj = u_wind_adj*fall_time_adj
        y_adj = v_wind_adj*fall_time_adj

        wind_sum_x += ft[k]*u
        wind_sum_y += ft[k]*v

        average_windspeed_x = (wind_sum_x + x_adj)/total_fall_time
        average_windspeed_y = (wind_sum_y + y_adj)/total_fall_time

        # converting back to degrees
        average_wind_direction = np.arctan2(average_windspeed_y, average_windspeed_x)

        average_windspeed = np.sqrt(average_windspeed_x**2 + \
            average_windspeed_y**2)

        s_sqr = sigma_squared(
            zh, total_fall_time, 
            diffusion_coefficient,
            plume_diffusion_coarse_particle[k],
            plume_diffusion_fine_particle[k],
            eddy_constant, 
            fall_time_threshold
        )
        dist = strat_average(
            average_wind_direction, 
            average_windspeed, 
            xx, yy, 
            total_fall_time, s_sqr)
        
        dep_mass += (q_mass[k]/(s_sqr*np.pi))*dist

        sig.append(s_sqr)
        total_fall_time_list.append(total_fall_time)
        x_adj_list.append(x_adj)
        y_adj_list.append(y_adj)
        wind_sum_x_list.append(wind_sum_x)
        wind_sum_y_list.append(wind_sum_y)
        wind_speed_list.append(average_windspeed)
        wind_angle_list.append(average_wind_direction)

    dep_df = construct_dataframe(dep_mass, xx, yy)

    input_data = np.asarray([
        z, 
        np.asarray(q_mass),
        [d]*len(z),
        [particle_density]*len(z),
        ft,
        total_fall_time_list,
        [fall_time_adj]*len(z),
        x_adj_list,
        y_adj_list,
        vv,
        plume_diffusion_coarse_particle,
        plume_diffusion_fine_particle,
        sig,
        wind_angle_list, 
        wind_speed_list,
        wind_sum_x_list, 
        wind_sum_y_list,
        [windspeed_adj]*len(z),
        [u_wind_adj]*len(z),
        [v_wind_adj]*len(z)
    ]).T

    input_table = pd.DataFrame(
        input_data,  
        columns=[
            "Release Height (z)", 
            "Suspended Mass (q)",
            "Ash Diameter",
            "Particle Density",
            "Fall Time",
            "Total Fall Time",
            "Fall Time Adj",
            "X Adj",
            "Y Adj",
            "Terminal Velocity",
            "Col Spead Coarse",
            "Col Spead Fine",
            "Diffusion",
            "Avg. Wind Angle",
            "Avg. Wind Speed",
            "Wind Sum x",
            "Wind Sum y",
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
    df_list = []
    for phi_step in phi_steps:
        mass_in_phi = tot_mass * phi_step["probability"]
        input_table, gsm_df, sig, vv, tft = gaussian_stack_single_phi(
            grid, column_steps, z_min, z_max,
            beta_params, mass_in_phi, wind, 
            phi_step["lower"], phi_step["density"], elevation, 
            diffusion_coefficient, eddy_constant, fall_time_threshold
        )
        df_list.append(gsm_df.rename(columns={"MassArea":phi_step["interval"]}))

    df_merge = reduce(lambda x, y: pd.merge(x, y, on =['Northing', 'Easting']), df_list)
    labels = [phi_step["interval"] for phi_step in phi_steps]
    df_merge["MassArea"] = np.sum(df_merge[labels], 1)
    for label in labels:
        df_merge[label] = df_merge.apply(lambda row: (row[label]/row["MassArea"])*100, 
                                         axis=1) 
    return df_merge

def beta_function(z, a, b, h0, h1):
    return beta.pdf(z, a, b, h0, h1)

# def beta_plume(a_star, b_star, h0_star, h1_star, tot_mass, z):
#     a, b, h0, h1 = plume_transform(a_star, b_star, h0_star, h1_star)
#     dist = beta.pdf(z, a, b, h0, h1)
#     return (dist/sum(dist))*tot_mass


# THIS METHOD is an absolute mess and is causing tons of confusion. 
# I think these should go in untransformed. I shouldn't have to deal 
# with transformed variables anywhere outside the solver. 
def beta_plume(a, b, h1, tot_mass, z, z_min, H):

    heights = z[(z>=z_min) & (z<h1)]
    x_k = [(z_k-z_min)/(h1-z_min) for z_k in heights]
    dist = [betainc(a, b, x_k[i+1]) - betainc(a, b, x_k[i]) 
       for i in range(len(x_k)-1)] + [0]
    plume = np.zeros(len(z))
    plume[(z>=z_min) & (z<h1)] = dist
    q = (plume/sum(plume))*tot_mass
    return q

def param_transform(p_star):
    return np.exp(p_star)

def param_inv_transform(p):
    return np.log(p)

def plume_inv_transform(a, b, h1, H):
    a_star = np.log(a - 1)
    b_star = np.log(b - 1)
    h1_star = -np.log(-np.log(h1/H))
    return a_star, b_star, h1_star

def plume_transform(a_star, b_star, h1_star, H):
    a = np.exp(a_star) + 1
    b = np.exp(b_star) + 1
    h1 = H*np.exp(-np.exp(-h1_star))
    return a, b, h1


def misfit(a, b, h1, A, z, m, tot_mass, z_min, H):
    q = beta_plume(a, b, h1, tot_mass, z, z_min, H)

    fit = np.matmul(A, q)

    sse_contributions = []
    factor_a = (1/((a-1)*(b-1)*(H-h1)))
    
    # for each i in the n observation masses (m)
    for i in range(len(m)):
        frac = (fit[i] - m[i])**2/(fit[i]+1e-10)
        sse_contributions += [frac]

    sse = sum(sse_contributions)#+factor_a

    # This factor forces a and b closer together
    # factor_b = 1+(np.abs(a-b)/100)

    return sse, sse_contributions, fit


# I wish my car was as dirty as this function
def total_misfit(k, setup, z, TGSD=None, total_mass=None, transformed=True):
    global PARAM_TRACE, MISFIT_TRACE, TGSD_TRACE, MASS_TRACE
    tot_sum = 0
    contributions = []
    pred_masses = []

    if TGSD is None:
        TGSD = TGSD_TRACE[-1]
    else:
        TGSD = TGSD

    if total_mass is None:
        total_mass = MASS_TRACE[-1]


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
        a, b, h1, u, v, diffusion_coefficient, fall_time_threshold = k
        
    PARAM_TRACE += [[a, b, h1, u, v, diffusion_coefficient, fall_time_threshold, total_mass]]


    for stp, phi_prob in zip(setup, TGSD):
        m, n, p, z, z_min, elev, ft, \
            eddy_constant, samp_df, H, \
            fall_time_adj = stp

        phi_mass = total_mass*phi_prob

        A = get_plume_matrix(
            u, v, n, p, z, z_min, 
            elev, ft, diffusion_coefficient, fall_time_threshold, 
            eddy_constant, samp_df, H, fall_time_adj
            )
        phi_sum, phi_contributions, fit = misfit(a, b, h1, A, z, m, phi_mass, z_min, H)

        pred_masses += [fit]
        contributions += [phi_contributions]
        tot_sum += phi_sum

    MISFIT_TRACE += [tot_sum]

    return tot_sum, contributions, pred_masses


def custom_minimize(func, old_k, old_TGSD, old_total_mass, setup, z, H, include_idxes, exclude_idxes, all_params, sol_iter=10, max_iter=200, tol=0.1):
    global TOTAL_ITER, TGSD_TRACE, MASS_TRACE

    TOTAL_ITER += 1
    #Get old misfit

    ## We need all parameters here. 
    all_params[include_idxes] = np.array(old_k, dtype=np.float64)

    old_misfit, contributions_old, pred_masses_old = total_misfit(all_params, setup, z, 
                                    TGSD=old_TGSD, total_mass=old_total_mass, transformed=False)

    #Adjust TGSD

    obs_masses = [stp[0] for stp in setup]

    new_TGSD = old_TGSD.copy()
    new_mass_in_phi = []
    for j in range(len(setup)):
        mass_in_phi = old_total_mass*old_TGSD[j]
        # new_mass_in_phi += [mass_in_phi + 0.5*(sum(obs_masses[j]) - sum(pred_masses_old[j]))]
        new_mass_in_phi += [0.5*mass_in_phi + 0.5*mass_in_phi*(sum(obs_masses[j])/sum(pred_masses_old[j]))]
        
    new_total_mass = sum(new_mass_in_phi)

    new_TGSD = [new_phi/new_total_mass for new_phi in new_mass_in_phi]

    TGSD_TRACE += [new_TGSD]

    MASS_TRACE += [new_total_mass]

    #Run Solver for max_iter steps

    ## Transform for inversion
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

    sol = minimize(func, np.array(k_star, dtype=np.float64)[include_idxes], method="Nelder-Mead", options={'maxiter':sol_iter})


    ## Untransform

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

    # Calculate new misfit, and compare

    new_misfit, contributions_new, pred_masses_new = total_misfit(new_k, setup, z, 
                                    TGSD=new_TGSD, total_mass=new_total_mass, transformed=False)

    conv_crit = np.abs(new_misfit - old_misfit)/old_misfit

    if sol.success == False and sol.status is not 2:
        #The status code may depend on the solver. 
        #I think 2 mostly means "max iterations", so we ignore that because we're using
        #our own convergence criteria. 
        return new_k, new_TGSD, new_total_mass, False, ("Solver failed: " + sol.message)
    elif TOTAL_ITER >= max_iter:
        return new_k, new_TGSD, new_total_mass, False, "Maximum iterations reached."
    else:
        if conv_crit < tol:
            return new_k, new_TGSD, new_total_mass, True, "Successful convergence."
        else:
            return custom_minimize(func, np.array(new_k, dtype=np.float64)[include_idxes], new_TGSD, new_total_mass, setup, z, H, include_idxes, exclude_idxes, all_params, sol_iter, max_iter, tol)



def get_plume_matrix(
    u, v, num_samples, column_steps, z, z_min, elevation, 
    ft, diffusion_coefficient, fall_time_threshold, 
    eddy_constant, samp_df, column_cap, fall_time_adj
):
    height_above_vent = z - z_min
    plume_diffusion_fine_particle = [column_spread_fine(ht) for ht in height_above_vent]
    plume_diffusion_coarse_particle = [column_spread_coarse(ht, diffusion_coefficient) for ht in height_above_vent]


    wind_angle = np.arctan2(v, u)


    wind_speed = np.sqrt(u**2 + v**2)

    windspeed_adj = (wind_speed*elevation)/z_min

    u_wind_adj = np.cos(wind_angle)*windspeed_adj
    v_wind_adj = np.sin(wind_angle)*windspeed_adj

    x_adj = u_wind_adj*fall_time_adj 
    y_adj = v_wind_adj*fall_time_adj

    samp_x = samp_df['Easting'].values
    samp_y = samp_df["Northing"].values
    
    A = np.zeros((num_samples,column_steps))

    wind_sum_x = 0
    wind_sum_y = 0

    for k in range(column_steps):
        total_fall_time = sum(ft[:k+1]) + fall_time_adj

        wind_sum_x += ft[k]*u
        wind_sum_y += ft[k]*v

        average_windspeed_x = (wind_sum_x + x_adj)/total_fall_time
        average_windspeed_y = (wind_sum_y + y_adj)/total_fall_time

        # NOPE
        # if average_windspeed_x < 0:
        #     average_wind_direction = \
        #     np.arctan(average_windspeed_y/average_windspeed_x) + np.pi
        # else:
        #     average_wind_direction = \
        #         np.arctan(average_windspeed_y/average_windspeed_x)

        average_wind_direction = np.arctan2(average_windspeed_y, average_windspeed_x)
        
        average_windspeed = np.sqrt(average_windspeed_x**2 + \
            average_windspeed_y**2)

        s_sqr = sigma_squared(
            z[k], total_fall_time, 
            diffusion_coefficient, 
            plume_diffusion_coarse_particle[k],
            plume_diffusion_fine_particle[k],
            eddy_constant, 
            fall_time_threshold
        )

        for i in range(num_samples):
            dist = strat_average(
                average_wind_direction, 
                average_windspeed, 
                samp_x[i], samp_y[i], 
                total_fall_time, s_sqr
            )
            A[i,k] = (1/(s_sqr*np.pi))*dist

    return A
    
       

def gaussian_stack_inversion(
    samp_df, num_samples, column_steps, 
    z_min, z_max, elevation, 
    phi_steps, total_mass, eddy_constant=.04, priors=None, 
    out="verb", invert_params=None, column_cap=45000, 
    sol_iter=10, max_iter=200, tol=0.1,
    adjust_TGSD=True, 
    adjust_mass=True
):
    global AIR_VISCOSITY, GRAVITY, AIR_DENSITY

    layer_thickness = ((z_max-z_min)/column_steps)
    z = np.linspace(z_min + layer_thickness, z_max, column_steps)

    height_above_vent = z - z_min
    # TODO: This should be generalized to take point elevation into
    # account
    distance_below_vent = z_min - elevation

    # plume_diffusion_fine_particle = [column_spread_fine(ht) for ht in height_above_vent]
    # plume_diffusion_coarse_particle = [column_spread_coarse(ht, diffusion_coefficient) for ht in height_above_vent]


    setup = []
    coef_matrices = []
    
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


        terminal_velocities = [-e[1] for e in fall_values]
        fall_times = [e[0] for e in fall_values]

        # Landing points of release point centers

        m = samp_df["MassArea"].values \
            * (samp_df[phi_step["interval"]].values / 100)
        
        setup.append([
            m, num_samples, 
            column_steps, z, z_min, elevation, fall_times,
            eddy_constant, samp_df, column_cap,
            fall_time_adj
        ])
    
    # These are meant to be default (uninformed) guesses
    guesses = {
        "a" : 2,
        "b" : 2,
        "h1" : z_max,
        "u" : 3,
        "v" : 3,
        "D": 4000,
        "ftt": 6000
    }
    
    # Ordering keys into same order as default guesses above
    for key in guesses.keys():
        priors[key] = priors.pop(key)
        invert_params[key] = invert_params.pop(key)

    if priors is not None:
        guesses.update(priors)


    include = list(invert_params.values())
    exclude = [not val for val in invert_params.values()]
    keys = list(guesses.keys())

    trans_vals = list(plume_inv_transform(guesses["a"],
                                  guesses["b"],
                                  guesses["h1"],
                                  column_cap))
    trans_vals += [guesses["u"],
                   guesses["v"],
                   param_inv_transform(guesses["D"]),
                   param_inv_transform(guesses["ftt"])]
    trans_guesses = dict(zip(keys, trans_vals))

    include_idxes = [keys.index(key) for key in np.array(keys)[include]]
    exclude_idxes = [keys.index(key) for key in np.array(keys)[exclude]]

    k0 = np.array(list(guesses.values()), dtype=np.float64)[include_idxes]

    all_params = np.array(list(guesses.values()), dtype=np.float64)

    #uses: keys, include_idxes, exclude_idxes, trans_vals

    def func(k):
        kt = np.zeros(len(keys))
        kt[include_idxes] = np.array(k, dtype=np.float64)
        kt[exclude_idxes] = np.array(trans_vals, 
            dtype=np.float64)[exclude_idxes]
        mf, _, _ = total_misfit(kt, setup, z)
        return mf
    
    global PARAM_TRACE, MISFIT_TRACE, TGSD_TRACE, MASS_TRACE, TOTAL_ITER
    PARAM_TRACE = []
    MISFIT_TRACE = []
    TOTAL_ITER = 0

    TGSD_TRACE = [[phi_step["probability"] for phi_step in phi_steps]]
    MASS_TRACE = [total_mass]


    # IT HAPPENS HERE

    # sol = minimize(func, k0, method='Nelder-Mead')
    ret = custom_minimize(func, k0, TGSD_TRACE[0], MASS_TRACE[0], setup, z, column_cap, include_idxes, exclude_idxes, all_params, sol_iter=sol_iter, max_iter=max_iter, tol=tol)
    # THIS IS THE THING

    sol_vals, new_tgsd, new_total_mass, status, message = ret

    iterations = TOTAL_ITER

    # sol_vals = np.zeros(len(keys))
    # sol_vals[include_idxes] = np.array(new_k, dtype=np.float64)
    # sol_vals[exclude_idxes] = np.array(guesses.values(), 
    #     dtype=np.float64)[exclude_idxes]

    params = dict(zip(keys, sol_vals))


    # I should at least make a wrapper function 
    # for beta trans to be used outside of inversion.
    # This only works if ALL phi classes are being inverted. 
    # Single phi-class should be reconstructed outside of this function
    q_inv_mass = beta_plume(params["a"], 
                                params["b"],
                                params["h1"],
                                new_total_mass, z, z_min,
                                column_cap)


    misfit, contributions, pred_masses = total_misfit(sol_vals, setup, z, TGSD=new_tgsd, total_mass=new_total_mass, transformed=False)
    
    
    if out == "verb":
        # print("a* = %.5f\tb* = %.5f\t\
        #     h1* = %.5f\tu* = %.5f\tv* = %.5f\t\
        #     D* = %.5f\tftt* = %.5f\tTM* = %.5f"%(
        #         trans_params["a"],
        #         trans_params["b"],
        #         trans_params["h1"],
        #         trans_params["u"],
        #         trans_params["v"],
        #         trans_params["D"],
        #         trans_params["ftt"],
        #         new_total_mass))
        print("a = %.5f\tb = %.5f\t\
            h1 = %.5f\tu = %.5f\tv = %.5f\t\
            D = %.5f\tftt = %.5f\tTM = %.5f"%(
                params["a"],
                params["b"],
                params["h1"],
                params["u"],
                params["v"],
                params["D"],
                params["ftt"],
                new_total_mass))
        print("Success: " + str(status))
        print("Status: " + str(message))
        print("Iterations: " + str(iterations))
        print("Misfit: " + str(misfit))


    param_trace = PARAM_TRACE.copy()
    misfit_trace = MISFIT_TRACE.copy()
    tgsd_trace = TGSD_TRACE.copy()
    mass_trace = MASS_TRACE.copy()
    
    inversion_data = np.asarray([np.asarray(z), q_inv_mass]).T
    inversion_table = pd.DataFrame(inversion_data, 
        columns=["Height", "Suspended Mass"])
    return inversion_table, params, misfit, status, param_trace, misfit_trace, tgsd_trace, mass_trace


def gaussian_stack_multi_run(
    data, num_samples, column_steps, 
    z_min, z_max, elevation, 
    phi_steps, total_mass, param_config, eddy_constant=.04, 
    out="verb", column_cap=45000, runs=5, pre_samples=1,
    sol_iter=20, max_iter=200, tol=0.01,
    adjust_TGSD=True, adjust_mass=True
):
    t_tot = process_time()
    single_run_time = 0

    inverted_masses_list = []
    priors_list = []
    params_list = []
    misfit_list = []
    tgsd_list = []
    mass_list = []

    i = 0
    while i < runs:
        t = process_time()

        print("Run %d%s"%(i, '='*(80-5)))

       
        invert = {}

        for key, val in param_config.items():
            invert[key] = val["invert"]

        pre_misfit_list = []
        pre_priors_list = []
        
        for s in range(pre_samples):
            prior_samples = {}


            for key, val in param_config.items():
                if val["invert"]:
                    prior_samples[key] = val["sample_function"](*val["value"])
                else:
                    prior_samples[key] = val["value"][0]
            pre_priors_list += [prior_samples]

            misfit, _, _ = get_error_contributions(
                data, num_samples, column_steps, 
                z_min, z_max, elevation, phi_steps, 
                prior_samples, total_mass, eddy_constant=eddy_constant, column_cap=column_cap)
            pre_misfit_list += [misfit]

        best_prior = np.argsort(pre_misfit_list)[0]
        
        
        output = gaussian_stack_inversion(
            data, num_samples, column_steps, z_min, 
            z_max, elevation, phi_steps, total_mass,
            invert_params=invert,
            priors=pre_priors_list[best_prior],
            column_cap=column_cap, out=out,
            sol_iter=sol_iter, max_iter=max_iter, tol=tol,
            adjust_TGSD=adjust_TGSD, adjust_mass=adjust_mass)
        inversion_table, params, new_misfit, status, param_trace, misfit_trace, tgsd_trace, mass_trace = output

        # TODO: THIS IS A HACK CHANGE IT TO FALSE
        # IF EVERYTHING FAILS
        if status is False:
        # I'M SO SORRY FUTURE NIC
            print("DID NOT CONVERGE")
            display(pd.DataFrame([prior_samples, params], index=["Priors", "Posteriors"]).T)
            print("Prior Misfit: %g,\t Post Misfit: %g"%(pre_misfit_list[best_prior], new_misfit))
        else:

            priors_list += [prior_samples]

            print("Prior Misfit: %g,\t Post Misfit: %g"%(pre_misfit_list[best_prior], new_misfit))

            display(pd.DataFrame([prior_samples, params], index=["Priors", "Posteriors"]).T)

            inverted_masses_list += [inversion_table["Suspended Mass"].values]
            params_list += [params]
            misfit_list += [new_misfit]
            tgsd_list += [tgsd_trace[-1]]
            mass_list += [mass_trace[-1]]

            i += 1
        run_time = process_time() - t
        print("Run %d Time: %.3f minutes\n\n"%(i, run_time/60))
        iter_left = runs - (i+1)
        avg_time_per_run = (process_time() - t_tot)/(i+1)
        print("Estimated remaining run time: %.3f minutes\n\n"%(avg_time_per_run*iter_left/60))
    total_run_time = process_time() - t_tot
    print("Total Run Time: %.5f minutes"%(total_run_time/60))
    return inverted_masses_list, misfit_list, params_list, priors_list, inversion_table["Height"].values, tgsd_list, mass_list

def get_error_contributions(    
    data, num_samples, column_steps, 
    z_min, z_max, elevation, 
    phi_steps, params, total_mass, eddy_constant=.04, column_cap=45000
):
    global AIR_VISCOSITY, GRAVITY, AIR_DENSITY

    layer_thickness = ((z_max-z_min)/column_steps)
    z = np.linspace(z_min + layer_thickness, z_max, column_steps)

    height_above_vent = z - z_min
    # TODO: This should be generalized to take point elevation into
    # account
    distance_below_vent = z_min - elevation

    # plume_diffusion_fine_particle = [column_spread_fine(ht) for ht in height_above_vent]
    # plume_diffusion_coarse_particle = [column_spread_coarse(ht, diffusion_coefficient) for ht in height_above_vent]

    setup = []
    coef_matrices = []
    
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


        terminal_velocities = [-e[1] for e in fall_values]
        fall_times = [e[0] for e in fall_values]

        # Landing points of release point centers

        m = data["MassArea"].values \
            * (data[phi_step["interval"]].values / 100)
        
        setup.append([
            m, num_samples, 
            column_steps, z, z_min, elevation, fall_times,
            eddy_constant, data, column_cap,
            fall_time_adj
        ])

    TGSD = [phi_step["probability"] for phi_step in phi_steps]
        
    param_vals = np.array(list(params.values()), dtype=np.float64)
    
    misfit, contributions, _ = total_misfit(param_vals, setup, z, total_mass=total_mass, TGSD=TGSD, transformed=False)

    return misfit, contributions, setup



def grid_search(
    config, globs, samp_df, n, p, z_min, z_max, tot_mass, wind, phi_steps, 
    priors, solver=None, out="verb", invert_column=True, column_cap=45000
):
    u, v = wind
    wind_angle = np.arctan(v/u)
    wind_speed = u/np.sin(wind_angle)

    layer_thickness = (z_max/p)
    z = np.linspace(z_min + layer_thickness, z_max, p)
    setup = []
    coef_matrices = []
    for phi_step in phi_steps:
        d = phi2d(phi_step["lower"])/1000
        vv = [-part_fall_time(zk, layer_thickness, d, phi_step["density"], 
                              globs["AIR_DENSITY"], 
                              globs["GRAVITY"], 
                              globs["AIR_VISCOSITY"])[1] for zk in z]
        ft = [part_fall_time(zk, layer_thickness, d, phi_step["density"], 
                              globs["AIR_DENSITY"], 
                              globs["GRAVITY"], 
                              globs["AIR_VISCOSITY"])[0] for zk in z]

        samp_x = samp_df['Easting'].values
        samp_y = samp_df["Northing"].values
        m = samp_df["MassArea"].values * (samp_df[phi_step["interval"]].values \
            / 100)
        # IMPORTANT THAT THE PHI VALS ARE IN RIGHT FORMAT
        A = np.zeros((n,p))
        for i in range(n):
            for k in range(p):
                s_sqr = sigma_squared(z[k], sum(ft[:k+1]), 
                                      config["DIFFUSION_COEFFICIENT"], 
                                      config["EDDY_CONST"], 
                                      config["FALL_TIME_THRESHOLD"])
                dist = strat_average(
                    wind_angle, wind_speed, samp_x[i], samp_y[i], 
                    sum(ft[:k+1]), s_sqr
                    )
                A[i,k] = (1/(s_sqr*np.pi))*dist
        coef_matrices.append(pd.DataFrame(A))
        if n == p:
            det = np.linalg.det(A)
        else: 
            det = None
        rank = np.linalg.matrix_rank(A)
        phi_mass = tot_mass * phi_step["probability"]
        setup.append([A, m, phi_mass, column_cap])

    h0 = 0.01

    sse_post = []

    a_list = priors["a"]
    b_list = priors["b"]
    h1_list = priors["h1"]


    a_post = []
    b_post = []
    h1_post = []


    for a in a_list:
        for b in b_list:
            for h1 in h1_list:
                a_star, b_star, h0_star, h1_star = plume_inv_transform(
                                                    a, b, h0, h1, column_cap
                                                    )
                k = np.array([a_star, b_star, h0_star, h1_star], 
                    dtype=np.float64)
                sse = plume_phi_sse(k, setup, z)
                sse_post.append(sse)
                a_post.append(a)
                b_post.append(b)
                h1_post.append(h1)

    return a_post, b_post, h1_post, sse_post
