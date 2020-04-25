import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta
from matplotlib.colors import LogNorm
from functools import reduce

TRACE = []
WIND_TRACE = []
SSE_TRACE = []

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
    beta_params, tot_mass, wind, phi, particle_density, 
    diffusion_coefficient, eddy_constant, fall_time_threshold
):
    global  AIR_DENSITY, GRAVITY, AIR_VISCOSITY
    u, v = wind
    wind_angle = np.arctan(v/u)
    wind_speed = u/np.sin(wind_angle)

    # Release points in column
    layer_thickness = ((z_max-z_min)/column_steps)
    z = np.linspace(z_min + layer_thickness, z_max, column_steps)

    height_above_vent = z - z_min

    distance_below_vent = z_min

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
    
    q_mass = mass_dist_in_plume(alpha, beta, z_min, z_max, z, tot_mass)

    
    xx = grid["Easting"].values
    yy = grid["Northing"].values
    dep_mass = np.zeros(xx.shape)
    sig = []



    # For each vertical height in the column
    for k, zh in enumerate(z):

        # Adjust the total fall time by the time it 
        # takes to fall below the vent height to the ground. 
        total_fall_time = sum(ft[:k+1]) + fall_time_adj

        s_sqr = sigma_squared(
            zh, total_fall_time, 
            diffusion_coefficient,
            plume_diffusion_coarse_particle[k],
            plume_diffusion_fine_particle[k],
            eddy_constant, 
            fall_time_threshold
        )
        dist = strat_average(
            wind_angle, wind_speed, xx, yy, 
            total_fall_time, s_sqr)
        
        dep_mass += (q_mass[k]/(s_sqr*np.pi))*dist
        sig.append(s_sqr)
    dep_df = construct_dataframe(dep_mass, xx, yy)

    input_data = np.asarray([
        z, 
        np.asarray(q_mass),
        [d]*len(z),
        [particle_density]*len(z),
        ft,
        vv,
        plume_diffusion_coarse_particle,
        plume_diffusion_fine_particle,
        sig,
        [wind_angle]*len(z), 
        [wind_speed]*len(z)
    ]).T

    input_table = pd.DataFrame(
        input_data,  
        columns=[
            "Release Height (z)", 
            "Suspended Mass (q)",
            "Ash Diameter",
            "Particle Density",
            "Fall Time",
            "Terminal Velocity",
            "Col Spead Coarse",
            "Col Spead Fine",
            "Diffusion",
            "Wind Angle",
            "Wind Speed"
        ])

    return input_table, dep_df, sig, vv, ft


def gaussian_stack_forward(
    grid, column_steps, z_min, z_max, phi_steps,
    beta_params, tot_mass, wind, diffusion_coefficient, 
    eddy_constant, fall_time_threshold
):
    df_list = []
    for phi_step in phi_steps:
        mass_in_phi = tot_mass * phi_step["probability"]
        input_table, gsm_df, sig, vv, tft = gaussian_stack_single_phi(
            grid, column_steps, z_min, z_max,
            beta_params, mass_in_phi, wind, 
            phi_step["lower"], phi_step["density"], 
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

# def beta_transform(a_star, b_star, h0_star, h1_star, tot_mass, z):
#     a, b, h0, h1 = param_transform(a_star, b_star, h0_star, h1_star)
#     dist = beta.pdf(z, a, b, h0, h1)
#     return (dist/sum(dist))*tot_mass

def beta_transform(a_star, b_star, h1_star, tot_mass, z, z_min, H):
    global TRACE
    a, b, h1 = param_transform(a_star, b_star, h1_star, H)
    TRACE += [[a, b, h1]]
    heights = z[(z>=z_min) & (z<h1)]

    dist = beta.pdf(x=heights, a=a, b=b, loc=z_min, scale=(h1-z_min))
    plume = np.zeros(len(z))
    plume[(z>=z_min) & (z<h1)] = dist
    ret = (plume/sum(plume))*tot_mass
    return ret, a, b, h1

def wind_transform(w_star):
    return np.exp(w_star)

def wind_inv_transform(w):
    return np.log(w)

def param_inv_transform(a, b, h1, H):
    a_star = np.log(a - 1)
    b_star = np.log(b - 1)
    h1_star = -np.log(-np.log(h1/H))
    return a_star, b_star, h1_star

def param_transform(a_star, b_star, h1_star, H):
    a = np.exp(a_star) + 1
    b = np.exp(b_star) + 1
    h1 = H*np.exp(-np.exp(-h1_star))
    return a, b, h1


def beta_sse(k, A, z, m, tot_mass, z_min, H, lamb=0):
    # n = np.shape(A)[1]
    # A1 = np.concatenate((A, lamb*np.matlib.identity(n)))
    # b1 = np.concatenate((np.array(m), np.zeros(shape=(n,))))
    q, a, b, h1 = beta_transform(*k, tot_mass, z, z_min, H)
    

    fit = np.matmul(A, q)
    # SSE
    sse = (np.linalg.norm(fit - m)**2)/np.linalg.norm(fit)


    # RMSE (16)
    # sse = np.linalg.norm((fit - m)/np.sqrt(len(m)))

    # E (17)
    # sse = np.linalg.norm(np.log(m/fit))**2

    sse = sse*(1+(1/((a-1)*(b-1)*(H-h1)))) # Away from bounds.

    return sse



def phi_sse(k, setup, z):
    global SSE_TRACE
    tot_sum = 0
    for stp in setup:
        A, m, phi_mass, z_min, H = stp
        beta_sum = beta_sse(k, A, z, m, phi_mass, z_min, H)
        tot_sum += beta_sum
    SSE_TRACE += [tot_sum]
    return tot_sum

def phi_sse_wind(k, setup, z):
    global WIND_TRACE, SSE_TRACE
    tot_sum = 0
    for stp in setup:
        m, phi_mass, n, p, z, ft, config, samp_df, H = stp
        u = wind_transform(k[4])
        v = wind_transform(k[5])
        WIND_TRACE += [[u, v]]
        A, det, rank = get_plume_matrix(u, v, n, p, z, ft, config, samp_df)
        beta_sum = beta_sse(k[:4], A, z, m, phi_mass, H)
        tot_sum += beta_sum
    SSE_TRACE += [tot_sum]
    return tot_sum
        

def gaussian_stack_inversion(config, globs, samp_df, n, p, z_min, z_max, 
    tot_mass, wind, phi_steps, priors=None, 
    out="verb", invert_params=None, 
    column_cap=45000
):
    # Release points in column

    u, v = wind
    wind_angle = np.arctan(v/u)
    wind_speed = u/np.sin(wind_angle)

    layer_thickness = ((column_cap-z_min)/p)
    z = np.linspace(z_min + layer_thickness, column_cap, p)
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
        m = samp_df["MassArea"].values * \
            (samp_df[phi_step["interval"]].values / 100)

        # IMPORTANT THAT THE PHI VALS ARE IN RIGHT FORMAT
        A = np.zeros((n,p))
        for i in range(n):
            for k in range(p):
                s_sqr = sigma_squared(z[k], sum(ft[:k+1]), 
                                      config["DIFFUSION_COEFFICIENT"], 
                                      config["EDDY_CONST"], 
                                      config["FALL_TIME_THRESHOLD"])
                dist = strat_average(wind_angle, 
                    wind_speed, 
                    samp_x[i], 
                    samp_y[i], 
                    sum(ft[:k+1]), 
                    s_sqr)
                A[i,k] = (1/(s_sqr*np.pi))*dist
        coef_matrices.append(pd.DataFrame(A))
        if n == p:
            det = np.linalg.det(A)
        else: 
            det = None
        rank = np.linalg.matrix_rank(A)
        phi_mass = tot_mass * phi_step["probability"]
        setup.append([A, m, phi_mass, z_min, column_cap])

    guesses = {
        "a" : 2,
        "b" : 2,
        "h1" : z_max,
    }

    if priors is not None:
        guesses.update(priors)

    include = list(invert_params.values())
    exclude = [not val for val in invert_params.values()]

    trans_guesses = dict(zip(guesses.keys(), 
        param_inv_transform(*list(guesses.values()), column_cap)))

    keys = list(guesses.keys())
    trans_vals = list(trans_guesses.values())

    include_idxes = [keys.index(key) for key in np.array(keys)[include]]
    exclude_idxes = [keys.index(key) for key in np.array(keys)[exclude]]

    k0 = np.array(trans_vals, dtype=np.float64)[include_idxes]

    def func(k):
        kt = np.zeros(len(keys))
        kt[include_idxes] = np.array(k, dtype=np.float64)
        kt[exclude_idxes] = np.array(trans_vals, 
            dtype=np.float64)[exclude_idxes]
        return phi_sse(kt, setup, z)

    global TRACE
    global SSE_TRACE
    TRACE = []
    SSE_TRACE = []

    # IT HAPPENS HERE
    sol = minimize(func, k0, method='Nelder-Mead')
    # THIS IS THE THING
    
    sol_vals = np.zeros(len(keys))
    sol_vals[include_idxes] = np.array(sol.x, dtype=np.float64)
    sol_vals[exclude_idxes] = np.array(trans_vals, 
        dtype=np.float64)[exclude_idxes]
    trans_params = dict(zip(keys, sol_vals))

    param_vals = list(param_transform(trans_params["a"],
                                  trans_params["b"],
                                  trans_params["h1"],
                                  column_cap))
    params = dict(zip(keys,param_vals))

    
    q_inv_mass, _, _, _ = beta_transform(trans_params["a"], 
                                trans_params["b"],
                                trans_params["h1"],
                                tot_mass, z, z_min, column_cap)
    sse = phi_sse(list(trans_params.values()), setup, z)
    if out == "verb":
        print("a* = %.5f\tb* = %.5f\
            \th1* = %.5f"%(trans_params["a"],
                                        trans_params["b"],
                                        trans_params["h1"]))
        print("a = %.5f\tb = %.5f\th1 = %.5f"%(params["a"],
                                               params["b"],
                                               params["h1"]))
        print("Success: " + str(sol.success) + ", " + str(sol.message))
        if(hasattr(sol, "nit")):
            print("Iterations: " + str(sol.nit))
        print("SSE: " + str(sse))

    trace = TRACE.copy()
    sse_trace = SSE_TRACE.copy()
    
    inversion_data = np.asarray([np.asarray(z), q_inv_mass]).T
    inversion_table = pd.DataFrame(inversion_data, 
        columns=["Height", "Suspended Mass"])
    ret = (inversion_table, coef_matrices, 
        det, rank, params, 
        sol, sse, trace, sse_trace)
    return ret

def get_plume_matrix(u, v, n, p, z, ft, config, samp_df):
    wind_angle = np.arctan(v/u)
    wind_speed = u/np.sin(wind_angle)
    samp_x = samp_df['Easting'].values
    samp_y = samp_df["Northing"].values
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
    if n == p:
        det = np.linalg.det(A)
    else: 
        det = None
    rank = np.linalg.matrix_rank(A)
    
    return A, det, rank
    
       

def gaussian_stack_wind_inversion(config, globs, samp_df, 
    n, p, z_min, z_max, tot_mass, phi_steps, 
    priors=None, out="verb", 
    invert_params=None, column_cap=45000):
    layer_thickness = (z_max/p)
    z = np.linspace(z_min + layer_thickness, z_max, p)
    setup = []
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

        # Landing points of release point centers

        m = samp_df["MassArea"].values \
            * (samp_df[phi_step["interval"]].values / 100)
        
        phi_mass = tot_mass * phi_step["probability"]
        setup.append([m, phi_mass, n, p, z, ft, config, samp_df, column_cap])
    

    guesses = {
        "a" : 2,
        "b" : 2,
        "h0" : max(z_min, 0.01),
        "h1" : z_max,
        "u" : 3,
        "v" : 3,
    }
    

    if priors is not None:
        guesses.update(priors)

    include = list(invert_params.values())
    exclude = [not val for val in invert_params.values()]
    keys = list(guesses.keys())
    trans_vals = list(param_inv_transform(guesses["a"],
                                  guesses["b"],
                                  guesses["h0"],
                                  guesses["h1"],
                                  column_cap))
    trans_vals += [wind_inv_transform(guesses["u"]),
                   wind_inv_transform(guesses["v"])]

    trans_guesses = dict(zip(keys, trans_vals))

    include_idxes = [keys.index(key) for key in np.array(keys)[include]]
    exclude_idxes = [keys.index(key) for key in np.array(keys)[exclude]]

    k0 = np.array(trans_vals, dtype=np.float64)[include_idxes]

    def func(k):
        kt = np.zeros(len(keys))
        kt[include_idxes] = np.array(k, dtype=np.float64)
        kt[exclude_idxes] = np.array(trans_vals, 
            dtype=np.float64)[exclude_idxes]
        return phi_sse_wind(kt, setup, z)
    
    global TRACE, WIND_TRACE, SSE_TRACE
    TRACE = []
    WIND_TRACE = []
    SSE_TRACE = []

    # IT HAPPENS HERE
    sol = minimize(func, k0, method='Nelder-Mead')
    # THIS IS THE THING

    sol_vals = np.zeros(len(keys))
    sol_vals[include_idxes] = np.array(sol.x, dtype=np.float64)
    sol_vals[exclude_idxes] = np.array(trans_vals, 
        dtype=np.float64)[exclude_idxes]

    trans_params = dict(zip(keys, sol_vals))
    param_vals = list(param_transform(trans_params["a"],
                                  trans_params["b"],
                                  trans_params["h0"],
                                  trans_params["h1"],
                                  column_cap))
    param_vals += [wind_transform(trans_params["u"]),
                   wind_transform(trans_params["v"])]

    params = dict(zip(keys,param_vals))


    q_inv_mass, _, _, _, _ = beta_transform(trans_params["a"], 
                                trans_params["b"],
                                trans_params["h0"],
                                trans_params["h1"],
                                tot_mass, z,
                                column_cap)



    sse = phi_sse_wind(sol_vals, setup, z)
    
    
    if out == "verb":
        print("a* = %.5f\tb* = %.5f\t\
            h0* = %.5f\th1* = %.5f\tu* = %.5f\tv* = %.5f"%(trans_params["a"],
                                                        trans_params["b"],
                                                        trans_params["h0"],
                                                        trans_params["h1"],
                                                        trans_params["u"],
                                                        trans_params["v"]))
        print("a = %.5f\tb = %.5f\t\
            h0 = %.5f\th1 = %.5f\tu = %.5f\tv = %.5f"%(params["a"],
                                                          params["b"],
                                                          params["h0"],
                                                          params["h1"],
                                                          params["u"],
                                                          params["v"]))
        print("Success: " + str(sol.success) + ", " + str(sol.message))
        if(hasattr(sol, "nit")):
            print("Iterations: " + str(sol.nit))
        print("SSE: " + str(sse))


    trace = TRACE.copy()
    wind_trace = WIND_TRACE.copy()
    sse_trace = SSE_TRACE.copy()
    
    
    inversion_data = np.asarray([np.asarray(z), q_inv_mass]).T
    inversion_table = pd.DataFrame(inversion_data, 
        columns=["Height", "Suspended Mass"])
    return inversion_table, params, sol, sse, trace, wind_trace, sse_trace



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
                a_star, b_star, h0_star, h1_star = param_inv_transform(
                                                    a, b, h0, h1, column_cap
                                                    )
                k = np.array([a_star, b_star, h0_star, h1_star], 
                    dtype=np.float64)
                sse = phi_sse(k, setup, z)
                sse_post.append(sse)
                a_post.append(a)
                b_post.append(b)
                h1_post.append(h1)

    return a_post, b_post, h1_post, sse_post
