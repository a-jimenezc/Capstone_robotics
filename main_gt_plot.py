from src.plotter import Plotter
from src.parameters import Parameters
from src.simulate import Simulate
import numpy as np
import os

# General initializations
plot = Plotter()
rng = np.random.default_rng(seed=42)

fixed_parameters = np.array([20, 120, -84, -60])
delta_t = 1e-1 # ms
no_timesteps = int(200000) # 20 s
#no_timesteps = int(20000) # 2 s

folder = 'results/gt_plot'
os.makedirs(folder, exist_ok=True)


# Simulate ground truth Hopf
sim_name = 'gt_hopf'
title = 'Hopf'
var_parameters_hopf = np.array([2, 8, 4, 0.04, -1.2, 18, 2, 30])
parameters = Parameters(fixed_parameters, var_parameters_hopf)
sim_gt = Simulate(parameters)

v0 = -40
n0 = 0.25
i_app_grid = np.linspace(0, 250, 251)
amp_threshold = 2

i_app_single = 100
voltages, ns, times = sim_gt.simulate_euler(delta_t, no_timesteps, 
                                     i_app_single, v0, n0)
plot.single_sim(voltages=voltages, 
                ns=ns, 
                times=times, 
                time_max=500, 
                title=f'{title} - I_app={i_app_single}', 
                folder=folder, 
                sim_name=sim_name, 
                show=False)

data_gt = sim_gt.generate_bifurcation_data(i_app_grid, delta_t, 
                                          no_timesteps, v0, n0, 
                                          amp_threshold)
plot.plot_one_biff(data=data_gt, 
                   title=title, 
                   folder=folder, 
                   sim_name=sim_name, 
                   parameters_array=None, 
                   show=False)

# Simulate ground truth SNIC
sim_name = 'gt_snic'
title = 'SNIC'
var_parameters_snic = np.array([2, 8, 4.0, 0.067, -1.2, 18, 12, 17.4])
parameters = Parameters(fixed_parameters, var_parameters_snic)
sim_gt = Simulate(parameters)

v0 = -10
n0 = 0.35
i_app_grid = np.linspace(-20, 150, 171)
amp_threshold = 2

i_app_single = 100
voltages, ns, times = sim_gt.simulate_euler(delta_t, no_timesteps, 
                                     i_app_single, v0, n0)
plot.single_sim(voltages=voltages, 
                ns=ns, 
                times=times, 
                time_max=500, 
                title=f'{title} - I_app={i_app_single}', 
                folder=folder, 
                sim_name=sim_name, 
                show=False)

data_gt = sim_gt.generate_bifurcation_data(i_app_grid, delta_t, 
                                          no_timesteps, v0, n0, 
                                          amp_threshold)
plot.plot_one_biff(data=data_gt, 
                   title=title, 
                   folder=folder, 
                   sim_name=sim_name, 
                   parameters_array=None, 
                   show=False)

# Simulate ground truth Homoclinic
sim_name = 'gt_homoclinic'
title = 'Homoclinic'
var_parameters_homo = np.array([2, 8, 4.0, 0.23,  -1.2, 18, 12, 17.4])
parameters = Parameters(fixed_parameters, var_parameters_homo)
sim_gt = Simulate(parameters)

v0 = -20
n0 = 0
i_app_grid = np.linspace(-20, 50, 71)
amp_threshold = 2

i_app_single = 38
voltages, ns, times = sim_gt.simulate_euler(delta_t, no_timesteps, 
                                     i_app_single, v0, n0)
plot.single_sim(voltages=voltages, 
                ns=ns, 
                times=times, 
                time_max=500, 
                title=f'{title} - I_app={i_app_single}', 
                folder=folder, 
                sim_name=sim_name, 
                show=False)

data_gt = sim_gt.generate_bifurcation_data(i_app_grid, delta_t, 
                                          no_timesteps, v0, n0, 
                                          amp_threshold)
plot.plot_one_biff(data=data_gt, 
                   title=title, 
                   folder=folder, 
                   sim_name=sim_name, 
                   parameters_array=None, 
                   show=False)
