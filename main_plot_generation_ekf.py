from src.plotter import Plotter
from src.parameters import Parameters
from src.estimate_ekf_joint import EstimateEkfJoint
from src.simulate import Simulate
from src.utils import add_noise_to_signal, results_to_table, results_to_table_error
import numpy as np
import os

# General initializations
plot = Plotter()
rng = np.random.default_rng(seed=42)

fixed_parameters = np.array([20, 120, -84, -60])
delta_t = 1e-1 # ms
no_timesteps = int(200000) # 20 s
#no_timesteps = int(20000) # 2 s
#no_timesteps = int(2000) # 

#folder = 'results/ekf_with_legend'
folder = 'results/ekf_no_legend'
os.makedirs(folder, exist_ok=True)

# EKF estimator initialization
state_dim = 2 + 8 # 2 states + 8 parameters
estimator = EstimateEkfJoint()

# Parameters
var_parameters_hopf = np.array([2, 8, 4, 0.04, -1.2, 18, 2, 30])
var_parameters_snic = np.array([2, 8, 4.0, 0.067, -1.2, 18, 12, 17.4])
var_parameters_homo = np.array([2, 8, 4.0, 0.23,  -1.2, 18, 12, 17.4])

results = {
    'hopf':{
        'actual': var_parameters_hopf.tolist(),
        'init_guess_hopf': None,
        'init_guess_snic': None,
        'init_guess_homo': None,},
    'snic':{
        'actual': var_parameters_snic.tolist(),
        'init_guess_hopf': None,
        'init_guess_snic': None,
        'init_guess_homo': None,},
    'homo':{
        'actual': var_parameters_homo.tolist(),
        'init_guess_hopf': None,
        'init_guess_snic': None,
        'init_guess_homo': None,}}

# ------------------Hopf------------------------------------------
# Simulate ground truth
parameters_gt = Parameters(fixed_parameters.copy(), var_parameters_hopf.copy()) # for gt sim only
sim_gt = Simulate(parameters_gt)

v0, n0 = -40, 0.25
i_app_grid = np.linspace(0, 250, 251)
amp_threshold = 2

data_gt = sim_gt.generate_bifurcation_data(i_app_grid, delta_t, 
                                          no_timesteps, v0, n0, 
                                          amp_threshold)

# Generate Noisy measurements
i_app = 100
voltages, _, _ = sim_gt.simulate_euler(
    delta_t, no_timesteps, i_app, v0, n0)

mean = 0
std_noise = np.std(voltages) * 0.01 # Paper suggestion
voltages_noisy = add_noise_to_signal(voltages, mean, std_noise, rng)

# Estimator parameters (from reference paper)
range_voltages = voltages_noisy.max() - voltages_noisy.min()
R = np.array([[std_noise**2]])

# Initial state: Hopf
init_state = np.concatenate([np.array([0, 0]), var_parameters_hopf])
init_cov = 0.001 * np.eye(state_dim)

diag_Q = np.concatenate([np.array([range_voltages, 1]), abs(var_parameters_hopf)])
Q = 1e-7 * np.diag(diag_Q)

parameters_placeholder_hopf = Parameters(fixed_parameters.copy(), init_state[2:].copy())# for fixed parameters only
states, covs = estimator.joint_estimate(init_state, init_cov,
                                           voltages_noisy, 
                                           Q, R, i_app, 
                                           parameters_placeholder_hopf, delta_t)

params_estimated = Parameters(fixed_parameters.copy(), states[-1,2:].copy())
sim_estimated = Simulate(params_estimated)
data_estimated = sim_estimated.generate_bifurcation_data(i_app_grid, delta_t, 
                                          no_timesteps, v0, n0, 
                                          amp_threshold)
                                          
plot.plot_two_biff(
        data_gt, 
        data_estimated,
        legends=('Ground Truth', 'Estimated'),
        title='Hopf Ground Truth with Hopf Initial Parameter Guess',
        folder=folder,
        sim_name='gt_hopf_initial_hopf',
        parameters_array=states[-1,2:],
        show=False
    )
results['hopf']['init_guess_hopf'] = states[-1,2:].tolist()

# Initial state SNIC
init_state = np.concatenate([np.array([0, 0]), var_parameters_snic])
init_cov = 0.001 * np.eye(state_dim)

diag_Q = np.concatenate([np.array([range_voltages, 1]), abs(var_parameters_snic)])
Q = 1e-7 * np.diag(diag_Q)

parameters_placeholder_hopf = Parameters(fixed_parameters.copy(), init_state[2:].copy()) # for fixed parameters only
states, covs = estimator.joint_estimate(init_state, init_cov,
                                           voltages_noisy, 
                                           Q, R, i_app, 
                                           parameters_placeholder_hopf, delta_t)

params_estimated = Parameters(fixed_parameters.copy(), states[-1,2:].copy())
sim_estimated = Simulate(params_estimated)
data_estimated = sim_estimated.generate_bifurcation_data(i_app_grid, delta_t, 
                                          no_timesteps, v0, n0, 
                                          amp_threshold)
                                          
plot.plot_two_biff(
        data_1=data_gt, 
        data_2=data_estimated,
        legends=('Ground Truth', 'Estimated'),
        title='Hopf Ground Truth with SNIC Initial Parameter Guess',
        folder=folder,
        sim_name='gt_hopf_initial_snic',
        parameters_array=states[-1,2:],
        show=False
    )
results['hopf']['init_guess_snic'] = states[-1,2:].tolist()

# Initial state Homoclinic
init_state = np.concatenate([np.array([0, 0]), var_parameters_homo])
init_cov = 0.001 * np.eye(state_dim)

diag_Q = np.concatenate([np.array([range_voltages, 1]), abs(var_parameters_homo)])
Q = 1e-7 * np.diag(diag_Q)

parameters_placeholder_hopf = Parameters(fixed_parameters.copy(), init_state[2:].copy()) # for fixed parameters only
states, covs = estimator.joint_estimate(init_state, init_cov,
                                           voltages_noisy, 
                                           Q, R, i_app, 
                                           parameters_placeholder_hopf, delta_t)

params_estimated = Parameters(fixed_parameters.copy(), states[-1,2:].copy())
sim_estimated = Simulate(params_estimated)
data_estimated = sim_estimated.generate_bifurcation_data(i_app_grid, delta_t, 
                                          no_timesteps, v0, n0, 
                                          amp_threshold)
                                          
plot.plot_two_biff(
        data_1=data_gt, 
        data_2=data_estimated,
        legends=('Ground Truth', 'Estimated'),
        title='Hopf Ground Truth with Homoclinic Initial Parameter Guess',
        folder=folder,
        sim_name='gt_hopf_initial_homoclinic',
        parameters_array=states[-1,2:],
        show=False
    )
results['hopf']['init_guess_homo'] = states[-1,2:].tolist()


# ------------- SNIC --------------------------------------------------
# Simulate ground truth
parameters_gt = Parameters(fixed_parameters.copy(), var_parameters_snic.copy()) # for sim gt only
sim_gt = Simulate(parameters_gt)

v0, n0 = -10, 0.35
i_app_grid = np.linspace(-20, 150, 171)
amp_threshold = 2

data_gt = sim_gt.generate_bifurcation_data(i_app_grid, delta_t, 
                                          no_timesteps, v0, n0, 
                                          amp_threshold)

# Generate Noisy measurements
i_app = 100 # check for infomative values
voltages, _, _ = sim_gt.simulate_euler(
    delta_t, no_timesteps, i_app, v0, n0)

mean = 0
std_noise = np.std(voltages) * 0.01 # Paper suggestion
voltages_noisy = add_noise_to_signal(voltages, mean, std_noise, rng)

# Estimator parameters (from reference paper)
range_voltages = voltages_noisy.max() - voltages_noisy.min()
R = np.array([[std_noise**2]])

# Initial state: Hopf
init_state = np.concatenate([np.array([0, 0]), var_parameters_hopf])
init_cov = 0.001 * np.eye(state_dim)

diag_Q = np.concatenate([np.array([range_voltages, 1]), abs(var_parameters_hopf)])
Q = 1e-7 * np.diag(diag_Q)

parameters_placeholder_snic = Parameters(fixed_parameters.copy(), init_state[2:].copy()) # for fixed parameters only
states, covs = estimator.joint_estimate(init_state, init_cov,
                                           voltages_noisy, 
                                           Q, R, i_app, 
                                           parameters_placeholder_snic, delta_t)

params_estimated = Parameters(fixed_parameters.copy(), states[-1,2:].copy())
sim_estimated = Simulate(params_estimated)
data_estimated = sim_estimated.generate_bifurcation_data(i_app_grid, delta_t, 
                                          no_timesteps, v0, n0, 
                                          amp_threshold)
                                          
plot.plot_two_biff(
        data_1=data_gt, 
        data_2=data_estimated,
        legends=('Ground Truth', 'Estimated'),
        title='SNIC Ground Truth with Hopf Initial Parameter Guess',
        folder=folder,
        sim_name='gt_snic_initial_hopf',
        parameters_array=states[-1,2:],
        show=False
    )
results['snic']['init_guess_hopf'] = states[-1,2:].tolist()

# Initial state SNIC
init_state = np.concatenate([np.array([0, 0]), var_parameters_snic])
init_cov = 0.001 * np.eye(state_dim)

diag_Q = np.concatenate([np.array([range_voltages, 1]), abs(var_parameters_snic)])
Q = 1e-7 * np.diag(diag_Q)

parameters_placeholder_snic = Parameters(fixed_parameters.copy(), init_state[2:].copy()) # for fixed parameters only
states, covs = estimator.joint_estimate(init_state, init_cov,
                                           voltages_noisy, 
                                           Q, R, i_app, 
                                           parameters_placeholder_snic, delta_t)

params_estimated = Parameters(fixed_parameters.copy(), states[-1,2:].copy())
sim_estimated = Simulate(params_estimated)
data_estimated = sim_estimated.generate_bifurcation_data(i_app_grid, delta_t, 
                                          no_timesteps, v0, n0, 
                                          amp_threshold)
                                          
plot.plot_two_biff(
        data_1=data_gt, 
        data_2=data_estimated,
        legends=('Ground Truth', 'Estimated'),
        title='Snic Ground Truth with SNIC Initial Parameter Guess',
        folder=folder,
        sim_name='gt_snic_initial_snic',
        parameters_array=states[-1,2:],
        show=False
    )
results['snic']['init_guess_snic'] = states[-1,2:].tolist()

# Initial state Homoclinic
init_state = np.concatenate([np.array([0, 0]), var_parameters_homo])
init_cov = 0.001 * np.eye(state_dim)

diag_Q = np.concatenate([np.array([range_voltages, 1]), abs(var_parameters_homo)])
Q = 1e-7 * np.diag(diag_Q)

parameters_placeholder_snic = Parameters(fixed_parameters.copy(), init_state[2:].copy()) # for fixed parameters only
states, covs = estimator.joint_estimate(init_state, init_cov,
                                           voltages_noisy, 
                                           Q, R, i_app, 
                                           parameters_placeholder_snic, delta_t)

params_estimated = Parameters(fixed_parameters.copy(), states[-1,2:].copy())
sim_estimated = Simulate(params_estimated)
data_estimated = sim_estimated.generate_bifurcation_data(i_app_grid, delta_t, 
                                          no_timesteps, v0, n0, 
                                          amp_threshold)
                                          
plot.plot_two_biff(
        data_1=data_gt, 
        data_2=data_estimated,
        legends=('Ground Truth', 'Estimated'),
        title='SNIC Ground Truth with Homoclinic Initial Parameter Guess',
        folder=folder,
        sim_name='gt_snic_initial_homoclinic',
        parameters_array=states[-1,2:],
        show=False
    )
results['snic']['init_guess_homo'] = states[-1,2:].tolist()

# ------Homoclinic---------------------------------------------------
# Simulate ground truth
parameters_gt = Parameters(fixed_parameters.copy(), var_parameters_homo.copy()) # for sim gt only
sim_gt = Simulate(parameters_gt)

v0, n0 = -20, 0
i_app_grid = np.linspace(-20, 50, 71)
amp_threshold = 2

data_gt = sim_gt.generate_bifurcation_data(i_app_grid, delta_t, 
                                          no_timesteps, v0, n0, 
                                          amp_threshold)

# Generate Noisy measurements
i_app = 38 # check for infomative values
voltages, _, _ = sim_gt.simulate_euler(
    delta_t, no_timesteps, i_app, v0, n0)

mean = 0
std_noise = np.std(voltages) * 0.01 # Paper suggestion
voltages_noisy = add_noise_to_signal(voltages, mean, std_noise, rng)

# Estimator parameters (from reference paper)
range_voltages = voltages_noisy.max() - voltages_noisy.min()
R = np.array([[std_noise**2]])

# Initial state: Hopf
init_state = np.concatenate([np.array([0, 0]), var_parameters_hopf])
init_cov = 0.001 * np.eye(state_dim)

diag_Q = np.concatenate([np.array([range_voltages, 1]), abs(var_parameters_hopf)])
Q = 1e-7 * np.diag(diag_Q)

parameters_placeholder_homo = Parameters(fixed_parameters.copy(), init_state[2:].copy()) # for fixed parameters only
states, covs = estimator.joint_estimate(init_state, init_cov,
                                           voltages_noisy, 
                                           Q, R, i_app, 
                                           parameters_placeholder_homo, delta_t)

params_estimated = Parameters(fixed_parameters.copy(), states[-1,2:].copy())
sim_estimated = Simulate(params_estimated)
data_estimated = sim_estimated.generate_bifurcation_data(i_app_grid, delta_t, 
                                          no_timesteps, v0, n0, 
                                          amp_threshold)
                                          
plot.plot_two_biff(
        data_1=data_gt, 
        data_2=data_estimated,
        legends=('Ground Truth', 'Estimated'),
        title='Homoclinic Ground Truth with Hopf Initial Parameter Guess',
        folder=folder,
        sim_name='gt_homo_initial_hopf',
        parameters_array=states[-1,2:],
        show=False
    )
results['homo']['init_guess_hopf'] = states[-1,2:].tolist()

# Initial state SNIC
init_state = np.concatenate([np.array([0, 0]), var_parameters_snic])
init_cov = 0.001 * np.eye(state_dim)

diag_Q = np.concatenate([np.array([range_voltages, 1]), abs(var_parameters_snic)])
Q = 1e-7 * np.diag(diag_Q)

parameters_placeholder_homo = Parameters(fixed_parameters.copy(), init_state[2:].copy()) # for fixed parameters only
states, covs = estimator.joint_estimate(init_state, init_cov,
                                           voltages_noisy, 
                                           Q, R, i_app, 
                                           parameters_placeholder_homo, delta_t)

params_estimated = Parameters(fixed_parameters.copy(), states[-1,2:].copy())
sim_estimated = Simulate(params_estimated)
data_estimated = sim_estimated.generate_bifurcation_data(i_app_grid, delta_t, 
                                          no_timesteps, v0, n0, 
                                          amp_threshold)
                                          
plot.plot_two_biff(
        data_1=data_gt, 
        data_2=data_estimated,
        legends=('Ground Truth', 'Estimated'),
        title='Homoclinic Ground Truth with SNIC Initial Parameter Guess',
        folder=folder,
        sim_name='gt_homo_initial_snic',
        parameters_array=states[-1,2:],
        show=False
    )
results['homo']['init_guess_snic'] = states[-1,2:].tolist()

# Initial state Homoclinic
init_state = np.concatenate([np.array([0, 0]), var_parameters_homo])
init_cov = 0.001 * np.eye(state_dim)

diag_Q = np.concatenate([np.array([range_voltages, 1]), abs(var_parameters_homo)])
Q = 1e-7 * np.diag(diag_Q)

parameters_placeholder_homo = Parameters(fixed_parameters.copy(), init_state[2:].copy()) # for fixed parameters only
states, covs = estimator.joint_estimate(init_state, init_cov,
                                           voltages_noisy, 
                                           Q, R, i_app, 
                                           parameters_placeholder_homo, delta_t)

params_estimated = Parameters(fixed_parameters.copy(), states[-1,2:].copy())
sim_estimated = Simulate(params_estimated)
data_estimated = sim_estimated.generate_bifurcation_data(i_app_grid, delta_t, 
                                          no_timesteps, v0, n0, 
                                          amp_threshold)
                                          
plot.plot_two_biff(
        data_1=data_gt, 
        data_2=data_estimated,
        legends=('Ground Truth', 'Estimated'),
        title='Homoclinic Ground Truth with Homoclinic Initial Parameter Guess',
        folder=folder,
        sim_name='gt_homo_initial_homoclinic',
        parameters_array=states[-1,2:],
        show=False
    )
results['homo']['init_guess_homo'] = states[-1,2:].tolist()


# Save in latex table format
parameters_table = results_to_table(results)
with open(os.path.join(folder, 'parameters_table.txt'), 'w') as f:
    f.write(parameters_table)


# Save percentage error in latex table format
parameters_table = results_to_table_error(results)
with open(os.path.join(folder, 'parameters_table_error.txt'), 'w') as f:
    f.write(parameters_table)