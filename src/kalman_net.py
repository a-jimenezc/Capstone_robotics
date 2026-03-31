import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from src.parameters import Parameters
from src.simulate import Simulate
from src.utils import add_noise_to_signal


def simulate_euler_step_torch(state, i_app, delta_t,
                              fixed_parameters):
    '''Reimplementation of Euler method simulation in Pytorch'''
    CAPACITANCE,  E_CA, E_K, E_L = fixed_parameters
    v, n, g_l, g_k, g_ca, phi, v_1, v_2, v_3, v_4 = state.T

    i_l = g_l * (v - E_L)
    i_k = g_k * n * (v - E_K)
    m_inf = 0.5 * (1 + torch.tanh((v - v_1) / v_2))
    i_ca = g_ca * m_inf * (v - E_CA)
    fv = (1/CAPACITANCE) * (i_app - i_l - i_k - i_ca)

    n_inf = 0.5 * (1 + torch.tanh((v - v_3) / v_4))
    tau = 1 / (torch.cosh((v - v_3) / (2 * v_4)))
    fn = phi * ((n_inf - n) / (tau))

    v_next = (v + delta_t * fv)
    n_next = (n + delta_t * fn).clamp(0.0, 1.0)

    state = torch.cat([v_next.unsqueeze(1), n_next.unsqueeze(1), state[:, 2:]], dim=1)
    return state

def h_torch(state):
    return state[:, 0] # (batch_size,)

class KalmanNet(nn.Module):
    def __init__(self, fixed_parameters, delta_t):
        super().__init__()
        self.fixed_parameters = fixed_parameters
        self.m = 10 # State dimension (joint state parameter)
        self.n = 1 # Output dimension
        self.hdim = 10 * (self.m**2 + self.n**2) # KalmanNet paper recommendation
        self.delta_t = delta_t

        # Scaling factors
        self.f2_scale = 6e4 # fine tune this
        self.f4_scale = 6e4

        # Architecture
        self.in_layer = nn.Linear(self.m+self.n, self.hdim) 
        self.gru = nn.GRU(self.hdim, self.hdim, num_layers=1) #bidirectional=False
        self.out_layer = nn.Linear(self.hdim, self.m)

    def forward(self, 
                vs_noisy_batch,
                i_app_batch,
                state0_batch
                ):
        batch_size, sim_len = vs_noisy_batch.shape
        device = vs_noisy_batch.device
        
        h = torch.zeros(1, batch_size, self.hdim, device=device)
        f4 = torch.zeros(batch_size, self.m, device=device) # Forward update difference
        state = state0_batch

        states = []
        for t in range(sim_len-1):
            state_pred = simulate_euler_step_torch(state,
                                                   i_app_batch, 
                                                   self.delta_t,
                                                   self.fixed_parameters)
            f2 = vs_noisy_batch[:, t+1] - h_torch(state_pred) # innovation difference
            features =  torch.cat([f4/self.f4_scale, f2[:,None]/self.f2_scale], dim=1) # scaling is important

            int_1 = self.in_layer(features).unsqueeze(0)
            int_2, h = self.gru(int_1, h)
            KG = self.out_layer(int_2.squeeze(0))
            KG = torch.cat([KG[:, :2], KG[:, 2:]], dim=1)

            # Correct and keep n within boundaries
            state = state_pred + KG * f2[:, None]
            state = torch.cat([state[:, :1], state[:, 1:2].clamp(0.0, 1.0), state[:, 2:]], dim=1)

            f4 = state - state_pred

            states.append(state.unsqueeze(1)) # Batch first format
        return torch.cat(states, dim=1) # (Batch_size, sim_len, 10)


class KalmanNetDataset(Dataset):
    def __init__(self,
                fixed_parameters,
                dataset_size,
                no_timesteps,
                i_app_range,
                delta_t):
        super().__init__()
        
        rng = np.random.default_rng(seed=42)

        # Hopf, SNIC, Homoclinic regines
        parameter_regimes = np.array([
            [2, 8, 4, 0.04, -1.2, 18, 2, 30],
            [2, 8, 4, 0.067, -1.2, 18, 12, 17.4],
            [2, 8, 4.0, 0.23,  -1.2, 18, 12, 17.4]
            ], dtype=float)
        
        initial_state = np.array([ # reference initial conditions for each regime
            [-40, 0.25],
            [-10, 0.35],
            [-20, 0]
        ])
        
        vs_noisy_all = [] # (dataset_size, sim_len)
        i_app_all = [] # (dataset_size,)
        states_gt_all = [] # (dataset_size, sim_len, 10)
        state0_guess_all = [] # (dataset_size, 10)
        for i in range(dataset_size):
            # Chose one out of the three regimes
            regime_idx = rng.integers(0, 3)
            params_gt = parameter_regimes[regime_idx].copy()
            v0, n0 = initial_state[regime_idx].copy()
            
            i_app = i_app_range[0] + (i_app_range[1]-i_app_range[0])*rng.random()

            parameters = Parameters(fixed_parameters, params_gt)
            sim_gt = Simulate(parameters)

            # Data is generated using the same function used for EKF
            vs, ns, _ = sim_gt.simulate_euler(
                delta_t, no_timesteps, i_app, v0, n0)
            
            sim_len = len(vs) # no_timesteps + 1
            states_gt = [vs,
                    ns,
                    params_gt[0] * np.ones(sim_len),
                    params_gt[1] * np.ones(sim_len),
                    params_gt[2] * np.ones(sim_len),
                    params_gt[3] * np.ones(sim_len),
                    params_gt[4] * np.ones(sim_len),
                    params_gt[5] * np.ones(sim_len),
                    params_gt[6] * np.ones(sim_len),
                    params_gt[7] * np.ones(sim_len),
                    ]
            states_gt = np.array(states_gt).T #(sim_len, 10)

            mean = 0
            std = np.std(vs) * 0.01 # Paper parameter
            vs_noisy = add_noise_to_signal(vs, mean, std, rng)

            # Initial guess chosenfrom the three regimes (this is similar to the inference conditions later)
            regime_idx_guess = rng.integers(0, 3)
            v0_guess, n0_guess = initial_state[regime_idx_guess].copy()
            theta_guess = parameter_regimes[regime_idx_guess].copy()
            
            # Necessary to add some randomness to force network to learn under diff. init conditions
            v0_guess = v0_guess + 1.0*rng.normal()
            n0_guess = np.clip(n0_guess + 0.01*rng.normal(), 0.0, 1.0)
            theta_guess = theta_guess + np.array([1, 2, 1, 0.01, 1, 3, 1, 3])*rng.normal(size=8)

            state0_guess = np.concatenate(([v0_guess, n0_guess], theta_guess))

            vs_noisy_all.append(vs_noisy)
            i_app_all.append(i_app)
            states_gt_all.append(states_gt)
            state0_guess_all.append(state0_guess)

        self.vs_noisy_all = torch.from_numpy(np.stack(vs_noisy_all)).float()
        self.i_app_all = torch.from_numpy(np.stack(i_app_all)).float()
        self.states_gt_all = torch.from_numpy(np.stack(states_gt_all)).float()
        self.state0_guess_all = torch.from_numpy(np.stack(state0_guess_all)).float()

    def __len__(self):
        return self.vs_noisy_all.shape[0]
    
    def __getitem__(self, idx):
        a = self.vs_noisy_all[idx]
        b = self.i_app_all[idx]
        c = self.states_gt_all[idx]
        d = self.state0_guess_all[idx]
        return a, b, c, d
