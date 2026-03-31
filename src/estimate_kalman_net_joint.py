from src.kalman_net import KalmanNet
import torch
import numpy as np

class EstimateKalmanNetJoint:
    def __init__(self, ckpt_path, device):
        self.device = torch.device(device)
        ckpt = torch.load(ckpt_path, map_location=self.device,weights_only=False)
        
        self.fixed_parameters = ckpt['fixed_parameters']
        self.delta_t = ckpt['delta_t']
        self.no_timesteps = ckpt['no_timesteps']

        self.model = KalmanNet(self.fixed_parameters, self.delta_t).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

    @torch.no_grad()
    def joint_estimate(self, 
                       init_state, 
                       init_cov, # Unused
                       voltages, 
                       Q, R, # Unused
                       i_app,
                       parameters, delta_t # Unused
                       ):
        
        sim_len = len(voltages)
        voltages = torch.from_numpy(voltages).unsqueeze(0).to(self.device, dtype=torch.float32) # (1, sim_len)
        i_app = torch.tensor([(i_app)], dtype=torch.float32, device=self.device) # (1,)
        init_state = torch.from_numpy(init_state).unsqueeze(0).to(self.device, dtype=torch.float32) # (1, 10)

        # Runs estimation over trajectory, output_shape: (1, sim_len-1, 10)
        states = self.model(voltages, i_app, init_state)

        # Adding inital state
        states = torch.cat([init_state, states.squeeze(0)], dim=0) # (sim_len, 10)
        states = states.detach().cpu().numpy()

        # Placeholder for covariance
        covs = np.full((sim_len, 10, 10), np.nan)
    
        return states, covs
