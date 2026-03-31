from src.parameters import Parameters
import numpy as np
import copy
import sys

class Models:
    def __init__(self):
        self.eps = sys.float_info.epsilon

    def fv(self, v, n, i_app, parameters):
        '''Morris-Lecar equation for voltage'''
        p = parameters
        i_l = p.g_l * (v - p.E_L)
        i_k = p.g_k * n * (v - p.E_K)
        m_inf = 0.5 * (1 + np.tanh((v - p.v_1) / p.v_2))
        i_ca = p.g_ca * m_inf * (v - p.E_CA)
        fv = (1/p.CAPACITANCE) * (i_app - i_l - i_k - i_ca)
        return fv
    
    def fn(self, v, n, parameters):
        '''Morris-Lecar equation for n'''
        p = parameters
        n_inf = 0.5 * (1 + np.tanh((v - p.v_3) / p.v_4))
        tau = 1 / (np.cosh((v - p.v_3) / (2 * p.v_4)))
        fn = p.phi * ((n_inf - n) / (tau + self.eps)) # eps to avoid division by zero
        return fn
    
    def f_morris_euler(self, state, i_app, parameters, delta_t):
        '''
        Model for EKF single state only estimation
        State dim: (2,)
        '''
        # Extract state
        v = state[0]
        n = state[1]

        fv = self.fv(v, n, i_app, parameters)
        fn = self.fn(v, n, parameters)

        # Update
        v_new = v + delta_t * fv
        n_new = n + delta_t * fn
        n_new = np.clip(n_new, 0.0, 1.0) # n should be within [0 1] range
  
        state = np.array([v_new, n_new]) # Fix mixing row and column vectors
        return state
    
    def h_morris(self, state):
        '''Same shape notes as with f_morris'''
        return state[0]
    
    # EKF
    def f_morris_euler_ekf_joint(self, state, i_app, parameters, delta_t):
        '''
        Wrapper for joint estimation
        '''
        # Extract state
        p = parameters # Has fixed parameters
        state_new = state.copy()

        param_temp = Parameters([p.CAPACITANCE, p.E_CA, p.E_K, p.E_L], state[2:].copy()) 
        state_new[:2] = self.f_morris_euler(state[:2], i_app, param_temp, delta_t)

        return state_new

    # UKF
    def theta_to_params(self, theta, parameters):
        p = copy.copy(parameters)  # temporal new parameter object to work with
        p.g_l, p.g_k, p.g_ca, p.phi, p.v_1, p.v_2, p.v_3, p.v_4 = theta
        return p

    def f_morris_euler_join_ukf(self, sigma_pts_aug, i_app, parameters, delta_t):
        p = parameters
        sigma_pts_aug_out = np.zeros_like(sigma_pts_aug)
        for k, sigma_pt in enumerate(sigma_pts_aug):
            param_temp = Parameters([p.CAPACITANCE, p.E_CA, p.E_K, p.E_L], sigma_pt[2:].copy()) # So you can reuse the function
            sigma_pts_aug_out[k, :2] = self.f_morris_euler(sigma_pt[:2], i_app, param_temp, delta_t)
            sigma_pts_aug_out[k, 2:] = sigma_pt[2:] # artificial model evolution model, cov is accounted for later

        return sigma_pts_aug_out

    def h_morris_join_ukf(self, sigma_pts_aug):
        return sigma_pts_aug[:, 0:1] # return (L,1) array
