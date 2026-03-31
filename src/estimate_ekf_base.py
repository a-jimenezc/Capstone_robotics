from src.models import Models
import numpy as np

class EstimateEkfBase:
    def __init__(self):
        '''UKF hyperparameters'''
        self.model = Models()
        self.f_morris_euler = self.model.f_morris_euler
        return None
    
    def A_k(self, state, parameters, delta_t):
        raise NotImplementedError
    
    def C_k(self):
        raise NotImplementedError

    def D_k(self):
        raise NotImplementedError

    def symmetrize(self, matrix):
        return 0.5 * (matrix + matrix.T)
    
    def joint_estimate_step(self, state, state_cov, measur, Q, R, i_app, parameters, delta_t):
        '''Do both prediction and update per call'''
        # Prediction
        A_k = self.A_k(state, parameters, delta_t)
        C_k = self.C_k()
        D_k = self.D_k()

        # Prediction
        state_pred = self.f_morris_euler(state, i_app, parameters, delta_t)

        state_cov_pred = A_k @ state_cov @ A_k.T + Q # use Q instead of B_k @ q @ B_k.T. See ekf notes
        state_cov_pred = self.symmetrize(state_cov_pred)

        measur_pred = self.model.h_morris(state_pred)

        measur_cov = C_k @ state_cov_pred @ C_k.T + D_k @ R @ D_k.T
        measur_cov = self.symmetrize(measur_cov)

        # Correction
        kalman_gain = state_cov_pred @ C_k.T @ np.linalg.inv(measur_cov)

        state_corrected = state_pred + kalman_gain @ (measur - measur_pred)

        state_cov_corrected = state_cov_pred - kalman_gain @ measur_cov @ kalman_gain.T
        state_cov_corrected = self.symmetrize(state_cov_corrected)

        return state_corrected, state_cov_corrected
    
    def joint_estimate(self, init_state, init_cov, voltages, 
                       Q, R, i_app,
                       parameters, delta_t):
        # Initial state
        state = init_state
        cov = init_cov

        # Run estimation
        states = []
        covs = []
        for measur in voltages:
            measur_vec = np.array([measur])
            state, cov = self.joint_estimate_step(state, cov,
                                                        measur_vec, Q, R, i_app,
                                                        parameters, delta_t)
            states.append(state)
            covs.append(cov)
        
        states = np.array(states)
        covs = np.array(covs)
        return states, covs
    