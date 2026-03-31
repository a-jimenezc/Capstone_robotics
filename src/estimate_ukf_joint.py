from src.models import Models
import numpy as np

class EstimateUkfJoint:
    '''
    Reimplementation of the shared code in
    https://doi.org/10.1186/s13408-018-0066-8
    '''
    def __init__(self, alpha, kappa, beta, state_dim):
        '''UKF hyperparameters'''
        self.model = Models()
        
        self.alpha = alpha # 10e-2 < alpha < 1
        self.kappa = kappa # belongs to {0, 3-L}
        self.beta = beta # = 2 for Gaussian RV

        # Weights calculation
        self.L = state_dim
        lam = self.alpha**2 * (self.L + self.kappa) - self.L
        self.gamma = np.sqrt(self.L + lam)

        alpha_0_m = lam / (self.L + lam)
        alpha_k_m = 0.5 / (self.L + lam)

        # mean weights
        self.weights_mean = np.zeros((2*self.L + 1))
        self.weights_mean[0] = alpha_0_m
        self.weights_mean[1:] = alpha_k_m

        # covariance weights
        alpha_0_c = alpha_0_m + (1 - self.alpha**2 + self.beta)
        alpha_k_c = alpha_k_m
        self.weights_cov = np.zeros((2*self.L + 1))
        self.weights_cov[0] = alpha_0_c
        self.weights_cov[1:] = alpha_k_c

    def symmetrize(self, matrix):
        return 0.5 * (matrix + matrix.T)

    def sigma_points(self, mean, cov):
        L = self.L
        gamma = self.gamma
        cov = self.symmetrize(cov)
        mean = mean.reshape(-1)

        sqrt_cov = np.linalg.cholesky(cov + 1e-12*np.eye(cov.shape[0])) # Lower triangular matrix
        sigma_points = np.empty((2*L + 1, L)) # points stored in rows
        sigma_points[0] = mean

        # Traspose covariance to store in rows, and rely on numpy broadcasting
        sigma_points[1:L+1] = mean + gamma * sqrt_cov.T
        sigma_points[L+1:2*L+1] = mean - gamma * sqrt_cov.T

        return sigma_points
    
    def mean_cov_from_sigma(self, sigma_points):
        # Calculate mean (rely on numpy broadcasting)
        mean = np.sum(self.weights_mean[:,None] * sigma_points, axis = 0)

        # Calculate covariance
        # (Also relying on numpy broadcasting rules to implement weighted sum of outer products)
        sigma_points_centered = sigma_points - mean # (L,) -> (2L+1, L), then diff.
        outer_product = sigma_points_centered[:,:,None] * sigma_points_centered[:,None,:]
        cov = np.sum((self.weights_cov[:,None,None] * outer_product), axis=0)
        cov = self.symmetrize(cov)
        return mean, cov
    
    def joint_estimate_step(self, state, state_cov, measur, Q, R, i_app, parameters, delta_t):
        '''Prediction and update per call'''
        # Prediction
        state_sigma_pts = self.sigma_points(state, state_cov)
        state_sigma_pts_pred = self.model.f_morris_euler_join_ukf(state_sigma_pts, 
                                            i_app, parameters, delta_t)
        state_pred, state_cov_pred = self.mean_cov_from_sigma(state_sigma_pts_pred)
        state_cov_pred = self.symmetrize(state_cov_pred + Q) # add process noise cov

        measur_sigma_points_pred = self.model.h_morris_join_ukf(state_sigma_pts_pred)
        measur_pred, measur_cov = self.mean_cov_from_sigma(measur_sigma_points_pred)
        measur_cov = self.symmetrize(measur_cov + R) # add measurement noise cov 

        # Calculate covariance state measurement
        state_sigma_pts_pred_c = state_sigma_pts_pred - state_pred
        measur_sigma_points_pred_c =  measur_sigma_points_pred - measur_pred
        outer_product = state_sigma_pts_pred_c[:, :, None] * measur_sigma_points_pred_c[:, None, :]
        state_measur_cov = np.sum(self.weights_cov[:, None, None] * outer_product, axis=0) 

        # Correction
        kalman_gain = state_measur_cov @ np.linalg.inv(measur_cov)

        measur = float(np.asarray(measur).squeeze()) # make sure it is scalar for broadcasting
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
        states = []
        covs = []
        for measur in voltages:
            measur_vec = np.array([measur])
            state, cov = self.joint_estimate_step(state, cov,
                                                        measur_vec, Q, R, i_app,
                                                        parameters, delta_t)
            states.append(state) # shape (2+8,)
            covs.append(cov)
        
        states = np.array(states)
        covs = np.array(covs)
        return states, covs
    