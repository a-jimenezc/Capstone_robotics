from src.models import Models
from tqdm import tqdm
import numpy as np
import sys

eps = sys.float_info.epsilon
models = Models()

class Simulate:
    def __init__(self, parameters):
        self.parameters = parameters

    def simulate_euler(self, delta_t, no_timesteps, i_app, v0, n0):
        '''Regular Euler method'''
        parameters = self.parameters
        v, n= v0, n0
        t = 0
        voltages = [v]
        ns = [n]
        times = [t]
        for i in tqdm(range(no_timesteps)):
            # Calculate fv and fn
            fv = models.fv(v, n, i_app, parameters)
            fn = models.fn(v, n, parameters)

            # Update
            v = v + delta_t * fv
            n = n + delta_t * fn
            n = max(0.0, min(1.0, n)) # n should be within [0 1] range
            t = t + delta_t

            # Store
            voltages.append(v)
            ns.append(n)
            times.append(t)
        return np.array(voltages), np.array(ns), np.array(times)

    def bisection(self, f, a, b):
        fa = f(a)
        fb = f(b)
        if np.sign(fa) == np.sign(fb): return None
        for i in range(80):
            c = 0.5 * (a+b)
            fc = f(c)
            if np.sign(fc) == np.sign(fa):
                a = c
                fa = fc
            else:
                b = c
        return 0.5 * (a + b)

    def n_inf(self, v):
        p = self.parameters
        n_inf = 0.5 * (1.0 + np.tanh((v - p.v_3) / (p.v_4) + eps))
        return n_inf
    
    def equilibria(self, i_app, v_min, v_max, nbr_points, tol_i=1e-6, tol_v=1e-2):
        '''
        Solve for dv(v, n_inf(v)/dt = 0, see reference book
        '''
        voltages = np.linspace(v_min, v_max, nbr_points)
        
        # Calculate dv(v, n_inf(v)/dt for voltages
        fv_inf = [models.fv(v, self.n_inf(v), i_app, self.parameters) for v in voltages]
        fv_inf = np.array(fv_inf)

        # Find zeros
        zeros = []
        zero_idxs = np.nonzero(np.abs(fv_inf) < tol_i)[0]
        for idx in zero_idxs:
            zeros.append(voltages[idx])

        f = lambda v: models.fv(v, self.n_inf(v), i_app, self.parameters)
        signs = np.sign(fv_inf)
        opposite_sign =  signs[:-1] * signs[1:] < 0
        sign_change = np.asarray(opposite_sign).nonzero()[0]
        for k in sign_change:
            a, b = voltages[k], voltages[k+1]
            zero = self.bisection(f, a, b) # refine zero location
            if zero is not None: # necessary safeguard
                zeros.append(zero)

        # Eliminate duplicates
        zeros = sorted(zeros)
        unique_zeros = []
        for zero in zeros:
            if not unique_zeros or abs(zero - unique_zeros[-1]) > tol_v:
                unique_zeros.append(zero)
        
        equilibria = []
        for v_eq in unique_zeros:
            n_eq = self.n_inf(v_eq)
            equilibria.append((v_eq, n_eq))
        
        return equilibria
    
    def cycle_envelope(self, i_app, delta_t, no_timesteps,
                       v0, n0, amp_threshold):
        v, n, _ = self.simulate_euler(delta_t, no_timesteps, 
                                      i_app, v0, n0)
        last_v, last_n = v[-1], n[-1]
        v_steady_state = v[int(no_timesteps/2):] # Hard coded steady state range
        amplitude = v_steady_state.max() - v_steady_state.min()
        if amplitude >= amp_threshold:
            v_min = v_steady_state.min()
            v_max = v_steady_state.max()
            return True, v_min, v_max, last_v, last_n
        else:
            v_mean = v_steady_state.mean()
            return False, v_mean, v_mean, last_v, last_n

    def generate_bifurcation_data(self, i_apps, delta_t, no_timesteps,
                       v0, n0, amp_threshold):
        
        # equilibria
        eq_i, eq_v = [], []
        v_min = -60
        v_max = 40
        nbr_voltage_points = 2000
        for i_app in i_apps:
            equilibria = self.equilibria(i_app, v_min, 
                                  v_max, nbr_voltage_points)
            for (v_eq, n_eq) in equilibria:
                eq_i.append(i_app)
                eq_v.append(v_eq)
        eq_i_v = np.array([eq_i, eq_v])
        
        # Envelope ascending pass
        v_init = v0
        n_init = n0
        asc_i_apps, asc_v_mins, asc_v_maxs = [], [], []
        for i_app in i_apps:
            is_cycle, v_min_env, v_max_env, last_v, last_n = self.cycle_envelope(i_app, delta_t, 
                                                                     no_timesteps,
                                                                     v_init, n_init, amp_threshold)
            v_init = last_v
            n_init = last_n
            if is_cycle:
                asc_i_apps.append(i_app)
                asc_v_mins.append(v_min_env)
                asc_v_maxs.append(v_max_env)

        asc_iapps_vmin_vmax = np.array([asc_i_apps,
                                    asc_v_mins,
                                    asc_v_maxs])

        # Envelope descending pass
        v_init = v0
        n_init = n0
        des_i_apps, des_v_mins, des_v_maxs = [], [], []
        for i_app in i_apps[::-1]:
            is_cycle, v_min_env, v_max_env, last_v, last_n = self.cycle_envelope(i_app, delta_t, 
                                                                     no_timesteps,
                                                                     v_init, n_init, amp_threshold)
            v_init = last_v
            n_init = last_n
            if is_cycle:
                des_i_apps.append(i_app)
                des_v_mins.append(v_min_env)
                des_v_maxs.append(v_max_env)
        des_iapps_vmin_vmax = np.array([des_i_apps[::-1],
                                        des_v_mins[::-1],
                                        des_v_maxs[::-1]])

        return {
            'equilibria_i_v' : eq_i_v,
            'asc_pass_iapps_vmin_vmax' : asc_iapps_vmin_vmax,
            'des_pass_iapps_vmin_vmax' : des_iapps_vmin_vmax
        }     
