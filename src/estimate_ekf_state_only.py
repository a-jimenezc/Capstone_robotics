from src.estimate_ekf_base import EstimateEkfBase
import numpy as np

class EstimateEkfStateOnly(EstimateEkfBase):
    def __init__(self):
        super().__init__()
        self.f_morris_euler = self.model.f_morris_euler

    def A_k(self, state, parameters, delta_t):
        '''
        A_k = [[a, b],[c, d]]
        Derivatives calculated by hand
        '''
        p = parameters
        v = state[0]
        n = state[1]

        # a
        tanh_1 = np.tanh((v - p.v_1) / p.v_2)
        temp_1 = ((1 - tanh_1**2)*(1/p.v_2) * (v - p.E_CA) + (1 + tanh_1))
        a = 1 + (delta_t/p.CAPACITANCE) * (-p.g_l - p.g_k*n - 0.5*p.g_ca*temp_1)

        # b
        b = (delta_t/p.CAPACITANCE) * (-p.g_k * (v - p.E_K))

        # c
        sinh = np.sinh((v - p.v_3) / (2*p.v_4))
        cosh = np.cosh((v - p.v_3) / (2*p.v_4))
        tanh_2 = np.tanh((v - p.v_3) / (p.v_4))
        temp_2 = (0.5/p.v_4) * sinh * p.phi * (0.5*(1+tanh_2) - n)
        temp_3 = cosh * p.phi * 0.5 * (1-tanh_2**2) * (1/p.v_4)
        c = delta_t * (temp_2 + temp_3)

        # d
        d = 1 + delta_t * (-cosh * p.phi)

        A_k = np.array([[a, b], [c, d]])

        return A_k
    
    def C_k(self):
        return np.array([[1.0, 0.0]])

    def D_k(self):
        return np.array([[1.0]])
    