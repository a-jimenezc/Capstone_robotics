from src.estimate_ekf_base import EstimateEkfBase
import numpy as np

class EstimateEkfJoint(EstimateEkfBase):
    def __init__(self):
        super().__init__()
        self.f_morris_euler = self.model.f_morris_euler_ekf_joint

    def A_k(self, state, parameters, delta_t):
        '''
        A_k: 10x10 matrix
        Derivatives calculated by hand
        '''
        p = parameters # for fixed parameters only
        [v, n, g_l, g_k, g_ca, phi, v_1, v_2, v_3, v_4] = state

        # (1): voltage equation morris-lecar model
        # a_1: d(1)/dv
        tanh_1 = np.tanh((v - v_1) / v_2)
        temp_1 = ((1-tanh_1**2)*(1/v_2) * (v - p.E_CA) + (1+tanh_1))
        a_1 = 1 + (delta_t/p.CAPACITANCE) * (-g_l - g_k*n - 0.5*g_ca*temp_1)

        # a_2: d(1)/dn
        a_2 = (delta_t/p.CAPACITANCE) * (-g_k * (v - p.E_K))

        # a_3: d(1)/dg_l
        a_3 = (delta_t/p.CAPACITANCE) * (-(v - p.E_L))

        # a_4: d(1)/dg_k
        a_4 = (delta_t/p.CAPACITANCE) * (-n * (v - p.E_K))

        # a_5: d(1)/dg_ca
        tanh_1 = np.tanh((v - v_1) / v_2)
        a_5 = (delta_t/p.CAPACITANCE) * (-0.5) * (1+tanh_1) * (v - p.E_CA)

        # a_6: d(1)/dphi
        a_6 = 0.0

        # a_7: d(1)/dv_1
        temp_2 = -0.5 * g_ca * (1-tanh_1**2) * (-1/v_2) * (v - p.E_CA)
        a_7 = (delta_t/p.CAPACITANCE) * temp_2

        # a_8: d(1)/dv_2
        temp_3 = -0.5 * g_ca * (1-tanh_1**2) * (-((v-v_1)/v_2**2)) * (v - p.E_CA)
        a_8 = (delta_t/p.CAPACITANCE) * temp_3

        # a_9: d(1)/dv_3
        a_9 = 0.0

        # a_10: d(1)/dv_4
        a_10 = 0.0

        a = np.array([a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_10])

        # (2): n equation morris-lecar model
        # b_1: d(2)/dv
        sinh = np.sinh((v - v_3) / (2*v_4))
        cosh = np.cosh((v - v_3) / (2*v_4))
        tanh_2 = np.tanh((v - v_3) / (v_4))
        temp_4 = (0.5/v_4) * sinh * phi * (0.5*(1+tanh_2) - n)
        temp_5 = cosh * phi * 0.5 * (1-tanh_2**2) * (1/v_4)
        b_1 = delta_t * (temp_4 + temp_5)

        # b_2: d(2)/dn
        b_2 = 1 + delta_t * (-cosh * phi)

        # b_3: d(2)/dg_l
        b_3 = 0.0

        # b_4: d(2)/dg_k
        b_4 = 0.0

        # b_5: d(2)/dg_ca
        b_5 = 0.0

        # b_6: d(2)/dphi
        b_6 = delta_t * cosh * (0.5 * (1+tanh_2) - n)

        # b_7: d(2)/dv_1
        b_7 = 0.0

        # b_8: d(2)/dv_2
        b_8 = 0.0

        # b_9: d(2)/dv_3
        temp_6= (-0.5/v_4) * sinh * phi * (0.5*(1+tanh_2) - n)
        temp_7 = cosh * phi * 0.5 * (1-tanh_2**2) * (-1/v_4)
        b_9 = delta_t * (temp_6 + temp_7)

        # b_10: d(2)/dv_4
        temp_8 = (-0.5*((v-v_3)/v_4**2)) * sinh * phi * (0.5*(1+tanh_2) - n)
        temp_9 = cosh * phi * 0.5 * (1-tanh_2**2) * (-1.0*((v-v_3)/v_4**2))
        b_10 = delta_t * (temp_8 + temp_9)

        b = np.array([b_1, b_2, b_3, b_4, b_5, b_6, b_7, b_8, b_9, b_10])

        # Creating the matrix ('parameter' rows are just identity)
        A_k = np.eye(10)
        A_k[0, :] = a
        A_k[1, :] = b

        return A_k
    
    def C_k(self):
        C_k = np.zeros((1, 10))
        C_k[0, 0] = 1.0
        return C_k

    def D_k(self):
        return np.array([[1.0]])
    