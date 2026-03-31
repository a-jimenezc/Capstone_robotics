class Parameters:
    '''Parameters class used throughout the project'''
    def __init__(self, fixed_parameters, parameters):
        self.CAPACITANCE = fixed_parameters[0]
        self.E_CA = fixed_parameters[1]
        self.E_K = fixed_parameters[2]
        self.E_L = fixed_parameters[3]

        self.g_l = parameters[0]
        self.g_k = parameters[1]
        self.g_ca = parameters[2]
        self.phi = parameters[3]
        self.v_1 = parameters[4]
        self.v_2 = parameters[5]
        self.v_3 = parameters[6]
        self.v_4 = parameters[7]