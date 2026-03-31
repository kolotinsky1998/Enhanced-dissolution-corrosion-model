import numpy as np
from scipy.optimize import fsolve
import sys
#sys.path.append('/home/common/kolotinskiy.da/steel/thermodynamics/corrosionthermo/')
sys.path.append('./data')
from D_Fe_Pb_Robertson import D_Fe_Pb_coefficients 
from D_Fe_LBE import D_Fe_LBE_coefficients
###################
# Basic constants #
###################

R = 8.31446261815324 # gas constant [J/K/mol]
N_A = 6.022e23 # Avogadro constant [1/mol]
m_O = 15.999e-3 # Oxygen molar mass [kg/mol]
m_Fe = 55.845e-3 # Iron molar mass [kg/mol]
m_Cr = 51.9961e-3 # Chromium molar mass [kg/mol]
m_Pb = 207.2e-3 # Lead molar mass [kg/mol]
m_Bi = 208.9804e-3 # Biswuth molar mass [kg/mol]
chi_Bi = 0.55 # Bismuth molar fraction in LBE
chi_Pb = 0.45 # Lead molar fraction in LBE

##############################
# Chemical elements database #
##############################

class Fe3O4_s:
    def __init__(self):
        self.T_I = 298.15
        self.T_II = 900.0

        self.A = 104.2096
        self.B = 178.5108
        self.C = 10.61510
        self.D = 1.132534
        self.E = -0.994202
        self.F = -1163.336
        self.G = 212.0585
        self.H = -1120.894

    def H_f(self):
        return -1120.89*1000
    
    def H_m(self, T):
        if T >= self.T_I and T < self.T_II:
            t = T/1000.0
            H_m = self.A*t + self.B*t**2/2.0 + self.C*t**3/3.0 + self.D*t**4/4.0 - self.E/t + self.F - self.H
            return 1000*H_m+self.H_f()
        else:
            print("Temperature is out of the range where this state exists")

    def S_m(self, T):
        if T >= self.T_I and T < self.T_II:
            t = T/1000.0
            S_m = self.A*np.log(t) + self.B*t + self.C*t**2/2.0 + self.D*t**3/3.0 - self.E/(2*t**2) + self.G
            return S_m
        else:
            print("Temperature is out of the range where this state exists")
    
    def G_m(self, T):
        if T >= self.T_I and T < self.T_II:
            G_m = self.H_m(T) - self.S_m(T)*T
            return G_m
        else:
            print("Temperature is out of the range where this state exists")
    def rho(self):
        rho_Fe3O4 = 5175.0
        return rho_Fe3O4

class Pb_s:
    def __init__(self):
        self.T_melting = 600.6
        self.A = 25.01450
        self.B = 5.441836
        self.C = 4.061367
        self.D = -1.236214
        self.E = -0.010657
        self.F = -7.772575
        self.G = 93.19902
        self.H = 0.000000

    def H_f(self):
        return 0
    
    def H_m(self, T):
        if T < self.T_melting:
            t = T/1000.0
            H_m = self.A*t + self.B*t**2/2.0 + self.C*t**3/3.0 + self.D*t**4/4.0 - self.E/t + self.F - self.H
            return 1000*H_m+self.H_f()
        else:
            print("Temperature is out of the range where this state exists")

    def S_m(self, T):
        if T < self.T_melting:
            t = T/1000.0
            S_m = self.A*np.log(t) + self.B*t + self.C*t**2/2.0 + self.D*t**3/3.0 - self.E/(2*t**2) + self.G
            return S_m
        else:
            print("Temperature is out of the range where this state exists")
    
    def G_m(self, T):
        if T < self.T_melting:
            G_m = self.H_m(T) - self.S_m(T)*T
            return G_m
        else:
            print("Temperature is out of the range where this state exists")

class Pb_l:
    def __init__(self):
        self.T_melting = 600.6
        self.A = 38.00449
        self.B = -14.62249
        self.C = 7.255475
        self.D = -1.033370
        self.E = -0.330775
        self.F = -7.944328
        self.G = 118.7992
        self.H = 4.282993

    def H_f(self):
        return 4.28*1000
    
    def H_m(self, T):
        if T >= self.T_melting:
            t = T/1000.0
            H_m = self.A*t + self.B*t**2/2.0 + self.C*t**3/3.0 + self.D*t**4/4.0 - self.E/t + self.F - self.H
            return 1000*H_m+self.H_f()
        else:
            print("Temperature is out of the range where this state exists")

    def S_m(self, T):
        if T >= self.T_melting:
            t = T/1000.0
            S_m = self.A*np.log(t) + self.B*t + self.C*t**2/2.0 + self.D*t**3/3.0 - self.E/(2*t**2) + self.G
            return S_m
        else:
            print("Temperature is out of the range where this state exists")
    
    def G_m(self, T):
        if T >= self.T_melting:
            G_m = self.H_m(T) - self.S_m(T)*T
            return G_m
        else:
            print("Temperature is out of the range where this state exists")
    
    def rho(self, T):
        rho_Pb_l = 11441 - 1.2795*T
        return rho_Pb_l

class Fe_alpha_delta_phase_s:
    def __init__(self):
    
        self.T_I = 298.0
        self.T_I_II = 700.0
        self.T_II = 1042

        self.A_I = 18.42868
        self.B_I = 24.64301
        self.C_I = -8.913720
        self.D_I = 9.664706
        self.E_I = -0.012643
        self.F_I = -6.573022
        self.G_I = 42.51488
        self.H_I = 0.000000

        self.A_II = -57767.65
        self.B_II = 137919.7
        self.C_II = -122773.2
        self.D_II = 38682.42
        self.E_II = 3993.080
        self.F_II = 24078.67
        self.G_II = -87364.01
        self.H_II = 0.000000

    def H_f(self):
        return 0
    
    def H_m(self, T):
        t = T/1000.0
        if T >= self.T_I and T < self.T_I_II:
            H_m = self.A_I*t + self.B_I*t**2/2.0 + self.C_I*t**3/3.0 + self.D_I*t**4/4.0 - self.E_I/t + self.F_I - self.H_I
            return 1000*H_m+self.H_f()
        elif T >= self.T_I_II and T < self.T_II:
            H_m = self.A_II*t + self.B_II*t**2/2.0 + self.C_II*t**3/3.0 + self.D_II*t**4/4.0 - self.E_II/t + self.F_II - self.H_II
            return 1000*H_m+self.H_f()
        else:
            print("Temperature is out of the range where this state exists")

    def S_m(self, T):
        t = T/1000.0
        if T >= self.T_I and T < self.T_I_II:
            S_m = self.A_I*np.log(t) + self.B_I*t + self.C_I*t**2/2.0 + self.D_I*t**3/3.0 - self.E_I/(2*t**2) + self.G_I
            return S_m
        elif T >= self.T_I_II and T < self.T_II:
            S_m = self.A_II*np.log(t) + self.B_II*t + self.C_II*t**2/2.0 + self.D_II*t**3/3.0 - self.E_II/(2*t**2) + self.G_II
            return S_m
        else:
            print("Temperature is out of the range where this state exists")
    
    def G_m(self, T):
        if T >= self.T_I and T < self.T_II:
            G_m = self.H_m(T) - self.S_m(T)*T
            return G_m
        else:
            print("Temperature is out of the range where this state exists")

class PbO_yellow_phase_s:
    def __init__(self):
        self.T_I = 298.0
        self.T_I_II = 762.0
        self.T_II = 1159.0

        self.A_I = 7.465570
        self.B_I = 179.5860
        self.C_I = -233.5490
        self.D_I = 109.2070
        self.E_I = 0.233832
        self.F_I = -226.9830
        self.G_I = 32.54460
        self.H_I = -219.4090

        self.A_II = 47.86340
        self.B_II = 12.55480
        self.C_II = -0.001810
        self.D_II = 0.000416
        self.E_II = 0.000200
        self.F_II = -234.8160
        self.G_II = 118.9100
        self.H_II = -219.4090

    def H_f(self):
        return -219.41*1000
    
    def H_m(self, T):
        t = T/1000.0
        if T >= self.T_I and T < self.T_I_II:
            H_m = self.A_I*t + self.B_I*t**2/2.0 + self.C_I*t**3/3.0 + self.D_I*t**4/4.0 - self.E_I/t + self.F_I - self.H_I
            return 1000*H_m+self.H_f()
        elif T >= self.T_I_II and T < self.T_II:
            H_m = self.A_II*t + self.B_II*t**2/2.0 + self.C_II*t**3/3.0 + self.D_II*t**4/4.0 - self.E_II/t + self.F_II - self.H_II
            return 1000*H_m+self.H_f()
        else:
            print("Temperature is out of the range where this state exists")

    def S_m(self, T):
        t = T/1000.0
        if T >= self.T_I and T < self.T_I_II:
            S_m = self.A_I*np.log(t) + self.B_I*t + self.C_I*t**2/2.0 + self.D_I*t**3/3.0 - self.E_I/(2*t**2) + self.G_I
            return S_m
        elif T >= self.T_I_II and T < self.T_II:
            S_m = self.A_II*np.log(t) + self.B_II*t + self.C_II*t**2/2.0 + self.D_II*t**3/3.0 - self.E_II/(2*t**2) + self.G_II
            return S_m
        else:
            print("Temperature is out of the range where this state exists")
    
    def G_m(self, T):
        if T >= self.T_I and T < self.T_II:
            G_m = self.H_m(T) - self.S_m(T)*T
            return G_m
        else:
            print("Temperature is out of the range where this state exists")

class O2_g:
    def __init__(self):
        self.T_100 = 100.0
        self.T_700 = 700.0
        self.T_2000 = 2000.0

        self.A_T_100_700 = 31.32234
        self.B_T_100_700 = -20.23531
        self.C_T_100_700= 57.86644
        self.D_T_100_700 = -36.50624
        self.E_T_100_700 = -0.007374
        self.F_T_100_700 = -8.903471
        self.G_T_100_700 = 246.7945
        self.H_T_100_700 = 0.0

        self.A_T_700_2000 = 30.03235
        self.B_T_700_2000 = 8.772972
        self.C_T_700_2000 = -3.988133
        self.D_T_700_2000 = 0.788313
        self.E_T_700_2000 = -0.741599
        self.F_T_700_2000 = -11.32468
        self.G_T_700_2000 = 236.1663
        self.H_T_700_2000 = 0.0

    def H_f(self):
        return 0
    
    def H_m(self, T):
        t = T/1000.0
        if T >= self.T_100 and T < self.T_700:
            H_m = self.A_T_100_700*t + self.B_T_100_700*t**2/2.0 + self.C_T_100_700*t**3/3.0 + self.D_T_100_700*t**4/4.0 - self.E_T_100_700/t + self.F_T_100_700 - self.H_T_100_700
            return 1000*H_m+self.H_f()
        elif T >= self.T_700 and T < self.T_2000:
            H_m = self.A_T_700_2000*t + self.B_T_700_2000*t**2/2.0 + self.C_T_700_2000*t**3/3.0 + self.D_T_700_2000*t**4/4.0 - self.E_T_700_2000/t + self.F_T_700_2000 - self.H_T_700_2000
            return 1000*H_m+self.H_f()
        else:
            print("Temperature is out of the range where this state exists")

    def S_m(self, T):
        t = T/1000.0
        if T >= self.T_100 and T < self.T_700:
            S_m = self.A_T_100_700*np.log(t) + self.B_T_100_700*t + self.C_T_100_700*t**2/2.0 + self.D_T_100_700*t**3/3.0 - self.E_T_100_700/(2*t**2) + self.G_T_100_700
            return S_m
        elif T >= self.T_700 and T < self.T_2000:
            S_m = self.A_T_700_2000*np.log(t) + self.B_T_700_2000*t + self.C_T_700_2000*t**2/2.0 + self.D_T_700_2000*t**3/3.0 - self.E_T_700_2000/(2*t**2) + self.G_T_700_2000
            return S_m
        else:
            print("Temperature is out of the range where this state exists")
    
    def G_m(self, T):
        if T >= self.T_100 and T < self.T_2000:
            G_m = self.H_m(T) - self.S_m(T)*T
            return G_m
        else:
            print("Temperature is out of the range where this state exists")

class Cr_s:
    def __init__(self):
        self.T_melting = 600
        self.A = 18.46508
        self.B = 5.477986
        self.C = 7.904329
        self.D = -1.147848
        self.E = 1.265791
        self.F = -2.676941
        self.G = 48.09341
        self.H = 0.000000
        self.T_I = 600
        self.T_II = 2130

    def H_f(self):
        return 0
    
    def H_m(self, T):
        t = T/1000.0
        if T >= self.T_I and T < self.T_II:
            H_m = self.A*t + self.B*t**2/2.0 + self.C*t**3/3.0 + self.D*t**4/4.0 - self.E/t + self.F - self.H
            return 1000*H_m+self.H_f()
        else:
            print("Temperature is out of the range where this state exists")

    def S_m(self, T):
        t = T/1000.0
        if T >= self.T_I and T < self.T_II:
            S_m = self.A*np.log(t) + self.B*t + self.C*t**2/2.0 + self.D*t**3/3.0 - self.E/(2*t**2) + self.G
            return S_m
        else:
            print("Temperature is out of the range where this state exists")
    
    def G_m(self, T):
        if T >= self.T_I and T < self.T_II:
            G_m = self.H_m(T) - self.S_m(T)*T
            return G_m
        else:
            print("Temperature is out of the range where this state exists")

class Cr2O3_s:
    def __init__(self):
        self.T_I = 305
        self.T_II = 2603

        self.A = 124.6550
        self.B = -0.337045
        self.C = 5.705010
        self.D = -1.053470
        self.E = -2.030501
        self.F = -1178.440
        self.G = 221.3300
        self.H = -1134.700

    
    def H_f(self):
        return -1134.70*1000 
    
    def H_magnetic(self):
        return 659.96
    
    def S_f(self):
        return -272.511
    
    def G_f(self):
        return -840.58*1000
    
    def H_m(self, T):
        if T >= self.T_I and T < self.T_II:
            t = T/1000.0
            H_m = self.A*t + self.B*t**2/2.0 + self.C*t**3/3.0 + self.D*t**4/4.0 - self.E/t + self.F - self.H
            return 1000*H_m+self.H_f()
        else:
            print("Temperature is out of the range where this state exists")

    def S_m(self, T):
        if T >= self.T_I and T < self.T_II:
            t = T/1000.0
            S_m = self.A*np.log(t) + self.B*t + self.C*t**2/2.0 + self.D*t**3/3.0 - self.E/(2*t**2) + self.G
            return S_m
        else:
            print("Temperature is out of the range where this state exists")
    
    def G_m(self, T):
        if T >= self.T_I and T < self.T_II:
            G_m = self.H_m(T) - self.S_m(T)*T
            return G_m
        else:
            print("Temperature is out of the range where this state exists")

class FeCr2O4_s:
    def __init__(self):
        self.T_melting = 298
        self.a = -27.894
        self.b = 0.02113
        self.c = -7.9344e-06
        self.d = 454.211
        self.e = -9.43604e+05
        self.f = -200.285
        self.g = 11403.1
        self.h = 0.000000
        self.T_I = 298
        self.T_II = 1185
        self.C = -651.27*1000
        self.D = 0.1495*1000

    def H_f(self):
        return -1438.52*1000
    
    def S_f(self):
        return -59.955
    
    def G_f(self):
        return -214.16*1000
    
    def G_m(self, T):
        if T >= self.T_I and T < self.T_II:
            cr2o3 = Cr2O3_s()
            o2 = O2_g()
            Fe = Fe_alpha_delta_phase_s()
            G_m = (1/2.0) * (self.C + self.D*T) + cr2o3.G_m(T) + 1/2.0 * o2.G_m(T) + Fe.G_m(T)
            return G_m
        else:
            print("Temperature is out of the range where this state exists")

    def rho(self):
        rho_FeCr2O4 = 4500.0
        return rho_FeCr2O4




#######################
# LBE characteristics #
#######################

def rho_LBE(T):
    rho_LBE = 11065.0 - 1.293*T # [kg/m^3]
    return rho_LBE

def nu_LBE(T):
    nu = 4.94e-4*np.exp(754.1/T)/rho_LBE(T)
    return nu

####################################################
# Elements saturation concentration in liquid lead #
####################################################

def C_O_s(T):
    '''Schroer review, in (33) is log10'''
    pb_l = Pb_l()
    C_O_s = 10**(3.21-5100/T)/100.0/m_O*pb_l.rho(T)
    return C_O_s

def C_Fe_s(T):
    '''Schroer review, in (37) is log10'''
    pb_l = Pb_l()
    C_Fe_s = 10.0**(1.824-4860/T)/100.0/m_Fe*pb_l.rho(T)
    return C_Fe_s

####################################################
# Elements saturation concentration in liquid LBE  #
####################################################

def C_O_s_LBE(T):
    ''' Schroer review T < 800 C, in (52) is log10'''
    C_O_s = 10.0**(2.62-4416.0/T)/100.0/m_O*rho_LBE(T) # [mol/m^3]
    return C_O_s

def C_Fe_s_LBE(T):
    ''' Schroer review, in (56) is log10'''
    C_Fe_s = 10.0**(2.012-4382.0/T)/100.0/m_Fe*rho_LBE(T) # [mol/m^3]
    return C_Fe_s

def C_Cr_s_LBE(T):
    ''' Schroer review, in (57) is log10'''
    C_Cr_s = 10.0**(-0.02-2280.0/T)/100.0/m_Cr*rho_LBE(T) # [mol/m^3]
    return C_Cr_s


##############################################
# Elements chemical potential in liquid lead #
##############################################

def mu_O(C_O, C_Fe, T):
    C_s = C_O_s(T)
    pb_l = Pb_l()
    pbo_yellow_phase_s = PbO_yellow_phase_s()
    mu_O = pbo_yellow_phase_s.G_m(T) - pb_l.G_m(T) + R*T*np.log(np.sqrt(C_O/C_s))
    return mu_O

def mu_Fe(C_O, C_Fe, T):
    C_s = C_Fe_s(T)
    fe_alpha_delta_phase_s = Fe_alpha_delta_phase_s()
    mu_Fe = fe_alpha_delta_phase_s.G_m(T) + R*T*np.log(C_Fe/C_s)
    return mu_Fe

def delta_G_m_Fe3O4_s(C_Fe, C_O, T):
    fe3o4_s = Fe3O4_s()
    delta_G_m_Fe3O4_s = fe3o4_s.G_m(T) - 4*mu_O(C_O, C_Fe[0], T) - 3*mu_Fe(C_O, C_Fe[0], T)
    return delta_G_m_Fe3O4_s

def drivingForce_Fe3O4_s(C_Fe, C_O, T):
    fe3o4_s = Fe3O4_s()
    drivingForce = (4*mu_O(C_O, C_Fe, T) + 3*mu_Fe(C_O, C_Fe, T) - fe3o4_s.G_m(T))/(R*T)
    return drivingForce

def drivingForce_Fe_alpha_delta_phase_s(C_Fe, C_O, T):
    drivingForce = np.log(C_Fe/C_Fe_s(T))
    return drivingForce

def drivingForce_PbO_yellow_phase_s(C_Fe, C_O, T):
    drivingForce = np.log(C_O/C_O_s(T))
    return drivingForce

#############################################
# Elements chemical potential in liquid LBE #
#############################################

def mu_Pb_LBE(T):
    '''Schroer review, equation (50)'''
    pb_l = Pb_l()
    mu_Pb = pb_l.G_m(T) + R*T*(-0.82912-166.80/T)
    return mu_Pb

def mu_O_LBE(C_O, C_Fe, T):
    C_s = C_O_s_LBE(T)
    pbo_yellow_phase_s = PbO_yellow_phase_s()
    mu_O = pbo_yellow_phase_s.G_m(T) - mu_Pb_LBE(T) + R*T*np.log(np.sqrt(C_O/C_s))
    return mu_O

def mu_Fe_LBE(C_O, C_Fe, T):
    C_s = C_Fe_s_LBE(T)
    fe_alpha_delta_phase_s = Fe_alpha_delta_phase_s()
    mu_Fe = fe_alpha_delta_phase_s.G_m(T) + R*T*np.log(C_Fe/C_s)
    return mu_Fe

def mu_Cr_LBE(C_Cr, T):
    C_s = C_Cr_s_LBE(T)
    cr_S = Cr_s()
    mu_Cr = cr_S.G_m(T) + R*T*np.log(C_Cr/C_s)
    return mu_Cr


##################################################
# Elements diffusion coefficients in liquid lead #
##################################################

def D_O_Pb(T):
    '''Ganesan (2006b)'''
    D_O = 1.0e-4*2.79e-3*np.exp(-45587/(R*T))
    return D_O

def D_Fe_Pb(T):
    a, b = D_Fe_Pb_coefficients()
    D_Fe = 1.0e-4*np.exp(a/T+b) 
    return D_Fe

##################################################
# Elements diffusion coefficients in liquid LBE  #
##################################################

def D_O_LBE(T):
    '''Ganesan (2006b)'''
    D_O = 1e-4*2.39e-2*np.exp(-43073/(R*T)) # m^2/s
    return D_O

def D_Fe_LBE(T):
    a, b = D_Fe_LBE_coefficients()
    D_Fe = 1.0e-4*np.exp(a/T+b)
    return D_Fe

def D_Cr_LBE(T):
    a, b = D_Fe_LBE_coefficients()
    D_Cr = 1.0e-4*np.exp(a/T+b)
    return D_Cr
