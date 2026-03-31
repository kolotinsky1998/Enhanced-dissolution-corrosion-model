from scipy.optimize import fsolve
import numpy as np
import sys
sys.path.append('src/corrosionthermo/')
from corrosionthermo import Fe3O4_s
from corrosionthermo import PbO_yellow_phase_s
from corrosionthermo import Fe_alpha_delta_phase_s
from corrosionthermo import Cr_s
from corrosionthermo import mu_Pb_LBE
from corrosionthermo import R
from corrosionthermo import C_O_s_LBE
from corrosionthermo import C_Fe_s_LBE
from corrosionthermo import C_Cr_s_LBE
from corrosionthermo import D_O_LBE
from corrosionthermo import D_Fe_LBE
from corrosionthermo import D_Cr_LBE
from corrosionthermo import m_O
from corrosionthermo import m_Fe
from corrosionthermo import m_Cr
from corrosionthermo import rho_LBE
from corrosionthermo import Pb_l
from corrosionthermo import nu_LBE
from corrosionthermo import FeCr2O4_s
from Cr2O3_dissolution_estimate import Cr2O3_sediment_check
from rich.console import Console
import matplotlib.pyplot as plt


def red_console_massage(text):
    console = Console()
    console.print(text, style="red on white")

def print_state(**variables):
    red_console_massage("")
    red_console_massage("=======================")
    for name, value in variables.items():
        red_console_massage(f"{name}: {value}")
        red_console_massage("=======================")
    red_console_massage("")


class Dissolution:
    def __init__(self, C_Fe_bulk, C_Cr_bulk, C_O_bulk, T, K):
        
        self.a_C = K**(1/7)
        self.a_D = 1e-7
        
        self.C_O_bulk = C_O_bulk/self.a_C
        self.C_Fe_bulk = C_Fe_bulk/self.a_C
        self.C_Cr_bulk = C_Cr_bulk/self.a_C
        
        self.D_O_LBE = D_O_LBE(T)/self.a_D
        self.D_Fe_LBE = D_Fe_LBE(T)/self.a_D
        self.D_Cr_LBE = D_Cr_LBE(T)/self.a_D
        
        self.K = K/(self.a_C**7.0)
 
    def printDimensionlessParameters(self):
        print("C_O_bulk = {}".format(self.C_O_bulk))
        print("C_Fe_bulk = {}".format(self.C_Fe_bulk))
        print("C_Cr_bulk = {}".format(self.C_Cr_bulk))
        print("D_Fe = {}".format(self.D_Fe_LBE))
        print("D_Cr= {}".format(self.D_Cr_LBE))
        print("D_O = {}".format(self.D_O_LBE))
        print("K = {}".format(self.K))

    def boundaryProblem(self, array, C_start, multiplyer):
        def equations(p):
            C_Fe, C_Cr, C_O = p
            eq_1 = 1.0 * (self.C_Fe_bulk - C_Fe) * (self.D_Fe_LBE*multiplyer[0])**(0.66) - 4.0*(self.C_O_bulk - C_O) * (self.D_O_LBE*multiplyer[2])**(0.66)
            eq_2 = 1.0 * (self.C_Fe_bulk - C_Fe) * (self.D_Fe_LBE*multiplyer[0])**(0.66) - 2.0*(self.C_Cr_bulk - C_Cr) * (self.D_Cr_LBE*multiplyer[1])**(0.66)
            eq_3 = (C_Fe**1)*(C_Cr**2.0)*(C_O**4.0) - self.K
            return (eq_1, eq_2, eq_3)
        C_Fe, C_Cr, C_O = fsolve(equations, C_start, maxfev=100)
        return C_Fe*self.a_C, C_Cr*self.a_C, C_O*self.a_C


def FeCr2O4_dissolution_with_D_multiplyer():
    T = 600 + 273.15
    fe3O4_s = Fe3O4_s()
    feCr2O4 = FeCr2O4_s()
    pbO_s = PbO_yellow_phase_s()
    fe_s = Fe_alpha_delta_phase_s()
    pb_l = Pb_l()

    C_O_bulk_mass_percent = 0 # mass, %
    C_Cr_bulk_mass_percent = 0 # mass, %
    C_Fe_bulk_mass_percent = 0 # mass, %
    C_O_bulk = C_O_bulk_mass_percent/100.0/m_O*rho_LBE(T)
    C_Cr_bulk = C_Cr_bulk_mass_percent/100.0/m_Cr*rho_LBE(T)
    C_Fe_bulk = C_Fe_bulk_mass_percent/100.0/m_Fe*rho_LBE(T)
    K_Fe3O4 = np.exp((fe3O4_s.G_m(T)-4.0*(pbO_s.G_m(T)-mu_Pb_LBE(T))-3.0*fe_s.G_m(T))/(R*T))*C_O_s_LBE(T)**4.0*C_Fe_s_LBE(T)**3.0
    #C_Fe_bulk = (K_Fe3O4/((1e-6/100.0/m_O*rho_LBE(T))**4.0))**(1.0/3.0)
    K = np.exp((feCr2O4.G_m(T) - 1*fe_s.G_m(T) - 2*Cr_s().G_m(T) - 4*(pbO_s.G_m(T) - mu_Pb_LBE(T)))/ (R*T))*(C_Fe_s_LBE(T)**1.0)*(C_Cr_s_LBE(T)**2.0)*(C_O_s_LBE(T)**4.0)
    n = 200
    N = 3
    V = np.linspace(0.5,2.5,n)
    d = 0.01

    C_Cr_bulk
    data = []
    x = [10**i for i in range(-1, 3)]
    #x = [10**i for i in range(0, 1)]
    multiplyerS = [[x0, y0, z0] for x0 in x for y0 in x for z0 in x]
    
    #multiplyerS = [[1, 1, 1]]
    N=len(multiplyerS)
    
    for multiplyer in multiplyerS:
        dissolution = Dissolution(C_Fe_bulk, C_Cr_bulk, C_O_bulk,  T, K)
        l_diff = 60.60*V**(-0.86)*d**(0.14)*nu_LBE(T)**(0.53)*(D_Cr_LBE(T)*multiplyer[1])**(0.33)
        #dissolution.printDimensionlessParameters()
        #C_Fe, C_Cr, C_O = dissolution.boundaryProblem([C_Fe_bulk, C_Cr_bulk, C_O_bulk], [2.6e-6, 1e-6, 2.5e5], multiplyer)
        #beta = (K/(C_O_bulk)**4/C_Fe_bulk)**(1.0/2.0)
        #C_Fe, C_Cr, C_O = dissolution.boundaryProblem([C_Fe_bulk, C_Cr_bulk, C_O_bulk], [C_Fe_bulk/dissolution.a_C, (beta)/dissolution.a_C, (C_O_bulk)/dissolution.a_C], multiplyer)
        C_Fe, C_Cr, C_O = dissolution.boundaryProblem([C_Fe_bulk, C_Cr_bulk, C_O_bulk], [1, 1, 1], multiplyer)

        if C_Fe>C_Fe_s_LBE(T):
            red_console_massage("Fe is oversaturated")
        if C_Cr>C_Cr_s_LBE(T):
            red_console_massage("Cr is oversaturated")
        if C_O>C_O_s_LBE(T):
            red_console_massage("O is oversaturated")
        #if Cr2O3_sediment_check(C_Cr, C_O, T): red_console_massage("Cr2O3 precipitates")
        #else: red_console_massage("Cr2O3 not precipitates")
        Omega_FeCr2O4 = (1.0*m_Fe+2.0*m_Cr+4.0*m_O)/feCr2O4.rho()
        #print_state(C_Fe=C_Fe, C_Cr=C_Cr, C_O=C_O, C_Fe_bulk=C_Fe_bulk, 
        #            C_Cr_bulk=C_Cr_bulk, C_O_bulk=C_O_bulk, delta_C_Cr=C_Cr_bulk-C_Cr, 
        #            J_multiplied_by_l_diff = Omega_FeCr2O4*(D_Cr_LBE(T)*multiplyer[1])/2.0*(C_Cr_bulk-C_Cr))
        data.append(3600*8760*1e6*Omega_FeCr2O4*(D_Cr_LBE(T)*multiplyer[1])/2.0*(C_Cr_bulk-C_Cr)/l_diff)



    file = open("txt_data_files/variation_of_parameters.txt",'w')
    file.write("V|")
    for p in range(N):
        if p != N-1:
            file.write("| multiplyer={} |".format(multiplyerS[p]))
        else:
            file.write("| multiplyer={} \n ".format(multiplyerS[p]))
    for i in range(n):
        file.write("{}\t".format(V[i]))
        for p in range(N):
            if p != N-1:
                file.write("{}\t".format(data[p][i]))
            else:
                file.write("{}\n".format(data[p][i]))
    file.close()

def FeCr2O4_dissolution_with_different_T():
    T_cel = [400, 450, 500, 550, 600]
    T_Kelvin = list(map(lambda x: x+273.15, T_cel))
    fe3O4_s = Fe3O4_s()
    feCr2O4 = FeCr2O4_s()
    pbO_s = PbO_yellow_phase_s()
    fe_s = Fe_alpha_delta_phase_s()
    pb_l = Pb_l()

    C_O_bulk_mass_percent = 1e-6 # mass, %
    C_Cr_bulk_mass_percent = 0 # mass, %
    C_Fe_bulk_mass_percent = 0 # mass, %


    n = 200
    N = 3
    V = np.linspace(0.5,2.5,n)
    d = 0.01

    data = []
    N = len(T_Kelvin)
    for T in T_Kelvin:
        C_O_bulk = C_O_bulk_mass_percent/100.0/m_O*rho_LBE(T)
        C_Cr_bulk = C_Cr_bulk_mass_percent/100.0/m_Cr*rho_LBE(T)
        C_Fe_bulk = C_Fe_bulk_mass_percent/100.0/m_Fe*rho_LBE(T)
        K_Fe3O4 = np.exp((fe3O4_s.G_m(T)-4.0*(pbO_s.G_m(T)-mu_Pb_LBE(T))-3.0*fe_s.G_m(T))/(R*T))*C_O_s_LBE(T)**4.0*C_Fe_s_LBE(T)**3.0
        C_Fe_bulk = (K_Fe3O4/((1e-6/100.0/m_O*rho_LBE(T)))**4.0)**(1.0/3.0)
        print(C_Fe_bulk*100*m_Fe/rho_LBE(T))
        K = np.exp((feCr2O4.G_m(T) - 1*fe_s.G_m(T) - 2*Cr_s().G_m(T) - 4*(pbO_s.G_m(T) - mu_Pb_LBE(T)))/ (R*T))*(C_Fe_s_LBE(T)**1.0)*(C_Cr_s_LBE(T)**2.0)*(C_O_s_LBE(T)**4.0)
        red_console_massage(f"K: {K}")
        dissolution = Dissolution(C_Fe_bulk, C_Cr_bulk, C_O_bulk,  T, K)
        l_diff = 60.60*V**(-0.86)*d**(0.14)*nu_LBE(T)**(0.53)*(D_Cr_LBE(T)*1)**(0.33)
        #dissolution.printDimensionlessParameters()
        beta = (K/(C_O_bulk)**4/C_Fe_bulk)**(1/2)
        #print(beta)
        C_Fe, C_Cr, C_O = dissolution.boundaryProblem([C_Fe_bulk, C_Cr_bulk, C_O_bulk], [C_Fe_bulk/dissolution.a_C, (beta)/dissolution.a_C, (C_O_bulk)/dissolution.a_C], [1, 1, 1])
        #C_Fe, C_Cr, C_O = dissolution.boundaryProblem([C_Fe_bulk, C_Cr_bulk, C_O_bulk], [1, 1, 1], [1, 1, 1])
        '''
        if not(K>2e-36 and C_O_bulk_mass_percent == 1e-6):
            C_Fe, C_Cr, C_O = dissolution.boundaryProblem([C_Fe_bulk, C_Cr_bulk, C_O_bulk], [C_Fe_bulk/dissolution.a_C, (beta)/dissolution.a_C, (C_O_bulk)/dissolution.a_C], [1, 1, 1])
        else:
            C_Fe, C_Cr, C_O = dissolution.boundaryProblem([C_Fe_bulk, C_Cr_bulk, C_O_bulk], [C_Fe_bulk/dissolution.a_C, (7.2e-12)/dissolution.a_C, (C_O_bulk)/dissolution.a_C], [1, 1, 1])
        '''
        #print("C_Fe_mass_percent")
        #print(C_Fe*100*m_Fe/rho_LBE(T))

        if C_Fe>C_Fe_s_LBE(T):
            red_console_massage("Fe is oversaturated")
        if C_Cr>C_Cr_s_LBE(T):
            red_console_massage("Cr is oversaturated")
        if C_O>C_O_s_LBE(T):
            red_console_massage("O is oversaturated")
        if Cr2O3_sediment_check(C_Cr, C_O, T): red_console_massage("Cr2O3 precipitates")
        else: red_console_massage("Cr2O3 not precipitates")
        Omega_FeCr2O4 = (1.0*m_Fe+2.0*m_Cr+4.0*m_O)/feCr2O4.rho()
        #print_state(C_Fe=C_Fe, C_Cr=C_Cr, C_O=C_O, C_Fe_bulk=C_Fe_bulk, 
        #            C_Cr_bulk=C_Cr_bulk, C_O_bulk=C_O_bulk, delta_C_Cr=C_Cr_bulk-C_Cr, 
        #            J_multiplied_by_l_diff = Omega_FeCr2O4*(D_Cr_LBE(T)*1)/2.0*(C_Cr_bulk-C_Cr))
        data.append(3600*8760*1e6*Omega_FeCr2O4*(D_Cr_LBE(T)*1)/2.0*(C_Cr_bulk-C_Cr)/l_diff)



    file = open("txt_data_files/different_T.txt",'w')
    file.write("V|")
    for p in range(N):
        if p != N-1:
            file.write("| T={} |".format(T_Kelvin[p]))
        else:
            file.write("| T={} \n ".format(T_Kelvin[p]))
    for i in range(n):
        file.write("{}\t".format(V[i]))
        for p in range(N):
            if p != N-1:
                file.write("{}\t".format(data[p][i]))
            else:
                file.write("{}\n".format(data[p][i]))
    file.close()

def checking_the_stability_of_the_solution():
    T = 600 + 273.15
    fe3O4_s = Fe3O4_s()
    feCr2O4 = FeCr2O4_s()
    pbO_s = PbO_yellow_phase_s()
    fe_s = Fe_alpha_delta_phase_s()
    pb_l = Pb_l()

    C_O_bulk_mass_percent = 1e-6 # mass, %
    C_Cr_bulk_mass_percent = 0 # mass, %
    C_Fe_bulk_mass_percent = 0 # mass, %
    C_O_bulk = C_O_bulk_mass_percent/100.0/m_O*rho_LBE(T)
    C_Cr_bulk = C_Cr_bulk_mass_percent/100.0/m_Cr*rho_LBE(T)
    C_Fe_bulk = C_Fe_bulk_mass_percent/100.0/m_Fe*rho_LBE(T)


    K_Fe3O4 = np.exp((fe3O4_s.G_m(T)-4.0*(pbO_s.G_m(T)-mu_Pb_LBE(T))-3.0*fe_s.G_m(T))/(R*T))*C_O_s_LBE(T)**4.0*C_Fe_s_LBE(T)**3.0
    C_Fe_bulk = (K_Fe3O4/((1e-6/100.0/m_O*rho_LBE(T)))**4.0)**(1.0/3.0)
        
    K = np.exp((feCr2O4.G_m(T) - 1*fe_s.G_m(T) - 2*Cr_s().G_m(T) - 4*(pbO_s.G_m(T) - mu_Pb_LBE(T)))/ (R*T))*(C_Fe_s_LBE(T)**1.0)*(C_Cr_s_LBE(T)**2.0)*(C_O_s_LBE(T)**4.0)
    beta = (K/(C_O_bulk)**4/C_Fe_bulk)**(1.0/2.0)
    dissolution = Dissolution(C_Fe_bulk, C_Cr_bulk, C_O_bulk,  T, K)
    n = 10000
    epsilon = 1/2
    C_start_Fe = np.linspace((C_Fe_bulk-beta*epsilon)/dissolution.a_C, (C_Fe_bulk+beta*epsilon)/dissolution.a_C, n)
    C_start_Cr = np.linspace(beta*(1-epsilon)/dissolution.a_C, beta*(1+epsilon)/dissolution.a_C, n)
    C_start_O = np.linspace((C_O_bulk-beta*epsilon)/dissolution.a_C, (C_O_bulk+beta*epsilon)/dissolution.a_C, n)
    #C_start_Fe = np.linspace(0.01, 1, n)
    #C_start_Cr = np.linspace(0.01, 1, n)
    #C_start_O = np.linspace(0.01, 1, n)
   
    Solution_Fe = []
    Error_Solution_Fe = []
    Solution_Cr = []
    Error_Solution_Cr = []
    Solution_O = []
    Error_Solution_O = []
    for i in range(n):
        '''
        C_Fe, _, _ = dissolution.boundaryProblem([C_Fe_bulk, C_Cr_bulk, C_O_bulk], [C_start_Fe[i], beta, C_O_bulk], [1, 1, 1])
        _, C_Cr, _ = dissolution.boundaryProblem([C_Fe_bulk, C_Cr_bulk, C_O_bulk], [C_Fe_bulk, C_start_Cr[i], C_O_bulk], [1, 1, 1])
        _, _, C_O = dissolution.boundaryProblem([C_Fe_bulk, C_Cr_bulk, C_O_bulk], [C_Fe_bulk, beta, C_start_O[i]], [1, 1, 1])
        '''
        C_Fe, C_Cr, C_O = dissolution.boundaryProblem([C_Fe_bulk, C_Cr_bulk, C_O_bulk], [C_start_Fe[i], C_start_Cr[i], C_start_O[i]], [1, 1, 1])

        Solution_Fe.append(C_Fe)
        Solution_Cr.append(C_Cr)
        Solution_O.append(C_O)

    C_Fe_avr = np.mean(Solution_Fe)
    C_Cr_avr = np.mean(Solution_Cr)
    C_O_avr = np.mean(Solution_O)

    Error_Solution_Fe = list(map(lambda C_Fe: abs((C_Fe_avr-C_Fe)/C_Fe_avr), Solution_Fe))
    Error_Solution_Cr = list(map(lambda C_Cr: abs((C_Cr_avr-C_Cr)/C_Cr_avr), Solution_Cr))
    Error_Solution_O = list(map(lambda C_O: abs((C_O_avr-C_O)/C_O_avr), Solution_O))
    
    C_start_Fe *= dissolution.a_C
    C_start_Cr *= dissolution.a_C
    C_start_O *= dissolution.a_C

    file = open("txt_data_files/stability.txt",'w')
    file.write("C_start_Fe Solution_Fe Error_Solution_Fe\n")
    for i in range(len(C_start_Fe)):
        file.write("{}\t".format(C_start_Fe[i]))
        file.write("{}\t".format(Solution_Fe[i]))
        file.write("{}\t".format(Error_Solution_Fe[i]))
        file.write("{}\t".format(C_start_Cr[i]))
        file.write("{}\t".format(Solution_Cr[i]))
        file.write("{}\t".format(Error_Solution_Cr[i]))
        file.write("{}\t".format(C_start_O[i]))
        file.write("{}\t".format(Solution_O[i]))
        file.write("{}\n".format(Error_Solution_O[i]))
    file.close()



def do_everything():
    checking_the_stability_of_the_solution()
    #FeCr2O4_dissolution_with_D_multiplyer()
    #FeCr2O4_dissolution_with_different_T()


do_everything()


