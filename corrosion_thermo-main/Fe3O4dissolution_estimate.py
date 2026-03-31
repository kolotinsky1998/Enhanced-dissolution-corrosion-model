from scipy.optimize import fsolve
import numpy as np
import sys
sys.path.append('src/corrosionthermo/')
from corrosionthermo import Fe3O4_s
from corrosionthermo import PbO_yellow_phase_s
from corrosionthermo import Fe_alpha_delta_phase_s
from corrosionthermo import mu_Pb_LBE
from corrosionthermo import R
from corrosionthermo import C_O_s_LBE
from corrosionthermo import C_Fe_s_LBE
from corrosionthermo import D_O_LBE
from corrosionthermo import D_Fe_LBE
from corrosionthermo import m_O
from corrosionthermo import m_Fe
from corrosionthermo import rho_LBE
from corrosionthermo import Pb_l
from corrosionthermo import nu_LBE
from Cr2O3_dissolution_estimate import Cr2O3_sediment_check
from plots import red_console_massage

def print_state(**variables):
    red_console_massage("")
    red_console_massage("=======================")
    for name, value in variables.items():
        red_console_massage(f"{name}: {value}")
        red_console_massage("=======================")
    red_console_massage("")



class Dissolution:
    def __init__(self, C_O_bulk, C_Fe_bulk, T, K):
        
        self.a_C = 3.0e-3
        self.a_D = 1e-6
        
        self.C_O_bulk = C_O_bulk/self.a_C
        self.C_Fe_bulk = C_Fe_bulk/self.a_C
        
        self.D_O_LBE = D_O_LBE(T)/self.a_D
        self.D_Fe_LBE = D_Fe_LBE(T)/self.a_D
        
        self.K = K/self.a_C**7.0
 
    def printDimensionlessParameters(self):
        print("C_O_bulk = {}".format(self.C_O_bulk))
        print("C_Fe_bulk = {}".format(self.C_Fe_bulk))
        print("D_Fe = {}".format(self.D_Fe_LBE))
        print("D_O = {}".format(self.D_O_LBE))
        print("K = {}".format(self.K))

    def boundaryProblem(self):
        def equations(p):
            C_O, C_Fe = p
            eq_1 = 3.0*(self.D_Fe_LBE/self.D_O_LBE)**0.33*self.D_O_LBE*(self.C_O_bulk-C_O)-4.0*self.D_Fe_LBE*(self.C_Fe_bulk-C_Fe)
            eq_2 = C_O**4.0*C_Fe**3.0 - self.K
            return (eq_1, eq_2)
        C_O, C_Fe = fsolve(equations, [0.01, 0.01])
        return C_O*self.a_C, C_Fe*self.a_C

T = 450 + 273.15
fe3O4_s = Fe3O4_s()
pbO_s = PbO_yellow_phase_s()
fe_s = Fe_alpha_delta_phase_s()
pb_l = Pb_l()

C_O_bulk_mass_percent = 1e-6 # mass, %
C_Fe_bulk_mass_percent = 1e-6 # mass, %
C_O_bulk = C_O_bulk_mass_percent/100.0/m_O*rho_LBE(T)
C_Fe_bulk = C_Fe_bulk_mass_percent/100.0/m_Fe*rho_LBE(T)

K = np.exp((fe3O4_s.G_m(T)-4.0*(pbO_s.G_m(T)-mu_Pb_LBE(T))-3.0*fe_s.G_m(T))/(R*T))*C_O_s_LBE(T)**4.0*C_Fe_s_LBE(T)**3.0
print(K)
print(C_Fe_bulk**3*C_O_bulk**4)
n = 10000
N = 5
V = np.linspace(0.5,2.5,n)
d = 0.01 
l_diff = 60.60*V**(-0.86)*d**(0.14)*nu_LBE(T)**(0.53)*D_Fe_LBE(T)**(0.33)

oversaturation = np.linspace(0.0,2.0,N)
C_Fe_bulk = oversaturation*(K/C_O_bulk**4.0)**(1.0/3.0)
red_console_massage(f"C_FE_BULF {C_Fe_bulk[2]}")


data = []
for p in range(N):
    dissolution = Dissolution(C_O_bulk, C_Fe_bulk[p], T, K)

    #dissolution.printDimensionlessParameters()
    C_O, C_Fe = dissolution.boundaryProblem()
    Omega_Fe3O4 = (3.0*m_Fe+4.0*m_O)/fe3O4_s.rho()
    print(C_Fe_bulk[p]-C_Fe)
    #print_state(C_Fe=C_Fe, C_O=C_O, C_Fe_bulk=C_Fe_bulk[p], 
    #            C_O_bulk=C_O_bulk, delta_C_Fe=C_Fe_bulk[p]-C_Fe, 
    #            J_multiplied_by_l_diff = Omega_Fe3O4*D_Fe_LBE(T)/3.0*(C_Fe_bulk[p]-C_Fe))
    
    data.append(3600*8760*1e6*Omega_Fe3O4*D_Fe_LBE(T)/3.0*(C_Fe_bulk[p]-C_Fe)/l_diff)





file = open("data.txt",'w')
file.write("V\t")
for p in range(N):
    if p != N-1:
        file.write("alpha={}\t".format(oversaturation[p]))
    else:
        file.write("alpha={}\n".format(oversaturation[p]))
for i in range(n):
    file.write("{}\t".format(V[i]))
    for p in range(N):
        if p != N-1:
            file.write("{}\t".format(data[p][i]))
        else:
            file.write("{}\n".format(data[p][i]))
file.close()

#visualisation
