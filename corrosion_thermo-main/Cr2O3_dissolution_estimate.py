from scipy.optimize import fsolve
import numpy as np
import sys
sys.path.append('src/corrosionthermo/')
from corrosionthermo import Fe3O4_s
from corrosionthermo import PbO_yellow_phase_s
from corrosionthermo import Fe_alpha_delta_phase_s
from corrosionthermo import Cr_s
from corrosionthermo import Cr2O3_s
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
from corrosionthermo import C_Fe_s_LBE



def Cr2O3_sediment_check(C_Cr, C_O, T):
    pbO_s = PbO_yellow_phase_s()
    cr_s = Cr_s()
    cr2O3_s = Cr2O3_s()
    K = (cr2O3_s.G_m(T) - 2*(2*cr_s.G_m(T)) - 3*(pbO_s.G_m(T) - mu_Pb_LBE(T)))/(R*T)
    if C_Cr==0 or C_O==0:
        #red_console_massage("Cr2O3 does not precipitate")
        return False
    elif   2 * np.log(C_Cr/C_Cr_s_LBE(T)) + 3 * np.log(C_O/C_O_s_LBE(T)) >= K:
        #red_console_massage("Cr2O3 precipitates")
        return True
    else:
        #red_console_massage("Cr2O3 does not precipitate")
        return False

def C_Cromium_from_C_O(C_0, T):
    K = np.exp((cr2O3_s.G_m(T) - 2*(2*cr_s.G_m(T)) - 3*(pbO_s.G_m(T) - mu_Pb_LBE(T)))/(R*T))*C_Cr_s_LBE(T)**2.0*C_O_s_LBE(T)**3.0
    C_Cr = (K/(C_0**3.0))**(1/2.0)
    return C_Cr


pbO_s = PbO_yellow_phase_s()
cr_s = Cr_s()
cr2O3_s = Cr2O3_s()
T = 450+273.15
K = np.exp((cr2O3_s.G_m(T) - 2*(2*cr_s.G_m(T)) - 3*(pbO_s.G_m(T) - mu_Pb_LBE(T)))/(R*T))*C_Cr_s_LBE(T)**2.0*C_O_s_LBE(T)**3.0

'''
pbO_s = PbO_yellow_phase_s()
cr_s = Cr_s()
cr2O3_s = Cr2O3_s()
T = 450 + 273.15
K = (cr2O3_s.G_m(T) - 2*(2*cr_s.G_m(T)) - 3*(pbO_s.G_m(T) - mu_Pb_LBE(T)))/(R*T)
C_0_1 = 0.006331625132820801
x = np.exp(1/2.0 * (K - 3 * np.log(C_0_1/C_O_s_LBE(T))))*C_Cr_s_LBE(T)
'''