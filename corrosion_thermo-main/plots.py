import matplotlib.pyplot as plt
import matplotlib.colors as colorslib
import matplotlib as mpl
import numpy as np
import sys
import math
import ast
from rich.console import Console
sys.path.append('./src/corrosionthermo')
import corrosionthermo as term
plt.rcParams['text.usetex'] = True
from corrosionthermo import C_Cr_s_LBE,  C_O_s_LBE, C_Fe_s_LBE
from Cr2O3_dissolution_estimate import C_Cromium_from_C_O
from matplotlib.ticker import ScalarFormatter


def f(X, Y, C_O_S, C_Fe_S, C_Cr_S, delta):
    #plane eq: Ax + By + Cz + D = 0, so if С not equal to zero, and it is not equal to zero,
    A = 2
    B = 1
    C = 4
    D = -1 * (4*C_O_S + 1*C_Fe_S + 2*C_Cr_S + delta)
    alpha = -1 * (A/C)
    beta = -1 * (B/C)
    gamma = -1 * (D/C)
    Z = X+Y
    for i in range(len(X)):
        for j in range(len(Y)):
            Z[i][j] = alpha * X[i][j] + beta * Y[i][j] + gamma
    return Z

def diagram(T, label):
    #next vars are functions of T
    #Saturation concentration of oxygen, ferrum, Chromium
    C_O_S = math.log10(term.C_O_s_LBE(T))
    C_Fe_S = math.log10(term.C_Fe_s_LBE(T))
    C_Cr_S = math.log10(term.C_Cr_s_LBE(T))
    #G_FeCr2O4_m - mu_Fe - 2mu_Cr - 4mu_O = delta
    delta = term.FeCr2O4_s().G_m(T) - term.Fe_alpha_delta_phase_s().G_m(T) - 2*term.Cr_s().G_m(T) - 4*(term.PbO_yellow_phase_s().G_m(T) - term.mu_Pb_LBE(T))
    delta = delta * (math.log10(math.e)/(term.R *T))
    #Plot resolution "figsize=[res_x, res_y]"
    fig = plt.figure(figsize=[10, 10])
    fig = plt.figure(figsize=[10, 10])
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    #G_FeCr2O4_m - mu_Fe - 2mu_Cr - 4mu_O
    n = 2
    
    x = np.linspace(delta/2 + C_Cr_S, C_Cr_S, n)
    y = np.linspace(delta + C_Fe_S, C_Fe_S, n)
    X_equlibrium, Y_equlibrium = np.meshgrid(x, y)

    X_equlibrium = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        X_equlibrium[i] = [x[-1] - (x[-1] - x[0]) * j * (i)/((n-1)**2)  for j in range(n)]
    Z_equlibrium = f(X_equlibrium, Y_equlibrium, C_O_S, C_Fe_S, C_Cr_S, delta)


    min = 100000
    for i in range(len(Z_equlibrium)):
        for j in range(len(Z_equlibrium[0])):
            if Z_equlibrium[i][j]<min:
                min = Z_equlibrium[i][j]
    ax1.plot_surface(X_equlibrium, Y_equlibrium, Z_equlibrium, color = "black", alpha=0.5)

    
    y = np.linspace(delta + C_Fe_S, C_Fe_S, n)
    z = np.linspace(delta/4 + C_O_S, C_O_S, n)
    Y_Cr, Z_Cr = np.meshgrid(y, z)
    X_Cr = C_Cr_S
    ax1.plot_surface(X_Cr, Y_Cr, Z_Cr, color = "red", alpha=0.6, label="$C_{Cr}^s = " + str(round(10**C_Cr_S, 4)) +"$")


    x = np.linspace(delta/2 + C_Cr_S, C_Cr_S, n)
    z = np.linspace(delta/4 + C_O_S, C_O_S, n)
    X_Fe, Z_Fe = np.meshgrid(x, z)
    Y_Fe = C_Fe_S
    ax1.plot_surface(X_Fe, Y_Fe, Z_Fe, color = "green", alpha=0.6, label="$C_{Fe}^s = " + str(round(10**C_Fe_S, 4)) +"$")

    x = np.linspace(delta/2 + C_Cr_S, C_Cr_S, n)
    y = np.linspace(delta + C_Fe_S, C_Fe_S, n)
    X_O, Y_O = np.meshgrid(x, y)
    f_O = lambda a, b: a*0 + b*0 + C_O_S
    Z_O = f_O(X_O,  Y_O)
    ax1.plot_surface(X_O, Y_O, Z_O, color = "blue", alpha=0.6, label="$C_{O}^s = " + str(round(10**C_O_S, 4)) +"$")

    locs = ax1.get_xticks()
    labels = []
    for i in range(len(locs)) :
        labels.append('$10^{' + str(round(locs[i], 2)) + '}$')
    ax1.set_xticks(locs, labels)

    locs = ax1.get_yticks()
    labels = []
    for i in range(len(locs)) :
       labels.append('$10^{' + str(round(locs[i], 2)) + '}$')
    ax1.set_yticks(locs, labels)

    locs = ax1.get_zticks()
    labels = []
    for i in range(len(locs)) :
        labels.append('$10^{' + str(round(locs[i], 2)) + '}$')
    ax1.set_zticks(locs, labels)

    ax1.set_xlabel('$C_{Cr}$', fontsize=8)
    ax1.set_ylabel('$C_{Fe}$', fontsize=8)
    ax1.set_zlabel('$C_{O}$', fontsize=8)
    ax1.legend()

    k = 0.3
    ax2.set_title('$C_{Fe}=C_{Fe}^s$')
    ax2.plot([delta/2 + C_Cr_S,  C_Cr_S], [C_O_S, delta/4 + C_O_S], color = (0, 0, 0), label="$Fe Cr_2 O_4$ saturation")
    ax2.plot([C_Cr_S,  C_Cr_S], [C_O_S-delta*k/4, delta/4 + C_O_S], color = (1, 0, 0), label="$Cr$ saturation")
    ax2.plot([delta/2 + C_Cr_S,  C_Cr_S-delta*k/2], [C_O_S, C_O_S], color = (0, 0, 1), label="$O$ saturation")
    ax2.grid()
    ax2.legend()
    locs = [-25, -20, -15, -10,  -5,   0,   5]
    labels = ['$10^{-25.0}$', '$10^{-20.0}$', '$10^{-15.0}$', '$10^{-10.0}$', '$10^{-5.0}$', '$10^{0.0}$', '$10^{5.0}$']
    ax2.set_xticks(locs, labels)
    locs = ax2.get_yticks()
    labels = []
    for i in range(len(locs)) :
        labels.append('$10^{' + str(round(locs[i], 2)) + '}$')
    ax2.set_yticks(locs, labels)




    plt.savefig("plots/"+label+".png")
    #plt.show()

def dissolution_visualisation_variation_of_parameters():
    fig = plt.figure(figsize=[7,5])
    ax = fig.add_subplot()
    f = open("txt_data_files/variation_of_parameters.txt")
    data = f.readline()
    lbls = data.split("||")[1:]
    lbls = list(map(lambda x: x[12:], lbls))
    n = len(lbls)
    nums = [[] for i in range(n)]
    data = f.readline()
    V = []
    while data!="":
        data = data.split()
        for i in range(len(data)):
            data[i] = float(data[i])
        V.append(data[0])
        for i in range(1, len(data)):
            nums[i-1].append(data[i])
        data = f.readline()
    f.close()
    j=0
    dLdT_mins = []
    for i in range(n):
        dLdT_mins.append(nums[i][-1])
    dLdT_min = min(dLdT_mins)
    dLdT_max = max(dLdT_mins)
    max_painted=False
    min_painted=False
    for i in range(n):
        list_M = ast.literal_eval(lbls[i])
        if ((nums[i][-1]==dLdT_min and not min_painted) or (nums[i][-1]==dLdT_max and not max_painted) or list_M==[1, 1, 1]):
            if nums[i][-1]==dLdT_min:
                min_painted=True
            if nums[i][-1]==dLdT_max:
                max_painted=True
            Fe_D_M, Cr_D_M, O_D_M = list_M
            ax.plot(V, nums[i], label="$D_{Fe} \\times "+str(Fe_D_M)+", D_{Cr} \\times "+str(Cr_D_M)+",D_{O} \\times "+str(O_D_M)+"$", linestyle="-", color="C"+str(j))
            j+=1
            if nums[i][-1]==dLdT_min:
                y_min = nums[i]
            if nums[i][-1]==dLdT_max:
                y_max = nums[i]
    ax.fill_between(V,  y_min, y_max, color = (100/255, 100/255, 100/255))
    ax.grid()
    fmt = ScalarFormatter()
    fmt.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel('$ V [m/s] $', fontsize=15)
    ax.set_ylabel('$ \\frac{dL}{dt} [\\mu m/ \\textit{year }] $', fontsize=15)
    plt.legend()
    plt.savefig("plots/dissolution")
    #plt.show()

def dissolution_visualisation_different_T():
    fig = plt.figure()
    ax = fig.add_subplot()
    f = open("txt_data_files/different_T.txt")
    data = f.readline()
    lbls = data.split("||")[1:]
    lbls = list(map(lambda x: float(x[3:]), lbls))
    n = len(lbls)
    nums = [[] for i in range(n)]
    data = f.readline()
    V = []
    while data!="":
        data = data.split()
        for i in range(len(data)):
            data[i] = float(data[i])
        V.append(data[0])
        for i in range(1, len(data)):
            nums[i-1].append(data[i])
        data = f.readline()
    f.close()
    j=0
    for i in range(n):
        j+=1
        ax.plot(V, nums[i], label="$T = {" + str(lbls[i]-273.15) + "}^\circ C$", linestyle="-", color="C"+str(j))
    ax.grid()
    ax.set_xlabel('$ V [m/s] $', fontsize=15)
    ax.set_ylabel('$ \\frac{dL}{dt} [\\mu m/ \\textit{year }] $', fontsize=15)
    fmt = ScalarFormatter()
    fmt.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(fmt)
    plt.legend()
    plt.savefig("plots/dissolution_different_T")
    #plt.show()

def concentration_surface(K, C_Fe, C_Cr, C_O, C_Fe_b, C_Cr_b, C_O_b, D_Fe, D_Cr, D_O):
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(projection='3d')
    n=-12
    x = np.linspace(np.abs(C_Fe/2),np.abs(2*C_Fe), 100)
    y = np.linspace(np.abs(C_Cr/2), np.abs(2*C_Cr), 100)
    Fe, Cr= np.meshgrid(x, y)
    O = (K/(Fe**2*Cr))**(1/4)
    ax.plot_surface(Fe, Cr, O)
    ax.scatter(C_Fe, C_Cr, C_O, s=40, color='orange', marker='o')
    t = C_Fe
    #ax.plot([C_Fe_b, C_Fe], [C_Cr_b, C_Cr], [C_O_b, C_O])
    plt.show()

def red_console_massage(text):
    console = Console()
    console.print(text, style="red on white")

def Gibbs_energy_Elements():
    fig = plt.figure()
    ax = fig.add_subplot()
    fe3o4 = term.Fe3O4_s()
    fecr2o4 = term.FeCr2O4_s()
    cr2o3 = term.Cr2O3_s()
    fe = term.Fe_alpha_delta_phase_s()
    x = np.linspace(425+273.15, 475+273.15, 50)
    #delta_real = list(map(lambda x: 2*fecr2o4.G_m(x) - 2*cr2o3.G_m(x) - 225000 + 60000, x))
    data_Fe3O4 = list(map(lambda x: fe3o4.G_m(x) , x))
    data_Cr2O3 = list(map(lambda x: cr2o3.G_m(x) , x))
    data_FeCr2O4 = list(map(lambda x: fecr2o4.G_m(x) , x))
    ax.plot(x, data_Cr2O3, label="$Cr_2 O_3$")
    ax.plot(x, data_Fe3O4, label="$Fe_3 O_4$")
    ax.plot(x, data_FeCr2O4, label="$Fe Cr_2 O_4$")
    ax.grid()
    ax.legend()
    ax.legend(fontsize=15)
    ax.set_xlabel("$T, K$", size=15)
    ax.set_ylabel("$G^m, J$", size=15)
    plt.savefig("plots/elements_energy")
    #plt.show()

def C_Cr_diagram():
    n = 100
    x = np.linspace(425+273.15, 475+273.15, n)
    C_O = 0.006331625132820801
    f = lambda x:  C_Cr_s_LBE(x)
    g = lambda x:  C_Cromium_from_C_O(C_O, x)
    y1 = np.array(list(map(f, x)))
    y2 = np.array(list(map(g, x)))
    y3 = [max(y1[i], y2[i]) for i in range(n)]
    y4 = [min(y1[i], y2[i]) for i in range(n)]
    max_value = max(y3)
    col1 = (229/255, 236/255, 185/255)
    col2 = (0, 0.5, 0)
    col3 = (254/255, 237/255, 133/255)
    col4 = (142/255, 169/255, 38/255)
    fig, (ax1, ax2) = plt.subplots(2, 1, layout='constrained', figsize=(6, 10))
    ax1.plot(x, y1, label="$Cr$ saturation concentration", color="black")
    ax1.plot(x, y2, label="$Cr_2 O_3$ saturation concentration", color="red")
    ax1.fill_between(x, y1, y2, where=(y1 > y2), color=col1)
    ax1.fill_between(x, y1, y2, where=(y1 < y2), color=col2)
    ax1.fill_between(x, y3, max_value, color=col3)
    ax1.fill_between(x, y4, 0, color=col4)
    ax2.plot(x, y2, label="$Cr_2 O_3$ saturation concentration", color="red")
    ax2.fill_between(x, y4, max(y4), color=col1)
    ax2.fill_between(x, y4, color = col4)
    ax1.set_ylabel('Chromium concentration', fontsize=15)
    ax1.set_xlabel('Temperature [K]', fontsize=15)
    ax2.set_ylabel('Chromium concentration', fontsize=15)
    ax2.set_xlabel('Temperature [K]', fontsize=15)
    ax1.legend()
    ax2.legend()
    plt.savefig("plots/C_Cr_diagram")

def test(T, label):
    C_O_S = math.log10(term.C_O_s_LBE(T))
    C_Fe_S = math.log10(term.C_Fe_s_LBE(T))
    C_Cr_S = math.log10(term.C_Cr_s_LBE(T))
    #G_FeCr2O4_m - mu_Fe - 2mu_Cr - 4mu_O = delta
    delta = term.FeCr2O4_s().G_m(T) - term.Fe_alpha_delta_phase_s().G_m(T) - 2*term.Cr_s().G_m(T) - 4*(term.PbO_yellow_phase_s().G_m(T) - term.mu_Pb_LBE(T))
    delta = delta * (math.log10(math.e)/(term.R *T))
    #Plot resolution "figsize=[res_x, res_y]"
    fig = plt.figure(figsize=[4.5, 4.5])
    ax2 = fig.add_subplot()
    k = 0.3

    ax2.set_xlim(xmin=delta/2 + C_Cr_S, xmax=C_Cr_S-delta*k/2)
    ax2.set_ylim(delta/4 + C_O_S, C_O_S-delta*k/4)
    ax2.set_title('$C_{Fe}=C_{Fe}^s$')
    ax2.plot([delta/2 + C_Cr_S,  C_Cr_S], [C_O_S, delta/4 + C_O_S], color = (0, 0, 0), label="$Fe Cr_2 O_4$ saturation")
    ax2.plot([C_Cr_S,  C_Cr_S], [C_O_S-delta*k/4, delta/4 + C_O_S], color = (1, 0, 0), label="$Cr$ saturation")
    ax2.plot([delta/2 + C_Cr_S,  C_Cr_S-delta*k/2], [C_O_S, C_O_S], color = (0, 0, 1), label="$O$ saturation")
    ax2.fill_between([delta/2 + C_Cr_S,  C_Cr_S], [C_O_S, delta/4 + C_O_S], [C_O_S, C_O_S], color = (0.2, 0.2, 0.2), label="$Fe Cr_2 O_4$ is saturated")
    ax2.fill_between([delta/2 + C_Cr_S,  C_Cr_S], C_O_S-delta*k/4, [C_O_S, C_O_S], color = (0, 0, 0.6), label="$Fe Cr_2 O_4$ and $O$ are saturated")
    ax2.fill_between([C_Cr_S,  C_Cr_S-delta*k/2], C_O_S, delta/4 + C_O_S, color = (0.6, 0, 0), label="$Fe Cr_2 O_4$ and $Cr$ are saturated")
    ax2.fill_between([C_Cr_S,  C_Cr_S-delta*k/2], C_O_S, C_O_S-delta*k/4, color = (0.6, 0, 0.6), label="$Fe Cr_2 O_4$, $O$ and $Cr$ are saturated")
    
    ax2.grid()
    ax2.legend()
    locs = [-25, -20, -15, -10,  -5,   0,   5]
    labels = ['$10^{-25.0}$', '$10^{-20.0}$', '$10^{-15.0}$', '$10^{-10.0}$', '$10^{-5.0}$', '$10^{0.0}$', '$10^{5.0}$']
    ax2.set_xticks(locs, labels)
    locs = ax2.get_yticks()
    labels = []
    for i in range(len(locs)) :
        labels.append('$10^{' + str(round(locs[i], 2)) + '}$')
    ax2.set_yticks(locs, labels)
    ax2.set_xlabel("$C_{Cr}$")
    ax2.set_ylabel("$C_{O}$")


    plt.savefig("plots/"+label+".png")
    
def compare():
    fig = plt.figure()
    ax = fig.add_subplot()
    n = 100
    cr2o3 = term.Cr2O3_s()
    fe304 = term.Fe3O4_s()
    fecr2o4 = term.FeCr2O4_s()
    o2 = term.O2_g()
    Cr = term.Cr_s()
    Fe = term.Fe_alpha_delta_phase_s()
    T = np.linspace(425+273.15, 475+273.15, n)
    C=-751.22*1000
    D=0.17*1000
    G_Cr2O3_CD = list(map(lambda x: 3/2.0*(C+D*x) + 2*Cr.G_m(x) + 3/2.0 * (o2.G_m(x)), T))
    G_Cr2O3_nist = list(map(lambda x: cr2o3.G_m(x) , T))
    ax.plot(T, G_Cr2O3_nist, label="$G^m_{Cr_2O_3}$ [1]", color="C1")
    ax.plot(T, G_Cr2O3_CD, label="$G^m_{Cr_2O_3}$ [2]", color="C2")

    C_1 = -608.94 * 1000
    D_1 = 0.2241 * 1000
    C_2 = -529.77* 1000
    D_2 = 0.1302 * 1000
    G_Fe3O4_CD = list(map(lambda x: 1/2.0*(C_1+D_1*x) + 3*(1/2.0*(C_2+D_2*x) + Fe.G_m(x)+ 1/2.0 * o2.G_m(x)) + 1/2.0 * (o2.G_m(x)), T))
    G_Fe3O4_nits = list(map(lambda x: fe304.G_m(x) , T))
    ax.plot(T, G_Fe3O4_nits, label="$G^m_{Fe_3O_4} $ [1]", color="C1", linestyle="-.")
    ax.plot(T, G_Fe3O4_CD, label="$G^m_{Fe_3O_4}$ [2]", color="C2", linestyle="-.")

    #G_FeCr2O4_CD = list(map(lambda x: fecr2o4.G_m(x), T))
    #ax.plot(T, G_FeCr2O4_CD, label="$G^m_{FeCr_2O_4}$", color="C1", linestyle="--")


    ax.set_title('$G^m_{Cr_2O_3}$', size=20)
    ax.set_xlabel('$T, K$', size=20)
    ax.set_ylabel('$G^m, J$', size=20)
    ax.grid()
    ax.legend(fontsize=15)
    plt.savefig("plots/comparison")

def stability():
    fig = plt.figure(figsize=[8,8])
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    f = open('txt_data_files/stability.txt')
    data = f.readline()
    data = f.readline()
    C_Fe = []
    Fe_Solution = []
    Error_Fe_Solution = []
    C_Cr = []
    Cr_Solution = []
    Error_Cr_Solution = []
    C_O = []
    O_Solution = []
    Error_O_Solution = []
    while data!="":
        data = data.split()
        data = list(map(float, data))
        C_Fe.append(data[0])
        Fe_Solution.append(data[1])
        Error_Fe_Solution.append(data[2])
        C_Cr.append(data[3])
        Cr_Solution.append(data[4])
        Error_Cr_Solution.append(data[5])
        C_O.append(data[6])
        O_Solution.append(data[7])
        Error_O_Solution.append(data[8])
        data=f.readline()
    f.close()
    ax1.plot(C_Fe, Error_Fe_Solution)
    ax2.plot(C_Cr, Error_Cr_Solution)
    ax3.plot(C_O, Error_O_Solution)
    ax4.plot(C_Fe, Fe_Solution, label = "Fe_Solution")
    ax4.plot(C_Cr, Cr_Solution, label = "Cr_Solution")
    ax4.plot(C_O, O_Solution, label = "O_Solution")
     
    #ax1.xaxis.tick_top()
    #ax1.xaxis.set_label_position("top")
    ax1.set_xlabel("$C_{Fe}$ on iteration 0 [Mole/m$^3$]")
    ax1.set_ylabel("$ \\frac{ | \\Delta C_{Fe} | }{C_{Fe}}$")
    #ax2.xaxis.tick_top()
    #ax2.xaxis.set_label_position("top")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_xlabel("$C_{Cr}$ on iteration 0 [Mole/m$^3$]")
    ax2.set_ylabel("$ \\frac{ | \\Delta C_{Cr} | }{C_{Cr}}$")
    ax3.set_xlabel("$C_{O}$ on iteration 0 [Mole/m$^3$]")
    ax3.set_ylabel("$ \\frac{ | \\Delta C_{O} | }{C_{O}}$")
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    ax4.set_xlabel("$C$ we start with [Mole/m$^3$]")
    ax4.set_ylabel("$C$ we finish with [Mole/m$^3$]")
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    ax4.legend()
    plt.savefig("plots/stability")

def do_everything():
    stability()
    test(450+273.15, "O_Cr_sat_diagram")
    Gibbs_energy_Elements()
    compare()
    dissolution_visualisation_variation_of_parameters()
    dissolution_visualisation_different_T()

do_everything()