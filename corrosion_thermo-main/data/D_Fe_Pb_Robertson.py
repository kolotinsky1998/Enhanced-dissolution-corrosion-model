from scipy.optimize import curve_fit
import numpy as np

def D_Fe_Pb_coefficients():
    file = open("/home/common/kolotinskiy.da/steel/thermodynamics/corrosionthermo/D_Fe_Pb_Robertson.txt")
    xdata = []
    ydata = []
    for line in file:
        xdata.append(float(line.split(" ")[0]))
        ydata.append(float(line.split(" ")[1]))

    def y(x, a, b):
        return a * x + b

    popt, pcov = curve_fit(y, xdata, ydata)

    a, b = popt
    return a, b

if __name__ == "__main__":
    n = 100
    T = np.linspace(400+273.15, 700+273.15, n)
    a, b = D_O_Pb_coefficients()
    for i in range(n):
        D_Fe = 1.0e-4*np.exp(a/T+b)
        print("{}\t{}".format(T[i], D_Fe[i]))

    




