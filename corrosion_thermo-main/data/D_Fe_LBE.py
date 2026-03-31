from scipy.optimize import curve_fit
import numpy as np

def D_Fe_LBE_coefficients():
    file = open("/home/common/kolotinskiy.da/steel/thermodynamics/corrosionthermo/D_Fe_LBE.txt")
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
    a, b = D_Fe_LBE_coefficients()
    print(a, b)

    




