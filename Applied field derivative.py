#Calculating the derivative of applied field wrt z

import numpy as np
from matplotlib import pyplot as plt

data = np.genfromtxt("dH by dz data.txt")
z = data[:, 0]
Ha = data[:, 1]

dHa_dz = np.gradient(Ha, z)

for zi, slope in zip(z, dHa_dz):
    #print(f"z = {zi:.2f} cm, dB/dz = {slope:.2f} mT/cm")
    print(f"{slope:.2f}")
