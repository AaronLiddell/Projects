#Interpolating magnetic field at z

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

data = np.genfromtxt("force.txt", filling_values=np.nan)

separation = data[:, 0]
force = data[:, 1]
z = data[:, 2]
B = data[:, 3]

f_interp = interp1d(z, B, kind='cubic', fill_value='extrapolate')
B_interp = f_interp(separation)


print("separation (m) | Force (N) | Interpolated B (T)")
for sep, F, B_val in zip(separation, force, B_interp):
    print(f"{sep:.7f}, {F:.7f}, {B_val:.7f}")

# Plot extrapolated/interpolated B vs separation and compare with original B vs z

plt.plot(z, B, 'o', label='Original B vs z (data)')
plt.plot(separation, B_interp, 's', label='Extrapolated B vs z')
plt.xlabel('Separation (m)')
plt.ylabel('B (T)')
plt.title('Comparison of Magnetic Field: Data vs Extrapolation')
plt.legend()
plt.grid(False)
plt.show()

