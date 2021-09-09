import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.integrate import solve_ivp


"""
This script outlines how finite differences can be used to solve 
1st order odes of the form:
dx/dt = f(x)

We shall consider a few different ways of solving them: 
1. Analytically where possible
2. Using an ODE solver 
3. Finite differences where we construct a matrix 
4. Finite differencing where we step in time / x
"""
#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')


"""
Visualizing the discretization
"""

#We can use numpy to generate a 2-d mesh:

x_vals = np.linspace(0,1,10)
y_vals = np.linspace(0,1,10)

x,y = np.meshgrid(x_vals,y_vals)


#plotting
fig1 = plt.figure(num=1,figsize=(4,4))
ax1 = fig1.add_subplot(1,1,1)
ax1.scatter(x,y,color="k")
segs1 = np.stack((x,y), axis=2)
segs2 = segs1.transpose(1,0,2)
fig1.gca().add_collection(LineCollection(segs1,color="k"))
fig1.gca().add_collection(LineCollection(segs2,color="k"))
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$y$")
fig1.tight_layout()


plt.show()
