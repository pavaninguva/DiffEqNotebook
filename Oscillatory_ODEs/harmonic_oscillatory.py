import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



""""
This script contains functions to solve three variants of the simple harmonic oscillator:
1. Undamped Harmonic Oscillator given by x'' + (k/m)x = 0.
x is the displacement, k is the spring constant and m is the mass of the object
2. Damped Harmonic Oscillator given by 
"""

#Plotting formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

# Define Undamped Harmonic Oscillator ODE Function
def Undamped(t,y,w_0):
    x = y[0]
    xdot = y[1]
    #Define system of ODEs
    dxdt = xdot
    dxdotdt = -(w_0**2)*x 
    return [dxdt, dxdotdt]

#Define tspan and initial conditions
tspan = np.linspace(0,10,1000)
x_0 = [5,0]
w_0 = 1.0

#Solve the differential equation
sol = solve_ivp(lambda t,x:Undamped(t,x,w_0),[0,10],x_0,method="LSODA",dense_output=True)

#Plotting Results
plt.figure()
plt.subplot(121)
plt.plot(sol.t, sol.y[0])
plt.xlabel(r"$t$")
plt.ylabel(r"$x$")
plt.show()



