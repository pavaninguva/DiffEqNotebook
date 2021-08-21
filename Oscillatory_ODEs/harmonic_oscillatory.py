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

"""
Case 1
"""
# Define Undamped Harmonic Oscillator ODE Function
def Undamped(t,y,w_0):
    x = y[0]
    xdot = y[1]
    #Define system of ODEs
    dxdt = xdot
    dxdotdt = -(w_0**2)*x 
    return [dxdt, dxdotdt]

#Define parmas initial conditions
x_0 = [5,0]
w_0 = 2.5

#Solve the differential equation
sol = solve_ivp(lambda t,x:Undamped(t,x,w_0),[0,10],x_0,method="LSODA",dense_output=True)

#Plotting Results
fig1 = plt.figure(num=1,figsize=(8,4))
#plot time series
ax1 = fig1.add_subplot(1,2,1)
ax1.plot(sol.t, sol.y[0],color="k")
ax1.set_xlabel(r"$t$")
ax1.set_ylabel(r"$x$")
ax1.tick_params(axis='y', labelcolor="k")
ax2 = ax1.twinx()
ax2.plot(sol.t,sol.y[1],color="r")
ax2.set_ylabel(r"$\dot{x}$")
ax2.tick_params(axis='y', labelcolor="r")
#plot phase portrait
ax3 = fig1.add_subplot(1,2,2)
ax3.plot(sol.y[0],sol.y[1])
ax3.set_xlabel(r"$x$")
ax3.set_ylabel(r"$\dot{x}$")
fig1.tight_layout()


"""
Case 2
"""
# Define Undamped Harmonic Oscillator ODE Function
def Damped(t,y,w_0,xi):
    x = y[0]
    xdot = y[1]
    #Define system of ODEs
    dxdt = xdot
    dxdotdt = -(w_0**2)*x -2*xi*w_0*xdot
    return [dxdt, dxdotdt]

# Define xi and use previously defined params w_0 and time
xi_1 = 0.2
xi_2 = 1.0
xi_3 = 1.5

#Solve the different cases
sol2 = solve_ivp(lambda t,x:Damped(t,x,w_0,xi_1),[0,10],x_0,method="LSODA",dense_output=True)
sol3 = solve_ivp(lambda t,x:Damped(t,x,w_0,xi_2),[0,10],x_0,method="LSODA",dense_output=True)
sol4 = solve_ivp(lambda t,x:Damped(t,x,w_0,xi_3),[0,10],x_0,method="LSODA",dense_output=True)

# Plotting
fig2 = plt.figure(num=2,figsize=(8,4))
ax4 = fig2.add_subplot(1,1,1)
ax4.plot(sol2.t, sol2.y[0],color="k", label=r"$\xi=0.2$")
ax4.plot(sol3.t, sol3.y[0],color="r", label=r"$\xi=1.0$")
ax4.plot(sol4.t, sol4.y[0],color="b", label=r"$\xi=1.5$")
ax4.set_xlabel(r"$t$")
ax4.set_ylabel(r"$x$")
ax4.tick_params(axis='y', labelcolor="k")
ax4.legend(loc=1)

fig3 = plt.figure(num=3, figsize=(8,4))
ax5 = fig3.add_subplot(1,3,1)
ax5.plot(sol2.y[0],sol2.y[1], label=r"$\xi=0.2$")
ax5.set_xlabel(r"$x$")
ax5.set_ylabel(r"$\dot{x}$")
ax5.legend()
ax6 = fig3.add_subplot(1,3,2)
ax6.plot(sol3.y[0],sol3.y[1], label=r"$\xi=1.0$")
ax6.set_xlabel(r"$x$")
ax6.set_ylabel(r"$\dot{x}$")
ax6.legend()
ax7 = fig3.add_subplot(1,3,3)
ax7.plot(sol4.y[0],sol4.y[1], label=r"$\xi=1.5$")
ax7.set_xlabel(r"$x$")
ax7.set_ylabel(r"$\dot{x}$")
ax7.legend()
fig3.tight_layout()




plt.show()