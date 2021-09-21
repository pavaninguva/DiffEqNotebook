import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""
This script explores solutions to the Van der Pol equation:
x'' -k(1-x^2)x' + x = 0
"""
#Plotting formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

# Define Van der Pol model:
def VdP(t,y,mu):
    x = y[0]
    y = y[1]
    #Define system of ODEs
    dxdt = y
    dydt = mu*(1-x**2)*y -x
    return [dxdt, dydt]

# Solve model for different initial conditions and mu

sol1 = solve_ivp(lambda t,x:VdP(t,x,0.1),[0,20],[0.5,0.5],method="BDF",dense_output=True)
sol2 = solve_ivp(lambda t,x:VdP(t,x,0.1),[0,10],[2.0,2.0],method="BDF",dense_output=True)
sol3 = solve_ivp(lambda t,x:VdP(t,x,0.1),[0,10],[-2.0,-2.0],method="BDF",dense_output=True)

sol4 = solve_ivp(lambda t,x:VdP(t,x,1.0),[0,10],[0.5,0.5],method="BDF",dense_output=True)
sol5 = solve_ivp(lambda t,x:VdP(t,x,1.0),[0,10],[2.0,2.0],method="BDF",dense_output=True)
sol6 = solve_ivp(lambda t,x:VdP(t,x,1.0),[0,10],[-2.0,-2.0],method="BDF",dense_output=True)

sol7 = solve_ivp(lambda t,x:VdP(t,x,5.0),[0,20],[0.1,0.1],method="LSODA",dense_output=True)
sol8 = solve_ivp(lambda t,x:VdP(t,x,5.0),[0,30],[2.0,2.0],method="LSODA",dense_output=True)
sol9 = solve_ivp(lambda t,x:VdP(t,x,5.0),[0,30],[-2.0,-2.0],method="LSODA",dense_output=True)

# Plot results

# Phase portrait
fig1 = plt.figure(num=1,figsize=(12,4))
ax1 = fig1.add_subplot(1,3,1)
ax1.plot(sol1.y[0],sol1.y[1],label=r"$(x_{0},y_{0}) = (0.5,0.5)$")
ax1.plot(sol2.y[0],sol2.y[1],label=r"$(x_{0},y_{0}) = (2.0,2.0)$")
ax1.plot(sol3.y[0],sol3.y[1],label=r"$(x_{0},y_{0}) = (-2.0,-2.0)$")
ax1.set_title(r"$\mu = 0.1$")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$y$")
ax1.legend()
ax2 = fig1.add_subplot(1,3,2)
ax2.plot(sol4.y[0],sol4.y[1],label=r"$(x_{0},y_{0}) = (0.5,0.5)$")
ax2.plot(sol5.y[0],sol5.y[1],label=r"$(x_{0},y_{0}) = (2.0,2.0)$")
ax2.plot(sol6.y[0],sol6.y[1],label=r"$(x_{0},y_{0}) = (-2.0,-2.0)$")
ax2.set_title(r"$\mu = 1.0$")
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$y$")
ax2.legend()
ax3 = fig1.add_subplot(1,3,3)
ax3.plot(sol7.y[0],sol7.y[1],label=r"$(x_{0},y_{0}) = (0.1,0.1)$")
ax3.plot(sol8.y[0],sol8.y[1],label=r"$(x_{0},y_{0}) = (2.0,2.0)$")
ax3.plot(sol9.y[0],sol9.y[1],label=r"$(x_{0},y_{0}) = (-2.0,-2.0)$")
ax3.set_title(r"$\mu = 5.0$")
ax3.set_xlabel(r"$x$")
ax3.set_ylabel(r"$y$")
ax3.legend()
fig1.tight_layout()

#Time series
fig2 = plt.figure(num=2,figsize=(8,6))
ax4 = fig2.add_subplot(3,1,1)
ax4.plot(sol1.t,sol1.y[0],label=r"$x$")
ax4.plot(sol1.t,sol1.y[1],label=r"$y$")
ax4.set_title(r"$\mu = 0.1, (x_{0},y_{0}) = (0.5,0.5)$")
ax4.set_xlabel(r"$t$")
ax4.set_ylabel(r"$\mathbf{x}$")
ax4.legend()
ax5 = fig2.add_subplot(3,1,2)
ax5.plot(sol4.t,sol4.y[0],label=r"$x$")
ax5.plot(sol4.t,sol4.y[1],label=r"$y$")
ax5.set_title(r"$\mu = 1.0, (x_{0},y_{0}) = (0.5,0.5)$")
ax5.set_xlabel(r"$t$")
ax5.set_ylabel(r"$\mathbf{x}$")
ax5.legend()
ax6 = fig2.add_subplot(3,1,3)
ax6.plot(sol7.t,sol7.y[0],label=r"$x$")
ax6.plot(sol7.t,sol7.y[1],label=r"$y$")
ax6.set_title(r"$\mu = 5.0, (x_{0},y_{0}) = (0.1,0.1)$")
ax6.set_xlabel(r"$t$")
ax6.set_ylabel(r"$\mathbf{x}$")
ax6.legend()
fig2.tight_layout()

plt.show()