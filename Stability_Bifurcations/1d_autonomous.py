import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


"""
This script contains exemplar functions demonostrating the solution of
1d autonomous ODEs of the form: dx/dt = f(x)
"""

#Plotting formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

"""
Case 1: dx/dt = sin(x), 0 < x(t=0) < pi 
"""

def sine (t,y):
    x = y[0]
    #Define ODE
    dxdt = np.sin(x)
    return [dxdt]

sol1 = solve_ivp(sine, [0,10], [0.1], method="LSODA")
sol2 = solve_ivp(sine, [0,10], [0.4],method="LSODA")
sol3 = solve_ivp(sine, [0,10], [0.7],method="LSODA")
sol4 = solve_ivp(sine, [0,10], [1.0],method="LSODA")
sol5 = solve_ivp(sine, [0,10], [1.5],method="LSODA")
sol6 = solve_ivp(sine, [0,10], [1.8],method="LSODA")
sol7 = solve_ivp(sine, [0,10], [2.5],method="LSODA")
sol8 = solve_ivp(sine, [0,10], [3.0],method="LSODA")
sol9 = solve_ivp(sine, [0,10], [3.14],method="LSODA")

fig1 = plt.figure(num=1,figsize=(5,4))
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(sol1.t, sol1.y[0],label=r"$x_{0} = 0.1$")
ax1.plot(sol2.t, sol2.y[0],label=r"$x_{0} = 0.4$")
ax1.plot(sol3.t, sol3.y[0],label=r"$x_{0} = 0.7$")
ax1.plot(sol4.t, sol4.y[0],label=r"$x_{0} = 1.0$")
ax1.plot(sol5.t, sol5.y[0],label=r"$x_{0} = 1.5$")
ax1.plot(sol6.t, sol6.y[0],label=r"$x_{0} = 1.8$")
ax1.plot(sol7.t, sol7.y[0],label=r"$x_{0} = 2.5$")
ax1.plot(sol8.t, sol8.y[0],label=r"$x_{0} = 3.0$")
ax1.plot(sol9.t, sol9.y[0],label=r"$x_{0} = 3.14$")
ax1.set_xlabel(r"$t$")
ax1.set_ylabel(r"$x$")
ax1.legend()
fig1.tight_layout()




"""
Fishing Example
"""

# Consider the fixed amount harvesting strategy. 

def fixed_harvest(N,r,K,y0):
    f = r*N*(1-(N/K)) - y0
    return f

N = np.linspace(0,500,1000)

fig2 = plt.figure(num=2,figsize=(5,4))
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(N, fixed_harvest(N,10,500,0),label=r"$Y_{0}=0$")
ax2.plot(N, fixed_harvest(N,10,500,500),label=r"$Y_{0}=500$")
ax2.plot(N, fixed_harvest(N,10,500,1250),label=r"$Y_{0}=1250$")
ax2.plot(N, fixed_harvest(N,10,500,1500),label=r"$Y_{0}=1500$")
ax2.plot(N,0*N,"k")
ax2.set_xlabel(r"$N$")
ax2.set_ylabel(r"$\frac{dN}{dt} = f(N)$")
ax2.legend()
fig2.tight_layout()

def prop_harvest(N,r,K,alpha):
    f = r*N*(1-(N/K)) - alpha*N
    return f

fig3 = plt.figure(num=3,figsize=(5,4))
ax3 = fig3.add_subplot(1,1,1)
ax3.plot(N, prop_harvest(N,10,500,0),label=r"$\alpha=0$")
ax3.plot(N, prop_harvest(N,10,500,2.0),label=r"$\alpha=2.0$")
ax3.plot(N, prop_harvest(N,10,500,5.0),label=r"$\alpha=5.0$")
ax3.plot(N, prop_harvest(N,10,500,9.0),label=r"$\alpha=9.0$")
ax3.plot(N,0*N,"k")
ax3.set_xlabel(r"$N$")
ax3.set_ylabel(r"$\frac{dN}{dt} = f(N)$")
ax3.legend()
fig3.tight_layout()











plt.show()
