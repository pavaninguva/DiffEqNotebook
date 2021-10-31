import numpy as np
from first_order import *
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

#Define functions and etc to solve the test case

def analytical(x):
    ana = x + (1/3)*x**3
    return ana

def ode(x,t):
    dfdx = 1 + x**2
    return dfdx

def rhs(x):
    g = 1 + x**2
    return g

#Initial conditions
f0 = 0.0

#Construct mesh
mesh = np.linspace(0,5,50)

#Solve using Finite Differences

#Forward Difference 1st Order
A1,b1 = forward_1(mesh,f0,rhs)
f1 = np.linalg.solve(A1,b1)

#Central Difference 2nd Order
A2,b2 = central_2(mesh,f0,rhs)
f2 = np.linalg.solve(A2,b2)

#solve ODE using scipy
f_scipy = solve_ivp(ode,[0,5],[f0],method="RK23",rtol=1e-8)

#Compute analytical solution
ana = analytical(mesh)

#plotting
fig1 = plt.figure(num=1, figsize=(5,5))
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(mesh,f1,label="Forward, 1st Order")
ax1.plot(mesh,f2,label="Central, 2nd Order")
ax1.plot(mesh,ana,label="Analytical")
ax1.plot(f_scipy.t,f_scipy.y[0],label="Scipy")
ax1.legend()
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$f$")
fig1.tight_layout()


# Compute MAE for different schemes

Ncells = [5,10,25,50,100,200,500,1000,1500,2000,2500,5000,10000]

mae_forward = np.zeros(len(Ncells))
mae_central = np.zeros(len(Ncells))

for i in range(len(Ncells)):
    N = Ncells[i]
    #Construct mesh
    mesh = np.linspace(0,5,N)
    #compute Ana
    ana = analytical(mesh)

    #Compute forward soln
    A1,b1 = forward_1(mesh,f0,rhs)
    f1 = np.linalg.solve(A1,b1)
    mae_forward[i] = abs(max(np.subtract(ana,f1),key=abs))

    #Compute central soln
    A2,b2 = central_2(mesh,f0,rhs)
    f2 = np.linalg.solve(A2,b2)
    mae_central[i] = abs(max(np.subtract(ana,f2),key=abs))


#Plot MAE
fig2 = plt.figure(num=2, figsize=(5,5))
ax2 = fig2.add_subplot(1,1,1)
ax2.loglog(Ncells,mae_forward,label="Forward, 1st Order")
ax2.loglog(Ncells,mae_central,label="Central, 2nd Order")
ax2.legend()
ax2.set_ylabel("MAE")
ax2.set_xlabel(r"$N_{Cells}$")

plt.show()