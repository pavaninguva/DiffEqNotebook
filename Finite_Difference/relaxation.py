import numpy as np
import matplotlib.pyplot as plt

"""
This script employs the relaxation method to solve
the following nonlinear ODE:

T'' -a(T^4 - Tb^4) + g(x) = 0,

with the following boundary conditions
T(x=0) = 0
T(x=L) = 0
"""

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

# parameters and functions
def g(x):
    f = np.sin(np.pi*x/4)
    return f

Tb = 0.5
a = 1.0
Nx = 101
L = 1.0
x = np.linspace(0.0,L,Nx)
dx = L/(Nx-1)

Nsteps = 10000
T_vec = np.zeros((Nx,Nsteps+1))

#perform iterations
for i in range(Nsteps):
    c1 = ((T_vec[2:-1,i] + T_vec[1:-2,i] + T_vec[0:-3,i])/3) 
    c2 = -(a*(dx**2)/3)*(np.power(T_vec[1:-2,i],4) - Tb**4)
    c3 = ((dx**2)/3)*g(x[1:-2])

    T_vec[1:-2,i+1] = c1 + c2 + c3


#Compute error
def residual(T,x):
    c1 = (1/dx**2)*(T[2:-1] -2*T[1:-2] + T[0:-3])
    c2 = -a*(np.power(T[1:-2],4) - Tb**4)
    c3 = g(x[1:-2])
    F = c1 + c2 + c3
    res = np.linalg.norm(F)
    return res

res_vec = np.zeros(Nsteps+1)
for i in range(Nsteps+1):
    T_iter = T_vec[:,i]
    res_vec[i] = residual(T_iter,x)

suc_vec = np.zeros(Nsteps)
for i in range(Nsteps):
    Err_vec = T_vec[:,i+1] - T_vec[:,i]
    suc_vec[i] = np.linalg.norm(Err_vec)


#Plotting
fig1 = plt.figure(num=1,figsize=(5,4))
ax1 = fig1.add_subplot(1,1,1)
for j in range(Nsteps+1):
    if j %1000 == 0:
        ax1.plot(x,T_vec[:,j],label="%dth Iteration" % j)
ax1.legend()
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$T$")
fig1.tight_layout()

#Plot errors
color = "tab:red"
fig2 = plt.figure(num=2,figsize=(5,4))
ax2 = fig2.add_subplot(1,1,1)
#Residual
ax2.semilogy(np.linspace(0,Nsteps,Nsteps+1), res_vec, color=color)
ax2.set_xlabel("Number of Iterations")
ax2.set_ylabel("Residual",color=color)
#Successive Error
color = "tab:blue"
ax3 = ax2.twinx()
ax3.semilogy(np.linspace(1,Nsteps,Nsteps), suc_vec,"x",color=color)
ax3.set_ylabel(r"$|\mathbf{T}_{i+1} - \mathbf{T}_{i}|$", color=color)
fig2.tight_layout()

plt.show()






