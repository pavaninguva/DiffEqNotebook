import numpy as np
from advection_schemes import *
import matplotlib.pyplot as plt

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')


def c0_fun(x,y):
    Nx = len(x)
    Ny = len(y)
    c0 = np.zeros((Nx,Ny))
    for i in range(Nx):
        for j in range(Ny):
            if (x[i,j]-0.5)**2 + (y[i,j]-0.5)**2 < 0.1:
                c0[i,j] = 1.0
            else:
                c0[i,j] = 0.0
    return c0

# Plot initial condition
x_vals = np.linspace(0,2,num=101)
y_vals = np.linspace(0,2,num=101)
xx,yy = np.meshgrid(x_vals,y_vals)

c0 = c0_fun(xx,yy)

fig1 = plt.figure(num=1,figsize=(5,4))
ax1 = fig1.add_subplot(1,1,1)
im = ax1.pcolormesh(x_vals,y_vals,c0, cmap="jet")
fig1.colorbar(im,label=r"$c$")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$y$")
fig1.tight_layout()


# Explicit upwind for 2D 
c_vals = explict_upwind_2d([0,2],[0,2],1.5,101,101,1.0,0.0,1.0,c0_fun)
fig2 = plt.figure(num=2,figsize=(5,4))
ax2 = fig2.add_subplot(1,1,1)
im2 = ax2.pcolormesh(x_vals,y_vals,c_vals, cmap="jet")
fig2.colorbar(im,label=r"$c$")
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$y$")
fig2.tight_layout()


plt.show()