import numpy as np
from advection_schemes import *
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

# define functions and etc to solve test case

def analytical_soln(x,u,t):
    c = (1.0/(np.sqrt(2*np.pi)))*np.exp(-0.5*(((x-u*t)-0.5)/0.05)**2)
    return c

def c0_fun(x):
    c0 = (1.0/(np.sqrt(2*np.pi)))*np.exp(-0.5*((x-0.5)/0.05)**2)
    return c0



# Compute solution for different schemes
#explicit upwind
c1 = explict_upwind_1d([0,2], 1.0, 101, 1.0, 1.0, c0_fun)
c2 = explict_upwind_1d([0,2],1.0, 101,1.0,0.8,c0_fun)
c3 = explict_upwind_1d([0,2], 1.0,101,1.0,0.5,c0_fun)

#lax wendroff
c4 = lax_wendroff_1d([0,2],1.0,101,1.0,1.0,c0_fun)
c5 = lax_wendroff_1d([0,2],1.0,101,1.0,0.8,c0_fun)
c6 = lax_wendroff_1d([0,2],1.0,101,1.0,0.5,c0_fun)

#Explicit FTCS
c7 = explicit_FTCS_1d([0,2],1.0,101,1.0,0.1,c0_fun)

#Implict Upwind
c8 = implicit_upwind_1d([0,2],1.0,101,1.0,1.0,c0_fun)
c9 = implicit_upwind_1d([0,2],1.0,101,1.0,0.5,c0_fun)
c10 = implicit_upwind_1d([0,2],1.0,101,1.0,2.0,c0_fun)

# Plotting
mesh = np.linspace(0,2,101)

c0 = c0_fun(mesh)
c_ana = analytical_soln(mesh, 1.0,1.0)

fig1 = plt.figure(num=1,figsize=(10,10))
#Explicit upwind
ax1 = fig1.add_subplot(2,2,1)
ax1.plot(mesh,c0,label="Initial Condition")
ax1.plot(mesh,c_ana,label="Analytical Solution")
ax1.plot(mesh,c1, label="CFL=1.0")
ax1.plot(mesh,c2, label="CFL=0.8")
ax1.plot(mesh,c3, label="CFL=0.5")
ax1.legend()
ax1.set_title("Explicit Upwind")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$c$")
#Lax wendroff
ax2 = fig1.add_subplot(2,2,2)
ax2.plot(mesh,c0,label="Initial Condition")
ax2.plot(mesh,c_ana,label="Analytical Solution")
ax2.plot(mesh,c4, label="CFL=1.0")
ax2.plot(mesh,c5, label="CFL=0.8")
ax2.plot(mesh,c6, label="CFL=0.5")
ax2.legend()
ax2.set_title("Lax-Wendroff")
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$c$")
# Explicit FTCS
ax3 = fig1.add_subplot(2,2,3)
ax3.plot(mesh,c0,label="Initial Condition")
ax3.plot(mesh,c_ana,label="Analytical Solution")
ax3.plot(mesh,c7, label="CFL=0.1")
ax3.legend()
ax3.set_title("Explicit FTCS")
ax3.set_xlabel(r"$x$")
ax3.set_ylabel(r"$c$")
# Implict FTCS
ax4 = fig1.add_subplot(2,2,4)
ax4.plot(mesh,c0,label="Initial Condition")
ax4.plot(mesh,c_ana,label="Analytical Solution")
ax4.plot(mesh,c8, label="CFL=1.0")
ax4.plot(mesh,c9, label="CFL=0.5")
ax4.plot(mesh,c10,label="CFL=2.0")
ax4.legend()
ax4.set_title("Implicit Upwind")
ax4.set_xlabel(r"$x$")
ax4.set_ylabel(r"$c$")
fig1.tight_layout()







plt.show()