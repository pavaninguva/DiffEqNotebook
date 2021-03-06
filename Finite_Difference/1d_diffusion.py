import numpy as np
from diffusion_schemes import *
import matplotlib.pyplot as plt

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

"""
Dirichlet Boundary Conditions
"""
#Define c0_fun
def c0_fun(x):
    # c0 = (1.0/(np.sqrt(2*np.pi)))*np.exp(-0.5*((x-0.5)/0.05)**2)
    c0 = np.zeros(int(len(x)))
    return c0

# Compute solution for different schemes and Fo
#Forward Euler
c1 = theta_scheme([0,1], 0.1, 501, 1.0, 0.5, 0, [1,0], c0_fun)
c2 = theta_scheme([0,1], 0.1, 501, 1.0, 0.2, 0, [1,0], c0_fun)
c3 = theta_scheme([0,1], 0.1, 501, 1.0, 0.50005, 0, [1,0], c0_fun)
c4 = theta_scheme([0,1], 0.1, 101, 1.0, 0.5, 0, [1,0], c0_fun)

#Crank-Nicolson and backward Euler
c5 = theta_scheme([0,1],0.1,101,1.0,5.0,0.5,[1,0],c0_fun)
c6 = theta_scheme([0,1],0.1,101,1.0,20,0.5,[1,0],c0_fun)
c7 = theta_scheme([0,1],0.1,101,1.0,10,1.0,[1,0],c0_fun)
c8 = theta_scheme([0,1],0.1,101,1.0,100,1.0,[1,0],c0_fun)


#Compute analytical solution
def analytical_dirichlet(t,x,k):
    #Initialize approximation
    Approx = 1 - x
    for n in range(1,k+1):
        Approx = Approx - (2.0/(n*np.pi))*np.sin(np.pi*n*x)*np.exp(-t*(n**2)*(np.pi**2))
    return Approx

#plotting
mesh = np.linspace(0,1,501)
c0 = c0_fun(mesh)

#Forward Euler Dirichlet BC
fig1 = plt.figure(num=1,figsize=(10,10))
ax1 = fig1.add_subplot(2,2,1)
ax1.plot(mesh,c0,label="Initial Condition")
ax1.plot(mesh, c1, label="Fo = 0.5")
ax1.plot(np.linspace(0,1,20),analytical_dirichlet(0.1,np.linspace(0,1,20),50),"x",label="Analytical")
ax1.legend()
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$c$")
ax2 = fig1.add_subplot(2,2,2)
ax2.plot(mesh,c0,label="Initial Condition")
ax2.plot(mesh,c2, label="Fo = 0.2")
ax2.plot(np.linspace(0,1,20),analytical_dirichlet(0.1,np.linspace(0,1,20),50),"x",label="Analytical")
ax2.legend()
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$c$")
ax3 = fig1.add_subplot(2,2,3)
ax3.plot(mesh,c0,label="Initial Condition")
ax3.plot(mesh,c3, label="Fo = 0.50005")
ax3.plot(np.linspace(0,1,20),analytical_dirichlet(0.1,np.linspace(0,1,20),50),"x",label="Analytical")
ax3.legend()
ax3.set_xlabel(r"$x$")
ax3.set_ylabel(r"$c$")
ax4 = fig1.add_subplot(2,2,4)
ax4.plot(mesh,c0,label="Initial Condition")
ax4.plot(np.linspace(0,1,101),c4, label="Fo = 0.5, Ncell = 101")
ax4.plot(np.linspace(0,1,20),analytical_dirichlet(0.1,np.linspace(0,1,20),50),"x",label="Analytical")
ax4.legend()
ax4.set_xlabel(r"$x$")
ax4.set_ylabel(r"$c$")

#Crank Nicolson & Backward Euler Dirichlet BC
mesh1 = np.linspace(0,1,101)
fig2 = plt.figure(num=2,figsize=(10,10))
ax5 = fig2.add_subplot(2,2,1)
ax5.plot(mesh,c0,label="Initial Condition")
ax5.plot(mesh1, c5, label="C-N,Fo = 5")
ax5.plot(np.linspace(0,1,20),analytical_dirichlet(0.1,np.linspace(0,1,20),50),"x",label="Analytical")
ax5.legend()
ax5.set_xlabel(r"$x$")
ax5.set_ylabel(r"$c$")
ax6 = fig2.add_subplot(2,2,2)
ax6.plot(mesh,c0,label="Initial Condition")
ax6.plot(mesh1,c6, label="C-N,Fo = 20")
ax6.plot(np.linspace(0,1,20),analytical_dirichlet(0.1,np.linspace(0,1,20),50),"x",label="Analytical")
ax6.legend()
ax6.set_xlabel(r"$x$")
ax6.set_ylabel(r"$c$")
ax7 = fig2.add_subplot(2,2,3)
ax7.plot(mesh,c0,label="Initial Condition")
ax7.plot(mesh1,c7, label="B-E,Fo = 10")
ax7.plot(np.linspace(0,1,20),analytical_dirichlet(0.1,np.linspace(0,1,20),50),"x",label="Analytical")
ax7.legend()
ax7.set_xlabel(r"$x$")
ax7.set_ylabel(r"$c$")
ax8 = fig2.add_subplot(2,2,4)
ax8.plot(mesh,c0,label="Initial Condition")
ax8.plot(mesh1,c8, label="B-E,Fo = 100")
ax8.plot(np.linspace(0,1,20),analytical_dirichlet(0.1,np.linspace(0,1,20),50),"x",label="Analytical")
ax8.legend()
ax8.set_xlabel(r"$x$")
ax8.set_ylabel(r"$c$")

"""
Neumann Boundary Conditions
"""
#Define c0_fun
def c0_fun2(x):
    c0 = 0.5 + 0.3*np.sin(2*np.pi*x)
    return c0

#Solve for different schemes
c9 = theta_scheme([0,1],0.1,101,1.0,0.5,0,"no-flux",c0_fun2)
c10 = theta_scheme([0,1],0.1,101,1.0,10,0.5,"no-flux",c0_fun2)

fig3 = plt.figure(num=3,figsize=(5,4))
ax9 = fig3.add_subplot(1,1,1)
ax9.plot(mesh1, c0_fun2(mesh1), label="Initial Condition")
ax9.plot(mesh1,c9,label="Forward Euler")
ax9.plot(mesh1,c10,"--",label="Crank-Nicolson")
ax9.legend()
ax9.set_xlabel(r"$x$")
ax9.set_ylabel(r"$c$")





plt.show()