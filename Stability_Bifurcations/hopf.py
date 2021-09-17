import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""
This script explores the normal cases for supercritical 
and subcritical Hopf bifurcations in 2-D systems
"""
#Plotting formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

"""
Supercritical Hopf Bifurcation
"""

def supercritical (t,y,mu):
    x = y[0]
    y = y[1]
    #Define system of ODEs
    dxdt = x*(mu -(x**2 + y**2)) -y
    dydt = y*(mu-(x**2 + y**2)) + x
    return [dxdt, dydt]


sol1 = solve_ivp(lambda t,x:supercritical(t,x,-1.0),[0,10],[0.5,0.5],method="LSODA",dense_output=True)
sol2 = solve_ivp(lambda t,x:supercritical(t,x,0.0),[0,20],[0.5,0.5],method="LSODA",dense_output=True)
sol3 = solve_ivp(lambda t,x:supercritical(t,x,1.0),[0,20],[0.1,0.1],method="LSODA",dense_output=True)

sol4 = solve_ivp(lambda t,x:supercritical(t,x,-1.0),[0,10],[1.5,1.5],method="LSODA",dense_output=True)
sol5 = solve_ivp(lambda t,x:supercritical(t,x,0.0),[0,20],[1.5,1.5],method="LSODA",dense_output=True)
sol6 = solve_ivp(lambda t,x:supercritical(t,x,1.0),[0,20],[2.0,2.0],method="LSODA",dense_output=True)


#Plotting Results

#Phase portrait
fig1 = plt.figure(num=1,figsize=(10,4))
ax1 = fig1.add_subplot(1,3,1)
ax1.plot(sol1.y[0],sol1.y[1],label=r"$(x_{0},y_{0}) = (0.5,0.5)$")
ax1.plot(sol4.y[0],sol4.y[1],label=r"$(x_{0},y_{0}) = (1.5,1.5)$")
ax1.set_title(r"$\mu = -1.0$")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$y$")
ax1.legend()
ax2 = fig1.add_subplot(1,3,2)
ax2.plot(sol2.y[0],sol2.y[1],label=r"$(x_{0},y_{0}) = (0.5,0.5)$")
ax2.plot(sol5.y[0],sol5.y[1],label=r"$(x_{0},y_{0}) = (1.5,1.5)$")
ax2.set_title(r"$\mu = 0$")
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$y$")
ax2.legend()
ax3 = fig1.add_subplot(1,3,3)
ax3.plot(sol3.y[0],sol3.y[1],label=r"$(x_{0},y_{0}) = (0.1,0.1)$")
ax3.plot(sol6.y[0],sol6.y[1],label=r"$(x_{0},y_{0}) = (2.0,2.0)$")
ax3.set_title(r"$\mu = 1.0$")
ax3.set_xlabel(r"$x$")
ax3.set_ylabel(r"$y$")
ax3.legend()
fig1.tight_layout()

#Time series
fig2 = plt.figure(num=2,figsize=(8,6))
ax4 = fig2.add_subplot(3,1,1)
ax4.plot(sol1.t,sol1.y[0],label=r"$x$")
ax4.plot(sol1.t,sol1.y[1],label=r"$y$")
ax4.set_title(r"$\mu = -1.0, (x_{0},y_{0}) = (0.5,0.5)$")
ax4.set_xlabel(r"$t$")
ax4.set_ylabel(r"$\mathbf{x}$")
ax4.legend()
ax5 = fig2.add_subplot(3,1,2)
ax5.plot(sol2.t,sol2.y[0],label=r"$x$")
ax5.plot(sol2.t,sol2.y[1],label=r"$y$")
ax5.set_title(r"$\mu = 0, (x_{0},y_{0}) = (0.5,0.5)$")
ax5.set_xlabel(r"$t$")
ax5.set_ylabel(r"$\mathbf{x}$")
ax5.legend()
ax6 = fig2.add_subplot(3,1,3)
ax6.plot(sol3.t,sol3.y[0],label=r"$x$")
ax6.plot(sol3.t,sol3.y[1],label=r"$y$")
ax6.set_title(r"$\mu = 1.0, (x_{0},y_{0}) = (0.1,0.1)$")
ax6.set_xlabel(r"$t$")
ax6.set_ylabel(r"$\mathbf{x}$")
ax6.legend()

fig2.tight_layout()


"""
Subcritical Hopf Bifurcation
"""

def subcritical (t,y,mu):
    x = y[0]
    y = y[1]
    #Define system of ODEs
    dxdt = x*(mu +(x**2 + y**2)) -y
    dydt = y*(mu +(x**2 + y**2)) + x
    return [dxdt, dydt]

sol7 = solve_ivp(lambda t,x:subcritical(t,x,-1.0),[0,10],[0.5,0.5],method="BDF",dense_output=True)
sol8 = solve_ivp(lambda t,x:subcritical(t,x,0.0),[0,24.5],[0.1,0.1],method="BDF",dense_output=True)
sol9 = solve_ivp(lambda t,x:subcritical(t,x,1.0),[0,4.0],[0.01,0.01],method="BDF",dense_output=True)

sol10 = solve_ivp(lambda t,x:subcritical(t,x,-1.0),[0,0.2],[1.1,1.1],method="LSODA",dense_output=True)
sol11 = solve_ivp(lambda t,x:subcritical(t,x,0.0),[0,0.1],[1.5,1.5],method="LSODA",dense_output=True)
sol12 = solve_ivp(lambda t,x:subcritical(t,x,1.0),[0,0.1],[1.1,1.1],method="LSODA",dense_output=True)

sol13 = solve_ivp(lambda t,x:subcritical(t,x,-1.0),[0,6],[1.0,0.5e-14],method="BDF",dense_output=True, atol=1e-16)
#Plotting Results

#Phase portrait
fig3 = plt.figure(num=3,figsize=(10,4))
ax7 = fig3.add_subplot(1,3,1)
ax7.plot(sol7.y[0],sol7.y[1],label=r"$(x_{0},y_{0}) = (0.5,0.5)$")
ax7.plot(sol10.y[0],sol10.y[1],label=r"$(x_{0},y_{0}) = (1.1,1.1)$")
ax7.plot(sol13.y[0], sol13.y[1], label=r"$(x_{0},y_{0}) = (1.0,0.0)$")
ax7.set_title(r"$\mu = -1.0$")
ax7.set_xlabel(r"$x$")
ax7.set_ylabel(r"$y$")
ax7.legend()
ax8 = fig3.add_subplot(1,3,2)
ax8.plot(sol8.y[0],sol8.y[1],label=r"$(x_{0},y_{0}) = (0.1,0.1)$")
ax8.plot(sol11.y[0],sol11.y[1],label=r"$(x_{0},y_{0}) = (1.5,1.5)$")
ax8.set_title(r"$\mu = 0$")
ax8.set_xlabel(r"$x$")
ax8.set_ylabel(r"$y$")
ax8.legend()
ax9 = fig3.add_subplot(1,3,3)
ax9.plot(sol9.y[0],sol9.y[1],label=r"$(x_{0},y_{0}) = (0.01,0.01)$")
ax9.plot(sol12.y[0],sol12.y[1],label=r"$(x_{0},y_{0}) = (1.1,1.1)$")
ax9.set_title(r"$\mu = 1.0$")
ax9.set_xlabel(r"$x$")
ax9.set_ylabel(r"$y$")
ax9.legend()
fig3.tight_layout()

#Time series
fig4 = plt.figure(num=4,figsize=(8,6))
ax10 = fig4.add_subplot(3,1,1)
ax10.plot(sol7.t,sol7.y[0],label=r"$x$")
ax10.plot(sol7.t,sol7.y[1],label=r"$y$")
ax10.set_title(r"$\mu = -1.0, (x_{0},y_{0}) = (0.5,0.5)$")
ax10.set_xlabel(r"$t$")
ax10.set_ylabel(r"$\mathbf{x}$")
ax10.legend()
ax11 = fig4.add_subplot(3,1,2)
ax11.plot(sol8.t,sol8.y[0],label=r"$x$")
ax11.plot(sol8.t,sol8.y[1],label=r"$y$")
ax11.set_title(r"$\mu = 0, (x_{0},y_{0}) = (0.1,0.1)$")
ax11.set_xlabel(r"$t$")
ax11.set_ylabel(r"$\mathbf{x}$")
ax11.legend()
ax12 = fig4.add_subplot(3,1,3)
ax12.plot(sol9.t,sol9.y[0],label=r"$x$")
ax12.plot(sol9.t,sol9.y[1],label=r"$y$")
ax12.set_title(r"$\mu = 1.0, (x_{0},y_{0}) = (0.01,0.01)$")
ax12.set_xlabel(r"$t$")
ax12.set_ylabel(r"$\mathbf{x}$")
ax12.legend()

fig4.tight_layout()



plt.show()