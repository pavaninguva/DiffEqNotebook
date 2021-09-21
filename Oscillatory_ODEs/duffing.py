import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""
This script explores solutions to the Van der Pol equation:
x'' + d*x' + ax + bx^3 = 0
"""
#Plotting formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

# Define Van der Pol model:
def Duffing(t,y,a,b,d):
    x = y[0]
    y = y[1]
    #Define system of ODEs
    dxdt = y
    dydt = -d*y -a*x -b*x**3
    return [dxdt, dydt]

"""
Undamped 

we simply set d = 0
"""

sol1 = solve_ivp(lambda t,x:Duffing(t,x,-1.0,1.0,0.0),[0,20],[0.5,0.5],method="BDF",dense_output=True)
sol2 = solve_ivp(lambda t,x:Duffing(t,x,-1.0,1.0,0.0),[0,10],[1.0,2.0],method="BDF",dense_output=True)
sol3 = solve_ivp(lambda t,x:Duffing(t,x,-1.0,1.0,0.0),[0,10],[-2.0,-2.0],method="BDF",dense_output=True)

sol4 = solve_ivp(lambda t,x:Duffing(t,x,1.0,1.0,0.0),[0,20],[0.5,0.5],method="BDF",dense_output=True)
sol5 = solve_ivp(lambda t,x:Duffing(t,x,1.0,1.0,0.0),[0,10],[1.0,2.0],method="BDF",dense_output=True)
sol6 = solve_ivp(lambda t,x:Duffing(t,x,1.0,1.0,0.0),[0,10],[-2.0,-2.0],method="BDF",dense_output=True)

sol7 = solve_ivp(lambda t,x:Duffing(t,x,-10,10.0,0.0),[0,20],[0.5,0.5],method="BDF",dense_output=True)
sol8 = solve_ivp(lambda t,x:Duffing(t,x,-10,10.0,0.0),[0,10],[1.0,2.0],method="BDF",dense_output=True)
sol9 = solve_ivp(lambda t,x:Duffing(t,x,-10,10.0,0.0),[0,10],[-2.0,-2.0],method="BDF",dense_output=True)
sol10 = solve_ivp(lambda t,x:Duffing(t,x,-10,10.0,0.0),[0,10],[-0.5,-0.5],method="BDF",dense_output=True)

# Phase portrait
fig1 = plt.figure(num=1,figsize=(12,4))
ax1 = fig1.add_subplot(1,3,1)
ax1.plot(sol1.y[0],sol1.y[1],label=r"$(x_{0},y_{0}) = (0.5,0.5)$")
ax1.plot(sol2.y[0],sol2.y[1],label=r"$(x_{0},y_{0}) = (1.0,2.0)$")
ax1.plot(sol3.y[0],sol3.y[1],label=r"$(x_{0},y_{0}) = (-2.0,-2.0)$")
ax1.set_title(r"$\alpha = -1.0, \beta = 1.0 $")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$y$")
ax1.legend()
ax2 = fig1.add_subplot(1,3,2)
ax2.plot(sol4.y[0],sol4.y[1],label=r"$(x_{0},y_{0}) = (0.5,0.5)$")
ax2.plot(sol5.y[0],sol5.y[1],label=r"$(x_{0},y_{0}) = (1.0,2.0)$")
ax2.plot(sol6.y[0],sol6.y[1],label=r"$(x_{0},y_{0}) = (-2.0,-2.0)$")
ax2.set_title(r"$\alpha = 1.0, \beta = 1.0 $")
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$y$")
ax2.legend()
ax3 = fig1.add_subplot(1,3,3)
ax3.plot(sol7.y[0],sol7.y[1],label=r"$(x_{0},y_{0}) = (0.5,0.5)$")
ax3.plot(sol8.y[0],sol8.y[1],label=r"$(x_{0},y_{0}) = (1.0,2.0)$")
ax3.plot(sol9.y[0],sol9.y[1],label=r"$(x_{0},y_{0}) = (-2.0,-2.0)$")
ax3.plot(sol10.y[0],sol10.y[1],label=r"$(x_{0},y_{0}) = (-0.5,-0.5)$")
ax3.set_title(r"$\alpha = -10.0, \beta = 10.0 $")
ax3.set_xlabel(r"$x$")
ax3.set_ylabel(r"$y$")
ax3.legend()
fig1.tight_layout()

# Time series
fig2 = plt.figure(num=2,figsize=(8,6))
ax4 = fig2.add_subplot(3,1,1)
ax4.plot(sol1.t,sol1.y[0])
ax4.set_title(r"$\alpha = -1.0, \beta = 1.0, (x_{0},y_{0}) = (0.5,0.5) $")
ax4.set_xlabel(r"$t$")
ax4.set_ylabel(r"$x$")
ax5 = fig2.add_subplot(3,1,2)
ax5.plot(sol5.t,sol5.y[0])
ax5.set_title(r"$\alpha = 1.0, \beta = 1.0, (x_{0},y_{0}) = (1.0,2.0) $")
ax5.set_xlabel(r"$t$")
ax5.set_ylabel(r"$x$")
ax6 = fig2.add_subplot(3,1,3)
ax6.plot(sol8.t,sol8.y[0])
ax6.set_title(r"$\alpha = -10.0, \beta = 10.0, (x_{0},y_{0}) = (1.0,2.0) $")
ax6.set_xlabel(r"$t$")
ax6.set_ylabel(r"$x$")
fig2.tight_layout()

"""
Damped

We consider d = 1.0
"""

sol11 = solve_ivp(lambda t,x:Duffing(t,x,-1.0,1.0,1.0),[0,20],[0.5,0.5],method="BDF",dense_output=True)
sol12 = solve_ivp(lambda t,x:Duffing(t,x,-1.0,1.0,1.0),[0,10],[1.0,2.0],method="BDF",dense_output=True)
sol13 = solve_ivp(lambda t,x:Duffing(t,x,-1.0,1.0,1.0),[0,10],[-2.0,-2.0],method="BDF",dense_output=True)

sol14 = solve_ivp(lambda t,x:Duffing(t,x,1.0,1.0,1.0),[0,20],[0.5,0.5],method="BDF",dense_output=True)
sol15 = solve_ivp(lambda t,x:Duffing(t,x,1.0,1.0,1.0),[0,10],[1.0,2.0],method="BDF",dense_output=True)
sol16 = solve_ivp(lambda t,x:Duffing(t,x,1.0,1.0,1.0),[0,10],[-2.0,-2.0],method="BDF",dense_output=True)

sol17 = solve_ivp(lambda t,x:Duffing(t,x,-10,10.0,1.0),[0,20],[0.5,0.5],method="BDF",dense_output=True)
sol18 = solve_ivp(lambda t,x:Duffing(t,x,-10,10.0,1.0),[0,10],[1.0,2.0],method="BDF",dense_output=True)
sol19 = solve_ivp(lambda t,x:Duffing(t,x,-10,10.0,1.0),[0,10],[-2.0,-2.0],method="BDF",dense_output=True)
sol20 = solve_ivp(lambda t,x:Duffing(t,x,-10,10.0,1.0),[0,10],[-0.5,-0.5],method="BDF",dense_output=True)

# Phase portrait
fig3 = plt.figure(num=3,figsize=(12,4))
ax7 = fig3.add_subplot(1,3,1)
ax7.plot(sol11.y[0],sol11.y[1],label=r"$(x_{0},y_{0}) = (0.5,0.5)$")
ax7.plot(sol12.y[0],sol12.y[1],label=r"$(x_{0},y_{0}) = (1.0,2.0)$")
ax7.plot(sol13.y[0],sol13.y[1],label=r"$(x_{0},y_{0}) = (-2.0,-2.0)$")
ax7.set_title(r"$\alpha = -1.0, \beta = 1.0 , \delta = 1.0$")
ax7.set_xlabel(r"$x$")
ax7.set_ylabel(r"$y$")
ax7.legend()
ax8 = fig3.add_subplot(1,3,2)
ax8.plot(sol14.y[0],sol14.y[1],label=r"$(x_{0},y_{0}) = (0.5,0.5)$")
ax8.plot(sol15.y[0],sol15.y[1],label=r"$(x_{0},y_{0}) = (1.0,2.0)$")
ax8.plot(sol16.y[0],sol16.y[1],label=r"$(x_{0},y_{0}) = (-2.0,-2.0)$")
ax8.set_title(r"$\alpha = 1.0, \beta = 1.0 , \delta = 1.0$")
ax8.set_xlabel(r"$x$")
ax8.set_ylabel(r"$y$")
ax8.legend()
ax9 = fig3.add_subplot(1,3,3)
ax9.plot(sol17.y[0],sol17.y[1],label=r"$(x_{0},y_{0}) = (0.5,0.5)$")
ax9.plot(sol18.y[0],sol18.y[1],label=r"$(x_{0},y_{0}) = (1.0,2.0)$")
ax9.plot(sol19.y[0],sol19.y[1],label=r"$(x_{0},y_{0}) = (-2.0,-2.0)$")
ax9.plot(sol20.y[0],sol20.y[1],label=r"$(x_{0},y_{0}) = (-0.5,-0.5)$")
ax9.set_title(r"$\alpha = -10.0, \beta = 10.0 , \delta = 1.0$")
ax9.set_xlabel(r"$x$")
ax9.set_ylabel(r"$y$")
ax9.legend()
fig3.tight_layout()

# Time series
fig4 = plt.figure(num=4,figsize=(8,6))
ax10 = fig4.add_subplot(3,1,1)
ax10.plot(sol12.t,sol12.y[0])
ax10.set_title(r"$\alpha = -1.0, \beta = 1.0, \delta = 1.0,(x_{0},y_{0}) = (1.0,2.0) $")
ax10.set_xlabel(r"$t$")
ax10.set_ylabel(r"$x$")
ax11 = fig4.add_subplot(3,1,2)
ax11.plot(sol15.t,sol15.y[0])
ax11.set_title(r"$\alpha = 1.0, \beta = 1.0, \delta = 1.0,(x_{0},y_{0}) = (1.0,2.0) $")
ax11.set_xlabel(r"$t$")
ax11.set_ylabel(r"$x$")
ax12 = fig4.add_subplot(3,1,3)
ax12.plot(sol19.t,sol19.y[0])
ax12.set_title(r"$\alpha = -10.0, \beta = 10.0, \delta = 1.0, (x_{0},y_{0}) = (-2.0,-2.0) $")
ax12.set_xlabel(r"$t$")
ax12.set_ylabel(r"$x$")
fig4.tight_layout()


"""
Forced Duffing equation
"""

def Forced_Duffing(t,y,a,b,d,F,w):
    x = y[0]
    y = y[1]
    #Define system of ODEs
    dxdt = y
    dydt = -d*y -a*x -b*x**3 + F*np.cos(w*t)
    return [dxdt, dydt]

sol21 = solve_ivp(lambda t,x:Forced_Duffing(t,x,1.0,5.0,0.02,10,0.5),[0,50],[2.0,2.0],method="BDF",dense_output=True)

fig5 = plt.figure(num=5,figsize=(8,4))
ax13 = fig5.add_subplot(1,2,1)
ax13.plot(sol21.y[0],sol21.y[1])
ax13.set_xlabel(r"$x$")
ax13.set_ylabel(r"$y$")
ax14 = fig5.add_subplot(1,2,2)
ax14.plot(sol21.t,sol21.y[0])
ax14.set_xlabel(r"$t$")
ax14.set_ylabel(r"$x$")
fig5.tight_layout()



plt.show()