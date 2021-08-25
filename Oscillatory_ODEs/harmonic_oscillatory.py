import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



""""
This script contains functions to solve three variants of the simple harmonic oscillator:
1. Undamped Harmonic Oscillator given by x'' + (k/m)x = 0.
x is the displacement, k is the spring constant and m is the mass of the object

2. Damped Harmonic Oscillator given by x'' + (b/m)x' + (k/m)x = 0.
b is the friction coefficient and is taken as a constant. 

3. Damped Harmonic Oscillator with a cosine driving force given by x'' + (b/m)x' + (k/m)x = (F0/m)cos(wt)
F0*cos(wt) is the driving force with F0 and w are the free parameters. 
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
# Define Damped Harmonic Oscillator ODE Function
def Damped(t,y,w_0,xi):
    x = y[0]
    xdot = y[1]
    #Define system of ODEs
    dxdt = xdot
    dxdotdt = -(w_0**2)*x -2*xi*w_0*xdot
    return [dxdt, dxdotdt]

def Overdamped_Ana(t,x_0, w_0,xi):
    #Unpack initial conditions
    x0 = x_0[0]
    xdot0 = x_0[1]
    # Compute solution
    x = (
        ((x0/2) - (xdot0/(2*w_0*np.sqrt(xi**2 - 1.0))) - ((xi*x0)/(2*np.sqrt(xi**2 - 1.0))))*np.exp(-w_0*(xi + np.sqrt(xi**2 -1.0))*t) 
        + ((x0/2) + (xdot0/(2*w_0*np.sqrt(xi**2 - 1.0))) + ((xi*x0)/(2*np.sqrt(xi**2 - 1.0))))*np.exp(-w_0*(xi - np.sqrt(xi**2 -1.0))*t) 
    )
    return x 

def Critdamped_Ana(t,x_0,w_0):
    #Unpack initial conditions
    x0 = x_0[0]
    xdot0 = x_0[1]
    # Compute solution
    x = ((xdot0+ w_0*x0)*t + x0)*np.exp(-w_0*t)
    return x

def Underdamped_Ana(t,x_0,w_0,xi):
    #Unpack initial conditions
    x0 = x_0[0]
    xdot0 = x_0[1]
    #Compute solution
    x = (np.exp(-xi*w_0*t))*(x0*np.cos(w_0*np.sqrt(1.0-xi**2)*t) + 
    ((xdot0 + xi*w_0*x0)/(w_0*np.sqrt(1-xi**2)))*np.sin(w_0*np.sqrt(1.0-xi**2)*t))
    return x

# Define xi and use previously defined params w_0 and time
xi_1 = 0.2
xi_2 = 1.0
xi_3 = 1.5

#Solve the different cases
sol2 = solve_ivp(lambda t,x:Damped(t,x,w_0,xi_1),[0,10],x_0,method="LSODA",dense_output=True)
sol3 = solve_ivp(lambda t,x:Damped(t,x,w_0,xi_2),[0,10],x_0,method="LSODA",dense_output=True)
sol4 = solve_ivp(lambda t,x:Damped(t,x,w_0,xi_3),[0,10],x_0,method="LSODA",dense_output=True)

# Plotting
fig2 = plt.figure(num=2,figsize=(4,4))
ax4 = fig2.add_subplot(1,1,1)
ax4.plot(sol2.t, sol2.y[0],color="k", label=r"$\xi=0.2$")
ax4.plot(sol3.t, sol3.y[0],color="r", label=r"$\xi=1.0$")
ax4.plot(sol4.t, sol4.y[0],color="b", label=r"$\xi=1.5$")
# ax4.plot(np.linspace(0,10,50), Overdamped_Ana(np.linspace(0,10,50),x_0,w_0,xi_3), "x")
# ax4.plot(np.linspace(0,10,50), Critdamped_Ana(np.linspace(0,10,50),x_0,w_0), "x")
# ax4.plot(np.linspace(0,10,50), Underdamped_Ana(np.linspace(0,10,50),x_0,w_0,xi_1), "x")
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

"""
Case 3
"""

# Steady state amplitude 
def amplitude(F0, m, w0, w, xi):
    A = (F0/m)*(1/np.sqrt((w0**2 - w**2)**2 + 4*(xi**2)*(w0**2)*(w**2)))
    return A

#Compute for different values of w and xi
case1 = amplitude(1,1,1,np.linspace(0,5,1000),0.05)
case2 = amplitude(1,1,1,np.linspace(0,5,1000),0.2)
case3 = amplitude(1,1,1,np.linspace(0,5,1000),0.5)
case4 = amplitude(1,1,1,np.linspace(0,5,1000),1.0)
case5 = amplitude(1,1,1,np.linspace(0,5,1000),2.0)


#Plotting
fig4 = plt.figure(num=4, figsize=(4,4))
ax8 = fig4.add_subplot(1,1,1)
ax8.plot(np.linspace(0,5,1000),case1, label=r"$\xi=0.05$")
ax8.plot(np.linspace(0,5,1000),case2, label=r"$\xi=0.2$")
ax8.plot(np.linspace(0,5,1000),case3, label=r"$\xi=0.5$")
ax8.plot(np.linspace(0,5,1000),case4, label=r"$\xi=1.0$")
ax8.plot(np.linspace(0,5,1000),case5, label=r"$\xi=2.0$")
ax8.legend()
ax8.set_xlabel(r"$\omega$")
ax8.set_ylabel("Amplitude")

#Performing numerical solution
# Define Forced Harmonic Oscillator ODE Function
def Forced(t,y,w_0,xi,F0,m,w):
    x = y[0]
    xdot = y[1]
    #Define system of ODEs
    dxdt = xdot
    dxdotdt = -(w_0**2)*x -2*xi*w_0*xdot +(F0/m)*np.cos(w*t)
    return [dxdt, dxdotdt]

sol5 = solve_ivp(lambda t,x:Forced(t,x,1.0,0.01,1.0,1.0,1.0),[0,100],x_0,method="LSODA",dense_output=True)
sol6 = solve_ivp(lambda t,x:Forced(t,x,1.0,0.01,1.0,1.0,2.0),[0,100],x_0,method="LSODA",dense_output=True)
sol7 = solve_ivp(lambda t,x:Forced(t,x,1.0,1.0,1.0,1.0,1.0),[0,100],x_0,method="LSODA",dense_output=True)

fig5 = plt.figure(num=5,figsize=(8,8))
ax9 = fig5.add_subplot(3,1,1)
ax9.plot(sol5.t, sol5.y[0],color="k")
ax9.set_xlabel(r"$t$")
ax9.set_ylabel(r"$x$")
ax9.title.set_text(r"$w_{0} = 1.0, \xi=0.01, F_{0}=1.0, m=1.0, \omega=1.0$")
ax10 = fig5.add_subplot(3,1,2)
ax10.plot(sol6.t, sol6.y[0],color="k")
ax10.set_xlabel(r"$t$")
ax10.set_ylabel(r"$x$")
ax10.title.set_text(r"$w_{0} = 1.0, \xi=0.01, F_{0}=1.0, m=1.0, \omega=2.0$")
ax11 = fig5.add_subplot(3,1,3)
ax11.plot(sol7.t, sol7.y[0],color="k")
ax11.set_xlabel(r"$t$")
ax11.set_ylabel(r"$x$")
ax11.title.set_text(r"$w_{0} = 1.0, \xi=1.0, F_{0}=1.0, m=1.0, \omega=1.0$")
fig5.tight_layout()

"""
Final Example
"""

def Example_Ana(t):
    # Particular solution
    xp = (1/(2*np.sqrt(13)))*np.cos(2*t - np.arctan(2/3))

    # Transient solution
    c1 = 5 - (1/(2*np.sqrt(13)))*np.cos(np.arctan(2/3))
    c2 = (5/3) - (1/(6*np.sqrt(13)))*np.cos(np.arctan(2/3)) - (1/(3*np.sqrt(13)))*np.sin(np.arctan(2/3))

    xt = np.exp(-t)*(c1*np.cos(3*t) + c2*np.sin(3*t))

    # Full solution
    x = xt + xp

    return x

sol_eg = solve_ivp(lambda t,x:Forced(t,x,np.sqrt(10),(1/np.sqrt(10)),1.0,1.0,2),[0,50],x_0,method="LSODA",dense_output=True)

#plotting
fig6 = plt.figure(num=6,figsize=(8,3))
ax12 = fig6.add_subplot(1,1,1)
ax12.plot(sol_eg.t, sol_eg.y[0],color="k",label="Numerical")
ax12.plot(np.linspace(0,50,20), Example_Ana(np.linspace(0,50,20)),"x" , label="Analytical")
ax12.plot
ax12.set_xlabel(r"$t$")
ax12.set_ylabel(r"$x$")
ax12.legend()
fig6.tight_layout()

plt.show()