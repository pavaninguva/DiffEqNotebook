import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""
This script contains outlines how the SIR/SIRS and variant models 
can be solved:

Assumptions:
1. In the vanilla SIR model, those who have recovered are immune 
2. No death or births (this is called demographics) for some models
3. In the SIR with demographics model, a constant and identical 
birth/death coefficient k is assumed
4. In the SIR with demographics models, those born are deemed susceptible 

SIR Model:
dS/dt = -bSI
dI/dt = bSI - gI
dR/dt = gI

SIR with Demographics:
dS/dt = k(S+I+R) - bSI - kS
dI/dt = bSI - gI - kI
dR/dt = gI - kR

SIR with Demographics and Vaccination at Birth:
dS/dt = k(S+I+R)(1-p) -bSI -kS
dI/dt = bSI - gI - kI
dR/dt = gI - kR + k(S+I+R)p

SIRS Model:
dS/dt = -bSI + aR
dI/dt = bSI - gI
dR/dt = gI - aR



We note that without demographics: S + I + R = N / 1 
depending on whether you are considering the total population
or the population fraction. 

"""
#Plotting formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')


"""
SIR Model
""" 
def SIR(t,y,b,g):
    S = y[0]
    I = y[1]
    R = y[2]
    #Define ODEs
    dSdt = -b*S*I
    dIdt = b*S*I - g*I
    dRdt = g*I
    return [dSdt, dIdt, dRdt]

# Model parameters
b = 1.0
g = 0.5

x0 = [(0.9999),(0.0001),0]

#Solve ODE
sol = solve_ivp(lambda t,x:SIR(t,x,b,g),[0,200],x0,method="LSODA",dense_output=True)

#Plotting
fig1 = plt.figure(num=1,figsize=(5,4))
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(sol.t, sol.y[0],label=r"$S$")
ax1.plot(sol.t, sol.y[1],label=r"$I$")
ax1.plot(sol.t, sol.y[2],label=r"$R$")
ax1.legend()
ax1.set_xlabel(r"$t$")
ax1.set_ylabel("Fraction of Population")
fig1.tight_layout()


"""
SIR model with demographics
"""
def SIR_Demo(t,y,b,g,k):
    S = y[0]
    I = y[1]
    R = y[2]
    #Define ODEs
    dSdt = k*(S+I+R) -b*S*I -k*S
    dIdt = b*S*I - g*I -k*I
    dRdt = g*I -k*R
    return [dSdt, dIdt, dRdt]

# Parameters
k = 0.1

#Solve ODE
sol1 = solve_ivp(lambda t,x:SIR_Demo(t,x,b,g,k),[0,200],x0,method="LSODA",dense_output=True)

#Plotting
fig2 = plt.figure(num=2,figsize=(5,4))
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(sol1.t, sol1.y[0],label=r"$S$")
ax2.plot(sol1.t, sol1.y[1],label=r"$I$")
ax2.plot(sol1.t, sol1.y[2],label=r"$R$")
ax2.legend()
ax2.set_xlabel(r"$t$")
ax2.set_ylabel("Fraction of Population")
fig2.tight_layout()

"""
SIR with Demographics and Vaccination at Birth
"""
def SIR_Demo_Vacc(t,y,b,g,k,p):
    S = y[0]
    I = y[1]
    R = y[2]
    #Define ODEs
    dSdt = k*(S+I+R)*(1-p) -b*S*I -k*S
    dIdt = b*S*I - g*I -k*I
    dRdt = g*I -k*R + k*(S+I+R)*p
    return [dSdt, dIdt, dRdt]

#parameters
# p is the proportion of those born that are vaccinated at birth
p = 0.95

#Solve ODE
sol2 = solve_ivp(lambda t,x:SIR_Demo_Vacc(t,x,b,g,k,p),[0,200],x0,method="LSODA",dense_output=True)

#Plotting
fig3 = plt.figure(num=3,figsize=(5,4))
ax3 = fig3.add_subplot(1,1,1)
ax3.plot(sol2.t, sol2.y[0],label=r"$S$")
ax3.plot(sol2.t, sol2.y[1],label=r"$I$")
ax3.plot(sol2.t, sol2.y[2],label=r"$R$")
ax3.legend()
ax3.set_xlabel(r"$t$")
ax3.set_ylabel("Fraction of Population")
fig3.tight_layout()

"""
SIRS Model
"""

def SIRS(t,y,b,g,a):
    S = y[0]
    I = y[1]
    R = y[2]
    #Define ODEs
    dSdt = -b*S*I + a*R
    dIdt = b*S*I - g*I
    dRdt = g*I - a*R
    return [dSdt, dIdt, dRdt]

#parameters
a = 0.02

#Solve ODE
sol3 = solve_ivp(lambda t,x:SIRS(t,x,b,g,a),[0,200],x0,method="LSODA",dense_output=True)

#Plotting
fig4 = plt.figure(num=4,figsize=(5,4))
ax4 = fig4.add_subplot(1,1,1)
ax4.plot(sol3.t, sol3.y[0],label=r"$S$")
ax4.plot(sol3.t, sol3.y[1],label=r"$I$")
ax4.plot(sol3.t, sol3.y[2],label=r"$R$")
ax4.legend()
ax4.set_xlabel(r"$t$")
ax4.set_ylabel("Fraction of Population")
fig4.tight_layout()

plt.show()