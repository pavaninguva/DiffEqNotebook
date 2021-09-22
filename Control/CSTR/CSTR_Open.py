""" 
This script implements a dynamic model for a non-isothermal CSTR 

Three variables are tracked: 
C: Reactant concentration
T: Reactor temperature
Tc: Coolant temperature

This script considers the dynamic behaviour of the reactor for 
different cooling water flowrates
"""

#Packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

#Constants and parameters
Ea  = 72750     # activation energy J/mol
R   = 8.314     # gas constant J/mol K
k0  = 7.2e10    # Arrhenius rate constant [1/min]
V   = 100.0     # Volume [L]
rho = 1000.0    # Density [g/L]
Cp  = 0.239     # Heat capacity [J/g K]
dHr = -5.0e4    # Enthalpy of reaction [J/mol]
UA  = 5.0e4     # Heat transfer [J/min K]
q = 100.0       # Flowrate [L/min]
Cf = 1.0        # Inlet feed concentration [mol/L]
Tf  = 300.0     # Inlet feed temperature [K]
C0 = 0.5        # Initial concentration [mol/L]
T0  = 350.0;    # Initial temperature [K]
Tcf = 300.0     # Coolant feed temperature [K]
qc = 50.0       # Nominal coolant flowrate [L/min]
Vc = 20.0       # Cooling jacket volume [L]

#Define model equations
#Rate expression
def k(T):
    return k0*np.exp(-Ea/(R*T))

#System of ODEs
def model(X,t):
    C,T,Tc = X
    dCdt = (q/V)*(Cf - C) - k(T)*C
    dTdt = (q/V)*(Tf - T) + (-dHr/(rho*Cp))*k(T)*C + (UA/(rho*Cp*V))*(Tc-T)
    dTcdt = (qc/Vc)*(Tcf-Tc) + (UA/(rho*Cp*Vc))*(T-Tc)

    return [dCdt, dTdt, dTcdt]

#Create a plotting function
def plotter (t,X):
    #First plot is for C vs t
    plt.subplot(1,2,1)
    plt.plot(t,X[:,0])
    plt.xlabel("Time/min")
    plt.ylabel("mol/L")
    plt.title("Reactor Concentration")
    plt.ylim(0,1)

    plt.subplot(1,2,2)
    plt.plot(t,X[:,1])
    plt.xlabel("Time/min")
    plt.ylabel("K")
    plt.title("Reactor Temperature")
    plt.ylim(300,520)

#Run simulation
# Initial conditions
IC = [C0,T0,Tcf]

#Time vector
t = np.linspace(0,4.0,1000)

# Vector of different qc
qList = np.linspace(0,200,11)

#Solving model equations
plt.figure()
for qc in qList:
    X = odeint(model,IC,t)
    plotter(t,X)

plt.legend(qList)
plt.show()
plt.tight_layout()