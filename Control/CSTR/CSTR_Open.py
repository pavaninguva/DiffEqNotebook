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
from scipy.integrate import solve_ivp

#Plotting formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')


#Constants and parameters
Ea  = 72750     # activation energy J/mol
R   = 8.314     # gas constant J/mol K
k0  = 7.2e10    # Arrhenius rate constant [1/min]
V   = 100.0     # Volume [L]
rho = 1000.0    # Density [g/L]
Cp  = 0.239     # Heat capacity [J/g K]
dHr = -5.0e4    # Enthalpy of reaction [J/mol]
UA  = 5.0e4     # Heat transfer [J/min K]
F = 100.0       # Flowrate [L/min]
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
def CSTR(t,X):
    C,T,Tc = X
    dCdt = (F/V)*(Cf - C) - k(T)*C
    dTdt = (F/V)*(Tf - T) + (-dHr/(rho*Cp))*k(T)*C + (UA/(rho*Cp*V))*(Tc-T)
    dTcdt = (qc/Vc)*(Tcf-Tc) + (UA/(rho*Cp*Vc))*(T-Tc)

    return [dCdt, dTdt, dTcdt]

#Create a plotting function
def plotter (X):
    #First plot is for C vs t
    plt.subplot(1,2,1)
    plt.plot(X.t,X.y[0])
    plt.xlabel("Time/min")
    plt.ylabel("Concentration/M")
    plt.title("Reactor Concentration")
    # plt.ylim(0,1)

    plt.subplot(1,2,2)
    plt.plot(X.t,X.y[1])
    plt.xlabel("Time/min")
    plt.ylabel("Temperature/K")
    plt.title("Reactor Temperature")
    # plt.ylim(300,520)

#Run simulation
# Initial conditions
IC = [C0,T0,Tcf]

# Vector of different qc
qList = np.linspace(20,200,10)

#Solving model equations
plt.figure(figsize=(10,4))
for qc in qList:
    X = solve_ivp(CSTR,[0,4],IC,method="BDF")
    plotter(X)

plt.legend(qList, ncol=2, bbox_to_anchor=(0.4,0.54))
plt.show()
plt.tight_layout()