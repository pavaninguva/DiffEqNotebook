""" 
This script implements a dynamic model for a non-isothermal CSTR 

Three variables are tracked: 
C: Reactant concentration
T: Reactor temperature
Tc: Coolant temperature

This script also implements a PID controller using 
cooling water flowrate to control reactor temperature
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
qc = 50.0
Vc = 20.0       # Cooling jacket volume [L]

# Set up model
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

# Coolant saturation
qc_min = 0.0
qc_max = 500.0

def sat(qc):
    qc = max(qc_min,min(qc,qc_max))
    return qc

#Create a plotting function

def qplot(log):
    log = np.asarray(log).T
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(log[0],log[1])
    plt.title('Concentration')
    plt.ylabel('moles/liter')
    plt.xlabel('Time [min]')

    plt.subplot(1,3,2)
    plt.plot(log[0],log[2],log[0],log[3])
    plt.title('Temperature')
    plt.ylabel('Kelvin')
    plt.xlabel('Time [min]')
    plt.legend(['Reactor','Cooling Jacket'])

    plt.subplot(1,3,3)
    plt.plot(log[0],log[4])
    plt.title('Cooling Water Flowrate')
    plt.ylabel('liters/min')
    plt.xlabel('Time [min]')
    plt.tight_layout()

#Run simulation
# Initial conditions
IC = [C0,T0,Tcf]

#Simulation parameters
ti = 0.0
tf = 10.0
dt = 0.01

kp = 50.0
ki = 30.0
kd = 0.0

# Define set point
def set_point(t):
    if t < 5:
        T = 400
    else:
        T = 400
    return T

# Initialize error and variable values
err = set_point(0) - T0
err_old = err
err_old_old = err

qc_old = qc
qc_old_old = qc

C,T,Tc = IC

log = []

for t in np.linspace(ti, tf, int((tf-ti)/dt)):
    #Calculate Error
    err = set_point(t) - T
    #Calculate Controller Output
    dqc = -(kp*(err-err_old) + ki*dt*err + (kd/dt)*(err - 2*err_old + err_old_old))
    #Enforce saturation limits
    qc_new = sat(dqc+qc)

    # Moving average filter
    qc = (1/3)*(qc_new + qc_old + qc_old_old)
    # qc = qc_new

    #Store values at current time:
    log.append([t,C,T,Tc,qc,err])

    #Advance forward in time by dt
    sol = solve_ivp(CSTR, [t,t+dt] ,[C,T,Tc], method="BDF")
    C = sol.y[0][-1]
    T = sol.y[1][-1]
    Tc = sol.y[2][-1]

    #Update errors
    err_old_old = err_old
    err_old = err

    qc_old_old = qc_old
    qc_old = qc
    

# Plotting Results
qplot(log)
    
plt.show()