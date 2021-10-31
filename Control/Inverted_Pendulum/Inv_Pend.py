""" 
This script implements a dynamic model for an inverted pendulum 

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
m_p = 1.0 # Mass of pendulum
m_c = 5.0 # Mass of cart
g = 9.81 # Gravity
l = 1.0 # Length of Pendulum Arm

# Set up model

def Inverted_Pendulum(t,X):
    theta, theta_dot, x, x_dot = X
    #Define ODEs
    dtheta_dt = theta_dot

