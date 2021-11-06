import numpy as np

"""
This script implements various schemes for the 1-D
and 2-D advection equation given by 

1-D: dc/dt + u_x dc/dx = 0
2-D: dc/dt + u_x dc/dx + u_y dc/dy = 0
"""

def explict_upwind_1d(xrange, t_final, Ncells,u, CFL, c0_fun):
    #Compute dt using CFL condition
    dx = (xrange[1])/(Ncells -1)
    dt = (CFL*dx)/(u)

    # Form matrix
    A = np.zeros((Ncells,Ncells))
    A[0,0] = 1 - CFL
    for i in range(1,Ncells):
        A[i,i] = 1 - CFL
        A[i,i-1] = CFL 

    # Initialize c0 and c
    c_old = c0_fun(np.linspace(xrange[0], xrange[1], Ncells))
    c_new = np.zeros(Ncells)

    # Perform time-stepping
    t = 0.0
    counter = 0
    while t < t_final - 1e-16:
        # Compute c_new
        c_new = A.dot(c_old)
        #update c_old
        c_old = c_new
        #Update t and counter
        t += dt
        counter += 1
    
    return c_new


def lax_wendroff_1d(xrange, t_final, Ncells,u, CFL, c0_fun):
    #Compute dt using CFL condition
    dx = (xrange[1])/(Ncells -1)
    dt = (CFL*dx)/(u)

    # Form matrix
    A = np.zeros((Ncells,Ncells))
    #First row
    A[0,0] = 1 - CFL**2
    A[0,1] = (CFL/2)*(CFL -1)
    #Body
    for i in range(1,Ncells-1):
        A[i,i] = 1 - CFL**2
        A[i,i-1] = (CFL/2)*(CFL + 1)
        A[i,i+1] = (CFL/2)*(CFL -1)
    #Last row
    A[-1,-1] = 1-CFL**2
    A[-1,-2] = CFL**2

    # Initialize c0 and c
    c_old = c0_fun(np.linspace(xrange[0], xrange[1], Ncells))
    c_new = np.zeros(Ncells)

    # Perform time-stepping
    t = 0.0
    counter = 0
    while t < t_final - 1e-16:
        # Compute c_new
        c_new = A.dot(c_old)
        #update c_old
        c_old = c_new
        #Update t and counter
        t += dt
        counter += 1
    
    return c_new

def explicit_FTCS_1d(xrange, t_final, Ncells,u, CFL, c0_fun):
    #Compute dt using CFL condition
    dx = (xrange[1])/(Ncells -1)
    dt = (CFL*dx)/(u)

    # Form matrix
    A = np.zeros((Ncells,Ncells))
    #First row
    A[0,0] = 1 
    A[0,1] = -(CFL/2)
    #Body
    for i in range(1,Ncells-1):
        A[i,i] = 1 
        A[i,i-1] = (CFL/2)
        A[i,i+1] = -(CFL/2)
    #Last row
    A[-1,-1] = 1

    # Initialize c0 and c
    c_old = c0_fun(np.linspace(xrange[0], xrange[1], Ncells))
    c_new = np.zeros(Ncells)

    # Perform time-stepping
    t = 0.0
    counter = 0
    while t < t_final - 1e-16:
        # Compute c_new
        c_new = A.dot(c_old)
        #update c_old
        c_old = c_new
        #Update t and counter
        t += dt
        counter += 1
    
    return c_new


def implicit_upwind_1d(xrange, t_final, Ncells,u, CFL, c0_fun):
    #Compute dt using CFL condition
    dx = (xrange[1])/(Ncells -1)
    dt = (CFL*dx)/(u)

    # Form matrix
    A = np.zeros((Ncells,Ncells))
    A[0,0] = 1 + CFL
    for i in range(1,Ncells):
        A[i,i] = 1 + CFL
        A[i,i-1] = -CFL 

    # Initialize c0 and c
    c_old = c0_fun(np.linspace(xrange[0], xrange[1], Ncells))
    c_new = np.zeros(Ncells)

    # Perform time-stepping
    t = 0.0
    counter = 0
    while t < t_final - 1e-16:
        # Compute c_new
        c_new = np.linalg.solve(A,c_old)
        #update c_old
        c_old = c_new
        #Update t and counter
        t += dt
        counter += 1
    
    return c_new