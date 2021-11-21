import numpy as np

"""
This script implements various schemes for the 1-D 
diffusion equation given by 

1-D: dc/dt = D d2c/dx2 
"""

def theta_scheme(xrange, t_final, Ncells, D, Fo, theta, bcs, c0_fun):
    # Compute dt using mesh Fo
    dx = (xrange[1] - xrange[0])/(Ncells-1)
    dt = ((Fo*dx**2)/D)**0.5

    if theta == 1:
        print("Backward Euler")
    elif theta == 0.5:
        print("Crank-Nicolson")
    elif theta == 0:
        print("Forward Euler")

    # Form matrix
    A = np.zeros((Ncells,Ncells))
    A_ = np.zeros((Ncells,Ncells))

    if bcs == "no-flux":
        A[0,0] = 1+ 2*theta*Fo
        A[0,1] = -2*theta*Fo
        A[-1,-1] = 1+ 2*theta*Fo
        A[-1,-2] = -2*theta*Fo

        A_[0,0] = 1 -2*(1-theta)*Fo
        A_[0,1] = 2*(1-theta)*Fo
        A[-1,-1] = 1 -2*(1-theta)*Fo
        A[-1,-2] = 2*(1-theta)*Fo
    else:
        A[0,0] = 1
        A[-1,-1] = 1

        A_[0,0] = 1
        A_[-1,-1] = 1

    for i in range(1,Ncells-1):
        A[i,i] = 1 + 2*theta*Fo
        A[i,i-1] = -theta*Fo
        A[i,i+1] = -theta*Fo

        A_[i,i] = 1 - 2*(1-theta)*Fo
        A_[i,i-1] = (1-theta)*Fo
        A_[i,i+1] = (1-theta)*Fo

    #Initialize c0 and c
    c_old = c0_fun(np.linspace(xrange[0], xrange[1], Ncells))
    if bcs != "no-flux": 
        c_old[0] = bcs[0]
        c_old[-1] = bcs[1]
    c_new = np.zeros(Ncells)

    # Perform timestepping
    t = 0.0
    counter = 0
    while t < t_final - 1e-16:

        #Compute LHS
        b = A_.dot(c_old)
        #Invert A to get c_new
        c_new = np.linalg.solve(A,b)
        #Update c_old
        c_old = c_new
        #Update t and counter
        t += dt
        counter += 1

    return c_new

        

