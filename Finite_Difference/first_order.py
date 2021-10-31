import numpy as np


"""
This script implements various finite difference schemes 
to solve first order ODEs of the form

df/dx = g(x)
"""

def forward_1 (mesh, f0, g_x):
    N = len(mesh)
    # Construct matrix
    A = np.zeros((N,N))
    # Specify initial/boundary condition
    A[0,0] = 1
    for i in range(1,N):
        A[i,i] = 1
        A[i,i-1] = -1
    #Compute dx of mesh
    dx = np.zeros(N-1)
    for i in range(len(mesh)-1):
        dx[i] = mesh[i+1] - mesh[i]
    dx = np.concatenate(([1.0],dx))
    # Construct RHS
    b = np.zeros(N)
    b[0] = f0
    b[1:N] = g_x(mesh[0:N-1])
    b = b*dx

    return A,b


def central_2(mesh,f0,g_x):
    N = len(mesh)
    # Construct matrix
    A = np.zeros((N,N))
    # Specify initial/boundary condition
    A[0,0] = 1
    for i in range(1,N-1):
        A[i,i+1] = 1
        A[i,i-1] = -1
    #Specify backwards approx for last node
    A[-1,-3] = 1
    A[-1,-2] = -4
    A[-1,-1] = 3
    #Compute dx of mesh
    dx = np.zeros(N-1)
    for i in range(len(mesh)-1):
        dx[i] = 2*(mesh[i+1] - mesh[i])
    dx = np.concatenate(([1.0],dx))
    # Construct RHS
    b = np.zeros(N)
    b[0] = f0
    b[1:N] = g_x(mesh[1:N])
    b = b*dx

    return A,b
