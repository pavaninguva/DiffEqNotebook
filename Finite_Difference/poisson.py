import numpy as np
import matplotlib.pyplot as plt

"""
This script implements a solver to solve the Poisson PDE
in 2-D on a cartesian grid

d2c/dx2 + d2c/dy2  = f(x,y),

where c is the variable of interest and
f is an arbitrary function f(x,y)
"""

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

def poisson_scheme_2d(xrange,yrange,nx,ny,bcs, f_fun):
    # bcs is an array with 4 elements
    # bcs[0,1,2,3] = bc_left, bc_top,bc_right,bc_bottom
    
    #compute dx and dy
    dx = (xrange[1] - xrange[0])/(nx-1)
    dy = (yrange[1] -yrange[0])/(ny-1)

    #Extract BCs
    bc_left = bcs[0]
    bc_top = bcs[1]
    bc_right = bcs[2]
    bc_bottom = bcs[3]

    #Evaluate mesh
    xvals = np.linspace(xrange[0],xrange[1],nx)
    yvals = np.linspace(yrange[0],yrange[1],ny)
    xx,yy = np.meshgrid(xvals,yvals)

    #Compute RHS 
    b_array = f_fun(xx,yy)
    b = np.zeros(nx*ny)
    for j in range(nx):
        for k in range(ny):
            p = j + (k-1)*nx
            b[p] = b_array[j,k] 

    #Form matrix
    A = np.zeros((nx*ny,nx*ny))
    for j in range(nx):
        for k in range(ny):
            #Compute p index
            p = j + (k-1)*nx
            #Enforce BCs:
            #Left BC
            if j == 0:
                A[p,p] = 1
                b[p] = bc_left
            #Right BC    
            elif j == nx -1:
                A[p,p] = 1
                b[p] = bc_right
            #Bottom BC
            elif k == 0:
                A[p,p] = 1
                b[p] = bc_bottom
            #Top BC
            elif k == ny-1:
                A[p,p] = 1
                b[p] = bc_top
            else:
                A[p,p] = -2*((1/dx**2) + (1/dy**2))
                A[p,p+1] = 1/dx**2
                A[p,p-1] = 1/dx**2
                A[p,p+nx] = 1/dy**2
                A[p,p-nx] = 1/dy**2

    # Solve for C
    c_vec = np.linalg.solve(A,b)

    #Reshape c_vec to output array
    c_out = c_vec.reshape(nx,-1)

    return xvals, yvals, c_out


# Solve Laplace Equation with Dirichlet BCs

def f_laplace(x,y):
    nx = len(x)
    ny = len(y)
    f = np.zeros((nx,ny))
    return f

xvals,yvals,phi = poisson_scheme_2d([0,1],[0,1], 101,101,[0,1,0,0], f_laplace)

#Plotting
fig1 = plt.figure(num=1,figsize=(5,4))
ax1 = fig1.add_subplot(1,1,1)
im = ax1.pcolormesh(xvals,yvals,phi, cmap="jet")
fig1.colorbar(im,label=r"$\phi$")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$y$")
fig1.tight_layout()


# Plot analytical
def analytical_expression(k,x,y):
    # Initialize approximation
    Approx = 0
    for mode in range(0,k):
        Approx = Approx + (4.0/((2*mode +1)*np.pi*np.sinh((2*mode +1)*np.pi)))*(np.sin((2*mode +1)*np.pi*x))*(np.sinh((2*mode +1)*np.pi*y))
    return Approx

xx, yy = np.meshgrid(xvals,yvals)

ana = analytical_expression(7,xx,yy)

fig2 = plt.figure(num=2,figsize=(5,4))
ax2 = fig2.add_subplot(1,1,1)
im2 = ax2.pcolormesh(xvals,yvals,ana, cmap="jet")
fig2.colorbar(im2,label=r"$\phi$")
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$y$")
fig2.tight_layout()


# Poisson equation
def f_poisson(x,y):
    f = x*np.sin(y)
    return f

xvals,yvals,phi_2 = poisson_scheme_2d([0,1],[0,1], 101,101,[0,0,0,0], f_poisson)

fig3 = plt.figure(num=3,figsize=(5,4))
ax3 = fig3.add_subplot(1,1,1)
im3 = ax3.pcolormesh(xvals,yvals,phi_2, cmap="jet")
fig3.colorbar(im3,label=r"$\phi$")
ax3.set_xlabel(r"$x$")
ax3.set_ylabel(r"$y$")
fig3.tight_layout()


plt.show()