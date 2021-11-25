import numpy as np
import matplotlib.pyplot as plt

#formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

"""
Plane Poiseuille Flow
"""

def plane_poiseuille_solver(mu, dpdx, h, ncells):
    # Compute dy
    dy = h/(ncells-1)
    #Construct matrix
    A = np.zeros((ncells,ncells))
    b = np.full(ncells, dpdx*(1/mu))
    #specify boundary conditions
    A[0,0] = 1
    b[0] = 0
    A[-1,-1] = 1
    b[-1] = 0
    #Build rest of matrix
    for i in range(1,ncells-1):
        A[i,i-1] = 1/(dy**2)
        A[i,i] = -2/(dy**2)
        A[i, i+1] = 1/(dy**2)
    #solve matrix 
    cond = np.linalg.cond(A)
    ux = np.linalg.solve(A,b)
    return ux,cond

# Solve numerical example
dpdx = -10
h = 1.0
mu = 1.0
ncells = 21
u_x_1, cond = plane_poiseuille_solver(mu, dpdx, h, ncells)

#Define analytical solution
def analytical_poiseuille(mu,dpdx,h,y):
    ux = (1/mu)*(dpdx)*(y/2)*(y-h)
    return ux


#plot numerical and analytical solution 
mesh = np.linspace(0,h,ncells)
ana = analytical_poiseuille(mu,dpdx,h,mesh)

fig1 = plt.figure(num=1, figsize=(5,4))
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(mesh,u_x_1, label="Numerical")
ax1.plot(mesh,ana,"x",label="Analytical")
ax1.legend()
ax1.set_xlabel(r"$y$")
ax1.set_ylabel(r"$u_{x}$")
fig1.tight_layout()


# Compute error
ncell_list = [21,51,101,151,201,251,401,501,751,1001,1501,2001]
rmse_vals = np.zeros(len(ncell_list))
cond_vals = np.zeros_like(rmse_vals)

for i in range(len(ncell_list)):
    ncell = ncell_list[i]
    numerical_soln,cond = plane_poiseuille_solver(mu,dpdx,h,ncell)
    analytical = analytical_poiseuille(mu,dpdx,h,np.linspace(0,h,ncell))
    #Compute RMSE
    rmse = np.sqrt(((analytical-numerical_soln)**2).mean())
    rmse_vals[i] = rmse
    cond_vals[i] = cond

fig2 = plt.figure(num=2, figsize=(5,4))
color = 'tab:red'
ax2 = fig2.add_subplot(1,1,1)
ax2.loglog(ncell_list,rmse_vals, color=color)
ax2.set_xlabel(r"$N_{cells}$")
ax2.set_ylabel("RMSE",color=color)
ax3 = ax2.twinx()
color = 'tab:blue'
ax3.loglog(ncell_list,cond_vals, color=color)
ax3.set_ylabel("Condition Number of "+r"$\mathbf{A}$",color=color)
fig2.tight_layout()

"""
Transport in Catalyst Pellet
"""

def catalyst_solver(k, D, L, c0, ncells):
    # Compute dy
    dx = L/(ncells-1)
    #Construct matrix
    A = np.zeros((ncells,ncells))
    b = np.zeros(ncells)
    #specify boundary conditions
    A[0,0] = 1
    b[0] = c0
    A[-1,-1] = -2 - (k/D)*(dx**2)
    A[-1,-2] = 2
    #Build rest of matrix
    for i in range(1,ncells-1):
        A[i,i-1] = 1
        A[i,i] = -2 - (k/D)*(dx**2)
        A[i, i+1] = 1
    #solve matrix 
    cond = np.linalg.cond(A)
    c = np.linalg.solve(A,b)
    return c,cond

#Analytical solution
def catalyst_analytical(k,D,L,c0,x):
    g = (k/D)**0.5
    c = (c0/(1+np.exp(2*g*L)))*np.exp(g*x) + (c0/(1+np.exp(-2*g*L)))*np.exp(-g*x)
    return c

#solve numerical example
k = 1.0
D = 1.0
c0 = 1.0
L = 1.0

mesh2 = np.linspace(0,L,ncells)
c,cond = catalyst_solver(k,D,L,c0,ncells)
ana = catalyst_analytical(k,D,L,c0,mesh2)

#Plotting
fig3 = plt.figure(num=3, figsize=(5,4))
ax4 = fig3.add_subplot(1,1,1)
ax4.plot(mesh2,c, label="Numerical")
ax4.plot(mesh2,ana,"x",label="Numerical")
ax4.legend()
ax4.set_xlabel(r"$x$")
ax4.set_ylabel(r"$c$")
fig3.tight_layout()

# Compute error
rmse_vals2 = np.zeros(len(ncell_list))
cond_vals2 = np.zeros_like(rmse_vals)

for i in range(len(ncell_list)):
    ncell = ncell_list[i]
    numerical_soln2,cond2 = catalyst_solver(k,D,L,c0,ncell)
    analytical2 = catalyst_analytical(k,D,L,c0,np.linspace(0,L,ncell))
    #Compute RMSE
    rmse2 = np.sqrt(((analytical2-numerical_soln2)**2).mean())
    rmse_vals2[i] = rmse2
    cond_vals2[i] = cond2

fig4 = plt.figure(num=4, figsize=(5,4))
color = 'tab:red'
ax4 = fig4.add_subplot(1,1,1)
ax4.loglog(ncell_list,rmse_vals2, color=color)
ax4.set_xlabel(r"$N_{cells}$")
ax4.set_ylabel("RMSE",color=color)
ax5 = ax4.twinx()
color = 'tab:blue'
ax5.loglog(ncell_list,cond_vals2, color=color)
ax5.set_ylabel("Condition Number of "+r"$\mathbf{A}$",color=color)
fig4.tight_layout()



plt.show()


