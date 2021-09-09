import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


"""
This script contains exemplar functions and plots for 1-d systems 
exhibiting bifurcations:

1: Saddle-Node Bifurcation
2: Transcritical Bifurcation
3: Pitchfork Bifurcation 
"""

#Plotting formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')


"""
Case 1: Saddle-Node Bifurcation
"""

#Plot exemplar solutions
def case_1(t,y,r):
    x = y[0]
    dxdt = x**2 + r
    return dxdt

sol1 = solve_ivp(lambda t,x:case_1(t,x,1.0),[0,1.0],[0.5],method="LSODA",dense_output=True)
sol0 = solve_ivp(lambda t,x:case_1(t,x,1.0),[0,1.8],[-0.5],method="LSODA",dense_output=True)
sol2 = solve_ivp(lambda t,x:case_1(t,x,0.0),[0,1.8],[0.5],method="LSODA",dense_output=True)
sol3 = solve_ivp(lambda t,x:case_1(t,x,0.0),[0,2.0],[-2.5],method="LSODA",dense_output=True)
sol4 = solve_ivp(lambda t,x:case_1(t,x,-1.0),[0,1.3],[1.1],method="LSODA",dense_output=True)
sol5 = solve_ivp(lambda t,x:case_1(t,x,-1.0),[0,2.0],[-2.5],method="LSODA",dense_output=True)
sol6 = solve_ivp(lambda t,x:case_1(t,x,-1.0),[0,2.0],[0.5],method="LSODA",dense_output=True)

fig1 = plt.figure(num=1,figsize=(5,4))
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(sol1.t, sol1.y[0],label=r"$r=1.0,x_{0} = 0.5$")
ax1.plot(sol0.t, sol0.y[0],label=r"$r=1.0,x_{0} = -0.5$")
ax1.plot(sol2.t, sol2.y[0],label=r"$r=0.0,x_{0} = 0.5$")
ax1.plot(sol3.t, sol3.y[0],label=r"$r=0.0,x_{0} = -2.5$")
ax1.plot(sol4.t, sol4.y[0],label=r"$r=-1.0,x_{0} = 1.1$")
ax1.plot(sol5.t, sol5.y[0],label=r"$r=-1.0,x_{0} = -2.5$")
ax1.plot(sol6.t, sol6.y[0],label=r"$r=-1.0,x_{0} = 0.5$")
ax1.set_xlabel(r"$t$")
ax1.set_ylabel(r"$x$")
ax1.legend()
fig1.tight_layout()

#Plot bifurcation diagram
r_vals = np.linspace(-10.0,0.0,1000)
x_crit_1 = np.sqrt(-r_vals)
x_crit_2 = -np.sqrt(-r_vals)

fig2 = plt.figure(num=2,figsize=(5,4))
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(r_vals,x_crit_1,ls="--",label="Unstable")
ax2.plot(r_vals,x_crit_2,label="Stable")
ax2.set_xlabel(r"$r$")
ax2.set_ylabel(r"$x$")
ax2.set_xlim(-10,10)
ax2.legend()
fig2.tight_layout()


"""
Case 2: Transcritical Bifurcation
"""

#Plot phase portrait for different values of r

def case_2(t,y,r):
    x = y[0]
    dxdt = r*x - x**2
    return dxdt

def case_2_fun(x,r):
    f = r*x - x**2
    return f

x_vals = np.linspace(-3,3,1000)

fig3 = plt.figure(num=3,figsize=(5,4))
ax3 = fig3.add_subplot(1,1,1)
ax3.plot(x_vals,case_2_fun(x_vals,-2.0), label=r"$r=-2.0$")
ax3.plot(x_vals,case_2_fun(x_vals,0.0), label=r"$r=0.0$")
ax3.plot(x_vals,case_2_fun(x_vals,2.0), label=r"$r=2.0$")
ax3.plot(x_vals,np.zeros(1000),"k")
ax3.set_xlabel(r"$x$")
ax3.set_ylabel(r"$\frac{dx}{dt} = f(x)$")
ax3.legend()
ax3.set_ylim(-2,1.8)
fig3.tight_layout()

#plot bifurcation diagram

trans_crit_1 = np.zeros(1000)
trans_crit_2 = np.linspace(0,3,1000)
trans_crit_3 = np.linspace(-3,0,1000)

fig4 = plt.figure(num=4,figsize=(5,4))
ax4 = fig4.add_subplot(1,1,1)
ax4.plot(np.linspace(-3,0,1000),trans_crit_1,"k",label=r"$x^{*} = 0$, Stable")
ax4.plot(np.linspace(0,3,1000),trans_crit_1,"--k",label=r"$x^{*} = 0$, Unstable")
ax4.plot(np.linspace(0,3,1000),trans_crit_2, "r",label=r"$x^{*} = r$, Stable")
ax4.plot(np.linspace(-3,0,1000),trans_crit_3, "--r",label=r"$x^{*} = r$, Unstable")
ax4.set_xlabel(r"$r$")
ax4.set_ylabel(r"$x$")
ax4.legend()
fig4.tight_layout()



"""
Laser example
"""
k = 1.0
alpha = 1.0
G = 1.0

def laser_f (n,k,alpha,G, N0):
    f = (G*N0 - k)*n - alpha*G*n**2
    return f

n_vals = np.linspace(0,2,1000)

laser_sub_crit = laser_f(n_vals, k,alpha,G,0.05)
laser_crit = laser_f(n_vals, k, alpha, G, 1.0)
laser_super_crit = laser_f(n_vals,k,alpha,G,2.0)

# Phase portrait
fig5 = plt.figure(num=5,figsize=(5,4))
ax5 = fig5.add_subplot(1,1,1)
ax5.plot(n_vals, laser_sub_crit, label=r"$N_{0} < \frac{k}{G}$")
ax5.plot(n_vals,laser_crit, label=r"$N_{0} = \frac{k}{G}$")
ax5.plot(n_vals, laser_super_crit, label=r"$N_{0} > \frac{k}{G}$")
ax5.legend()
ax5.set_xlabel(r"$n$")
ax5.set_ylabel(r"$\frac{dn}{dt} = f(n)$")
ax5.tick_params(axis="x",which="both",bottom=False,top=False,labelbottom=False)
ax5.axhline(color="black")
fig5.gca().spines[["bottom"]].set_position('center')
fig5.gca().spines[["top","right"]].set_visible(False)
fig5.tight_layout()

# Bifurcation plot
laser_crit_1 = np.zeros(1000)
laser_crit_2 = np.linspace(0,3,1000)

fig6 = plt.figure(num=6,figsize=(5,4))
ax6 = fig6.add_subplot(1,1,1)
ax6.plot(np.linspace(0,1,1000),laser_crit_1, "k", label=r"$n^{*} = 0$, Stable")
ax6.plot(np.linspace(1,3,1000),laser_crit_2,"r",label=r"$n^{*} = \frac{N_{0}}{\alpha} - \frac{k}{\alpha G}$, Stable")
ax6.plot(np.linspace(1,3,1000),laser_crit_1,"--k", label=r"$n^{*} = 0$, Unstable")
ax6.set_xlabel(r"$N_{0}$")
ax6.set_ylabel(r"$n$")
ax6.set_xticks([1.0])
ax6.set_xticklabels([r"$\frac{k}{G}$"])
ax6.set_yticks([0.0])
ax6.set_yticklabels([r"$0.0$"])
ax6.legend()
fig6.tight_layout()

"""
Pitchfork bifurcation
"""
def supercritical(x,r):
    f = r*x - x**3
    return f

def subcritical(x,r):
    f = r*x + x**3
    return f

x_vals = np.linspace(-2,2,1000)

# Supercritical phase plot
fig7 = plt.figure(num=7, figsize=(5,4))
ax7 = fig7.add_subplot(1,1,1)
ax7.plot(x_vals, supercritical(x_vals,-3), label=r"$r = -3$")
ax7.plot(x_vals, supercritical(x_vals,0), label=r"$r = 0$")
ax7.plot(x_vals, supercritical(x_vals,3), label=r"$r = 3$")
ax7.axhline(color="black")
ax7.set_xlabel(r"$x$")
ax7.set_ylabel(r"$\frac{dx}{dt} = f(x)$")
ax7.legend()
fig7.tight_layout()

# Subcritical phase plot
fig8 = plt.figure(num=8, figsize=(5,4))
ax8 = fig8.add_subplot(1,1,1)
ax8.plot(x_vals, subcritical(x_vals,-3), label=r"$r = -3$")
ax8.plot(x_vals, subcritical(x_vals,0), label=r"$r = 0$")
ax8.plot(x_vals, subcritical(x_vals,3), label=r"$r = 3$")
ax8.axhline(color="black")
ax8.set_xlabel(r"$x$")
ax8.set_ylabel(r"$\frac{dx}{dt} = f(x)$")
ax8.legend()
fig8.tight_layout()

r_vals_1 = np.linspace(-3,0,1000)
r_vals_2 = np.linspace(0,3,1000)
#Supercritical bifurcation diagram
fig9 = plt.figure(num=9, figsize=(5,4))
ax9 = fig9.add_subplot(1,1,1)
ax9.plot(r_vals_1, np.zeros(1000), "k", label=r"$x^{*} = 0$, Stable")
ax9.plot(r_vals_2, np.zeros(1000), "--k", label=r"$x^{*} = 0$, Unstable")
ax9.plot(r_vals_2, np.sqrt(r_vals_2), "r", label=r"$x^{*}=\sqrt{r}$, Stable")
ax9.plot(r_vals_2, -np.sqrt(r_vals_2), "r", label=r"$x^{*} = -\sqrt{r}$, Stable")
ax9.set_xlabel(r"$r$")
ax9.set_ylabel(r"$x$")
ax9.legend()
fig9.tight_layout()

#Subcritical bifurcation diagram
fig10 = plt.figure(num=10, figsize=(5,4))
ax10 = fig10.add_subplot(1,1,1)
ax10.plot(r_vals_1, np.zeros(1000), "k", label=r"$x^{*} = 0$, Stable")
ax10.plot(r_vals_2, np.zeros(1000), "--k", label=r"$x^{*} = 0$, Unstable")
ax10.plot(r_vals_1, np.sqrt(-r_vals_1), "--r", label=r"$x^{*}=\sqrt{-r}$, Unstable")
ax10.plot(r_vals_1, -np.sqrt(-r_vals_1), "--r", label=r"$x^{*} = -\sqrt{-r}$, Unstable")
ax10.set_xlabel(r"$r$")
ax10.set_ylabel(r"$x$")
ax10.legend()
fig10.tight_layout()




plt.show()