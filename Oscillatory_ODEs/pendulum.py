import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

"""
This script contains functions to solve the pendulum model for 
the following cases:

1. Frictionless pendulum given by x'' + (g/l)sin(x) = 0, where x 
is the angular displacement, g is the magnitutde of the gravitational field
and l is the length of the pendulum. We also consider the small angle 
approximation where sin(x) ~ x for small values of x. 

2. Damped Pendulum given by x'' + b


"""
#Plotting formatting
plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')

"""
Part 1
"""

def frictionless (t,y,g,l):
    theta = y[0]
    thetadot = y[1]
    #Define system of ODEs
    dthetadt = thetadot
    dthetadotdt = -(g/l)*np.sin(theta)
    return [dthetadt, dthetadotdt]

def frictionless_small (t,y,g,l):
    theta = y[0]
    thetadot = y[1]
    #Define system of ODEs
    dthetadt = thetadot
    dthetadotdt = -(g/l)*theta
    return [dthetadt, dthetadotdt]

#Define parameters and solution
theta0_1 = [np.pi/9,0.0]
theta0_2 = [np.pi-0.1,0.0]
g = 9.81
l = 1.0

sol_1 = solve_ivp(lambda t,x:frictionless(t,x,g,l),[0,4*np.pi],theta0_1,method="LSODA",dense_output=True)
sol_2 = solve_ivp(lambda t,x:frictionless_small(t,x,g,l),[0,4*np.pi],theta0_1,method="LSODA",dense_output=True)
sol_3 = solve_ivp(lambda t,x:frictionless(t,x,g,l),[0,4*np.pi],theta0_2,method="LSODA",dense_output=True)
sol_4 = solve_ivp(lambda t,x:frictionless_small(t,x,g,l),[0,4*np.pi],theta0_2,method="LSODA",dense_output=True)

#Plotting Results
fig1 = plt.figure(num=1,figsize=(8,6))
ax1 = fig1.add_subplot(2,1,1)
ax1.plot(sol_1.t, sol_1.y[0],label="Full")
ax1.plot(sol_2.t, sol_2.y[0],label="Small Angle")
ax1.set_xlabel(r"$t$")
ax1.set_ylabel(r"$\theta$")
ax1.set_ylim(-1.5,2.5)
ax1.legend()
ax1.set_title(r"$\theta_{0} = \frac{\pi}{9}, \dot{\theta}_{0} = 0.0$")
ax2 = fig1.add_subplot(2,1,2)
ax2.plot(sol_3.t, sol_3.y[0],label="Full")
ax2.plot(sol_4.t, sol_4.y[0],label="Small Angle")
ax2.set_xlabel(r"$t$")
ax2.set_ylabel(r"$\theta$")
ax2.set_ylim(-4,5.5)
ax2.set_title(r"$\theta_{0} = \pi - 0.1, \dot{\theta}_{0} = 0.0$")
ax2.legend()
fig1.tight_layout()

#Animation
# fig2 = plt.figure(num=2, figsize=(5,5))
# ax3 = fig2.add_subplot(111,autoscale_on=False, xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))

# x_vals = l*np.sin(sol_1.y[0])
# y_vals = -l*np.cos(sol_1.y[0])
# t_vals = sol_1.t

# line, = ax3.plot([],[],"-o",lw=2)
# time_template = "time = %.2fs"
# time_text = ax3.text(0.05,0.9,"",transform=ax3.transAxes)

# def init():
#     line.set_data([], [])
#     time_text.set_text('')
#     return line, time_text

# def animate(i):
#     thisx = [0, x_vals[i]]
#     thisy = [0, y_vals[i]]

#     line.set_data(thisx, thisy)
#     time_text.set_text(time_template % (t_vals[i]))
#     return line, time_text

# ani = animation.FuncAnimation(fig2, animate,
#                               interval= 10 ,blit=True,init_func=init)

theta0_3 = [np.pi/3,5.45]
sol_5 = solve_ivp(lambda t,x:frictionless(t,x,g,l),[0,4*np.pi],theta0_3,method="LSODA",dense_output=True)
sol_6 = solve_ivp(lambda t,x:frictionless_small(t,x,g,l),[0,4*np.pi],theta0_3,method="LSODA",dense_output=True)

fig3 = plt.figure(num=3,figsize=(8,3))
ax4 = fig3.add_subplot(1,1,1)
ax4.plot(sol_5.t, sol_5.y[0],label="Full")
ax4.plot(sol_6.t, sol_6.y[0],label="Small Angle")
ax4.set_xlabel(r"$t$")
ax4.set_ylabel(r"$\theta$")
ax4.legend()
fig3.tight_layout()

"""
Part 2
"""

def damped (t,y,g,l,b):
    theta = y[0]
    thetadot = y[1]
    #Define system of ODEs
    dthetadt = thetadot
    dthetadotdt = -b*thetadot -(g/l)*np.sin(theta)
    return [dthetadt, dthetadotdt]

sol_7 = solve_ivp(lambda t,x:damped(t,x,g,l,1.0),[0,4*np.pi],[np.pi/1.1,1.0],method="LSODA",dense_output=True)
sol_8 = solve_ivp(lambda t,x:damped(t,x,g,l,8.0),[0,4*np.pi],[np.pi/1.1,1.0],method="LSODA",dense_output=True)
sol_9 = solve_ivp(lambda t,x:damped(t,x,g,l,0.5),[0,4*np.pi],[np.pi/1.1,1.0],method="LSODA",dense_output=True)

fig4 = plt.figure(num=4,figsize=(8,3))
ax5 = fig4.add_subplot(1,1,1)
ax5.plot(sol_9.t, sol_9.y[0],label=r"$b=0.5$")
ax5.plot(sol_7.t, sol_7.y[0],label=r"$b=1.0$")
ax5.plot(sol_8.t, sol_8.y[0],label=r"$b=8.0$")
ax5.set_xlabel(r"$t$")
ax5.set_ylabel(r"$\theta$")
ax5.legend()
fig4.tight_layout()


# Animation
# fig5 = plt.figure(num=5, figsize=(5,5))
# ax6 = fig5.add_subplot(111,autoscale_on=False, xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))

# x_vals = l*np.sin(sol_8.y[0])
# y_vals = -l*np.cos(sol_8.y[0])
# t_vals = sol_8.t

# line, = ax6.plot([],[],"-o",lw=2)
# time_template = "time = %.2fs"
# time_text = ax6.text(0.05,0.9,"",transform=ax6.transAxes)

# def init():
#     line.set_data([], [])
#     time_text.set_text('')
#     return line, time_text

# def animate(i):
#     thisx = [0, x_vals[i]]
#     thisy = [0, y_vals[i]]

#     line.set_data(thisx, thisy)
#     time_text.set_text(time_template % (t_vals[i]))
#     return line, time_text

# ani = animation.FuncAnimation(fig5, animate,
#                               interval= 10 ,blit=True,init_func=init)


plt.show()