# problem 9
import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# 1st BVP
def fun1(x, y):
    return np.vstack((y[1], -np.exp(-2*y[0])))

def bc1(ya, yb):
    return np.array([ya[0], yb[0] - np.log(2)])

x1 = np.linspace(1, 2, 10)
y_guess1 = np.zeros((2, x1.size))  # Initial guess for y and y'

sol_1 = solve_bvp(fun1, bc1, x1, y_guess1)
#Analytic solution:
x_1=np.linspace(1,2,100)
sol_analytic_1=np.log(x_1)
plt.subplot(221)
plt.plot(sol_1.x, sol_1.y[0], label='Numerical solution')
plt.plot(x_1,sol_analytic_1,label='Analytic solution')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution to y\'\' = -e^(-2y)')
plt.legend()
plt.grid()
plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#2nd BVP
def fun2(x, y):
    return np.vstack((y[1], y[1] * np.cos(x) - y[0] * np.log(y[0])))
def bc2(ya, yb):
    return np.array([ya[0] - 1, yb[0] - np.exp(1)])

# Define the x values for the solution
x2 = np.linspace(0, np.pi/2, 100)

# Initial guess for the solution
y_guess_2 = np.ones((2, x2.size))

# Solve the boundary value problem
sol_2 = solve_bvp(fun2, bc2, x2, y_guess_2)
# analytic solution
y2=np.exp(np.sin(x2))
# Check if the solution converged
plt.subplot(222)
plt.plot(x2,y2,label="Analytic solution")
plt.plot(sol_2.x, sol_2.y[0], label='Numerical solution')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Solution to y'' = y' cos(x) - y ln(y)")
plt.legend()
plt.grid(True)
plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#3rd BVP
def fun3(x, y):
    return np.vstack((y[1], -(2*(y[1])**3 + y[0]**2 * y[1]) * 1/np.cos(x)))

def bc3(ya, yb):
    return np.array([ya[0] - 2**(-1/4), yb[0] - (12**(1/4))/2])

x3 = np.linspace(np.pi/4, np.pi/3, 100)
y_guess_3 = np.zeros((2, x3.size))  # Initial guess for y and y'

sol_3 = solve_bvp(fun3, bc3, x3, y_guess_3)
y_3=np.sqrt(np.sin(x3))
plt.subplot(223)
plt.plot(x3,y_3,label="Analytic solution")
plt.plot(sol_3.x, sol_3.y[0], label='Numerical solution')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Solution to y'' = -(2(y')^3 + y^2 y') sec(x)")
plt.legend()
plt.grid(True)
plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#4th BVP
def fun4(x, y):
    return np.vstack((y[1], 0.5 - 0.5 * (y[1])**2 - y[0] * np.sin(x)*0.5))

def bc4(ya, yb):
    return np.array([ya[0] - 2, yb[0] - 2])

x4 = np.linspace(0, np.pi, 100)
y_guess_4 = np.zeros((2, x4.size))  # Initial guess for y and y'

sol_4 = solve_bvp(fun4, bc4, x4, y_guess_4)

#Analytic solution
y4=2+np.sin(x4)
plt.subplot(224)
plt.plot(x4,y4,label="Analytic solution")
plt.plot(sol_4.x, sol_4.y[0], label='Numerical solution')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Solution to y'' = 1/2 - (y')^2/2 - y sin(x)/2")
plt.legend()
plt.grid(True)
plt.show()


