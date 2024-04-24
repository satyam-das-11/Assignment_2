# Question_1 Backward integration with Euler's Method
import numpy as np
import matplotlib.pyplot as plt

def backward_euler(f, y0, x):
    n = len(x)
    y = np.zeros(n)
    y[0] = y0
    dx = x[1] - x[0]

    for i in range(1, n):
        y_guess = y[i-1] + dx * f(y[i-1], x[i-1])
        y[i] = y[i-1] + dx * f(y_guess, x[i])

    return y 
#defining the first derivative (for the first problem):
def f(y, x):
    return -9*y
y0_1 = np.exp(1)  # Initial value
#defining the first derivative (for the second problem) :
def g(y,x):
    return -20*(y-x)**2 + 2*x

# x points
x = np.linspace(0, 1, 50)
y0_2 = 1.0/3  # Initial value
# Solve the ODE using backward Euler method:
y_backward_euler_1 = backward_euler(f, y0_1, x)
y_backward_euler_2 = backward_euler(g, y0_2, x)

# Plot the solution
plt.subplot(211)
plt.plot(x, y_backward_euler_1, label='Backward Euler')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('dy/dx=-9y') 
plt.legend()
plt.grid()
plt.show()
plt.subplot(212)
plt.plot(x, y_backward_euler_2, label='Backward Euler')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('dy/dx=-20(y-x)^2+2x') 
plt.legend()
plt.grid()
plt.show()