#Question_3
import numpy as np
import matplotlib.pyplot as plt

def second_order_ode(y, t):
    y1, y2 = y
    dy1dt = y2
    dy2dt = 2*y2-y1+t*np.exp(t)-t    
    return [dy1dt, dy2dt]

def rk4_method(func, y0, t):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    h = t[1] - t[0]  # step size

    for i in range(1, n):
        k1 = h * np.array(func(y[i-1], t[i-1]))
        k2 = h * np.array(func(y[i-1] + 0.5 * k1, t[i-1] + 0.5 * h))
        k3 = h * np.array(func(y[i-1] + 0.5 * k2, t[i-1] + 0.5 * h))
        k4 = h * np.array(func(y[i-1] + k3, t[i-1] + h))
        y[i] = y[i-1] + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

    return y

# Initial conditions
y0 = [0.0, 0.0]  # Initial displacement and velocity
t = np.linspace(0, 1, 100)  # Time points

# Solve the ODE using RK4 method
solution = rk4_method(second_order_ode, y0, t)

# Extract displacement and velocity
displacement = solution[:, 0]
velocity = solution[:, 1]

# Plot the results
plt.plot(t, displacement, label='Displacement')
#plt.plot(t, velocity, label='Velocity')
plt.xlabel('Time')
plt.ylabel('Displacement / Velocity')
plt.title('Solution of Second Order ODE using RK4 Method')
plt.legend()
plt.grid()
plt.show()